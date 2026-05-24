"""Always-on background perception loop.

A pure ``asyncio.Task`` driver: one tick captures a frame, runs the
pipeline, upserts each detection into the store, emits a
:class:`TickReport`, then ``await asyncio.sleep(...)`` until the next
tick. Cancel via ``task.cancel()`` — the loop's ``try/finally`` block
guarantees an in-progress upsert finishes before exit.

Sequential ticks: a slow tick does not pile up. We always ``await``
between ticks, but we measure the tick duration and skip the sleep if
the tick already exceeded ``interval_sec``.

All blocking work (HTTP, ChromaDB write) is wrapped in
``asyncio.to_thread`` so the same event loop can drive the agent at the
same time without starving.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Awaitable, Callable, Optional

from PIL import Image

from .pipeline import process_frame
from .store import SceneStore
from .types import (
    CameraSource,
    Captioner,
    Detector,
    Embedder,
    PositionLifter,
    TickReport,
)


_log = logging.getLogger("perception.loop")


async def run_scene_perception(
    *,
    camera: CameraSource,
    detector: Detector,
    captioner: Captioner,
    embedder: Embedder,
    lifter: PositionLifter,
    store: SceneStore,
    interval_sec: float = 2.0,
    position_timeout: float = 2.0,
    min_confidence: float = 0.0,
    caption_per_object: bool = False,
    archive_source_frame: bool = True,
    prune_ttl_sec: Optional[float] = None,
    prune_interval_sec: float = 30.0,
    prune_radius_m: Optional[float] = None,
    prune_max_records: Optional[int] = None,
    pose_provider: Optional[Callable[[], Optional[tuple[float, float, float]]]] = None,
    on_tick: Optional[Callable[[TickReport], None]] = None,
) -> None:
    """Run the perception loop until the surrounding task is cancelled.

    ``on_tick`` is invoked synchronously inside the loop after each tick —
    keep it cheap. For prometheus/file logging, prefer subscribing to the
    ``perception.loop`` logger; the loop emits one structured INFO line
    per tick on its own.

    Eviction: when ``prune_ttl_sec`` (and/or ``prune_max_records``) is set,
    every ``prune_interval_sec`` the loop calls :meth:`SceneStore.prune` so
    objects that have been removed from the world stop lingering in the store.
    With ``prune_radius_m`` and a ``pose_provider`` the TTL sweep is gated to
    the robot's current vicinity — out-of-view objects elsewhere are never
    wrongly evicted (see :meth:`SceneStore.prune`).
    """
    _log.info(
        "perception loop start interval=%.2fs caption_per_object=%s "
        "prune_ttl=%s prune_radius=%s prune_every=%.1fs",
        interval_sec,
        caption_per_object,
        prune_ttl_sec,
        prune_radius_m,
        prune_interval_sec,
    )
    tick_idx = 0
    prune_enabled = prune_ttl_sec is not None or prune_max_records is not None
    last_prune = time.monotonic()
    try:
        while True:
            tick_start = time.perf_counter()
            try:
                report = await _run_one_tick(
                    camera=camera,
                    detector=detector,
                    captioner=captioner,
                    embedder=embedder,
                    lifter=lifter,
                    store=store,
                    position_timeout=position_timeout,
                    min_confidence=min_confidence,
                    caption_per_object=caption_per_object,
                    archive_source_frame=archive_source_frame,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  (we want broad isolation here)
                _log.exception("perception tick raised; continuing loop")
                report = TickReport(
                    ts=time.time(),
                    n_detections=0,
                    n_inserts=0,
                    n_updates=0,
                    n_skipped=0,
                    error=str(e),
                )

            if prune_enabled and (time.monotonic() - last_prune) >= prune_interval_sec:
                try:
                    n_pruned = await asyncio.to_thread(
                        _run_prune,
                        store,
                        ttl_sec=prune_ttl_sec,
                        max_records=prune_max_records,
                        radius_m=prune_radius_m,
                        pose_provider=pose_provider,
                    )
                    if n_pruned:
                        report = replace(report, n_pruned=n_pruned)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — pruning must never kill the loop
                    _log.exception("scene prune failed; continuing loop")
                finally:
                    last_prune = time.monotonic()

            _emit_log(report)
            if on_tick is not None:
                try:
                    on_tick(report)
                except Exception:  # pragma: no cover  (user callback hygiene)
                    _log.exception("on_tick callback raised; suppressing")

            tick_idx += 1
            elapsed = time.perf_counter() - tick_start
            remaining = max(0.0, interval_sec - elapsed)
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                # Yield to the event loop even when we're behind, so cancel
                # can deliver and other tasks make progress.
                await asyncio.sleep(0)
    except asyncio.CancelledError:
        _log.info("perception loop cancelled after %d tick(s)", tick_idx)
        raise
    finally:
        _log.info("perception loop stopped (ticks=%d)", tick_idx)


async def _run_one_tick(
    *,
    camera: CameraSource,
    detector: Detector,
    captioner: Captioner,
    embedder: Embedder,
    lifter: PositionLifter,
    store: SceneStore,
    position_timeout: float,
    min_confidence: float,
    caption_per_object: bool,
    archive_source_frame: bool,
) -> TickReport:
    t0 = time.perf_counter()
    frame: Image.Image = await asyncio.to_thread(camera.capture_pil)
    capture_ms = (time.perf_counter() - t0) * 1000

    # process_frame is sync but calls into HTTP under the hood — keep it
    # off the event loop. Returned latency dict excludes capture+store.
    detections, latency = await asyncio.to_thread(
        process_frame,
        frame,
        detector=detector,
        lifter=lifter,
        captioner=captioner,
        embedder=embedder,
        position_timeout=position_timeout,
        min_confidence=min_confidence,
        caption_per_object=caption_per_object,
    )

    t0 = time.perf_counter()
    n_inserts = 0
    n_updates = 0
    n_skipped = 0
    for det in detections:
        try:
            _, decision = await asyncio.to_thread(
                store.upsert,
                det,
                source_frame=frame if archive_source_frame else None,
            )
            if decision.action == "insert":
                n_inserts += 1
            elif decision.action == "update":
                n_updates += 1
        except Exception:  # noqa: BLE001
            _log.exception("upsert failed for detection class=%s", det.class_name)
            n_skipped += 1
    store_ms = (time.perf_counter() - t0) * 1000

    latency_ms = {"capture": capture_ms, "store": store_ms, **latency}
    return TickReport(
        ts=time.time(),
        n_detections=len(detections),
        n_inserts=n_inserts,
        n_updates=n_updates,
        n_skipped=n_skipped,
        latency_ms=latency_ms,
    )


def _run_prune(
    store: SceneStore,
    *,
    ttl_sec: Optional[float],
    max_records: Optional[int],
    radius_m: Optional[float],
    pose_provider: Optional[Callable[[], Optional[tuple[float, float, float]]]],
) -> int:
    """Resolve the spatial gate, then run one :meth:`SceneStore.prune`.

    When a ``radius_m`` gate is requested but the robot's pose is currently
    unknown, the TTL sweep is *skipped* this cycle rather than run globally —
    a global TTL prune would wrongly evict objects the robot saw elsewhere.
    A ``max_records`` cap still applies (it's position-independent).
    """
    within = None
    ttl = ttl_sec
    if ttl is not None and radius_m is not None:
        pose = pose_provider() if pose_provider is not None else None
        if pose is None:
            _log.debug("scene.prune: no pose for spatial gate; skipping TTL this cycle")
            ttl = None
        else:
            within = (pose, radius_m)
    if ttl is None and max_records is None:
        return 0
    return store.prune(ttl_sec=ttl, max_records=max_records, within=within)


def _emit_log(report: TickReport) -> None:
    if report.error is not None:
        _log.error(
            "tick error ts=%.3f err=%r",
            report.ts,
            report.error,
        )
        return
    _log.info(
        "tick ts=%.3f n_det=%d ins=%d upd=%d skip=%d prune=%d latency_ms=%s",
        report.ts,
        report.n_detections,
        report.n_inserts,
        report.n_updates,
        report.n_skipped,
        report.n_pruned,
        {k: round(v, 1) for k, v in report.latency_ms.items()},
    )
