"""Thread adapter for the async scene-perception loop.

``main.py`` is synchronous and the rest of ``services/`` is built on
``threading.Thread``; the CLIP perception loop in ``perception.loop`` is an
``asyncio`` coroutine. This wraps it so it presents the same
``start()`` / ``stop_and_join()`` surface as ``PerceptionService``.

The coroutine runs on a private event loop owned by this thread. Stopping
cancels the task from the caller's thread via ``call_soon_threadsafe`` and
relies on the loop's ``try/finally`` to flush an in-progress upsert.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional, Sequence

from interfaces.walkie_interface import WalkieInterface
from perception import SceneStore, run_scene_perception
from perception.types import Embedder, PositionLifter


_log = logging.getLogger("services.scene_perception")


class ScenePerceptionService(threading.Thread):
    """Run :func:`perception.run_scene_perception` in a daemon thread.

    Collaborators are pulled from the existing ``walkie`` / ``walkieAI``
    objects, so this service owns nothing the rest of the app doesn't
    already have — it just orchestrates them on a background event loop.
    """

    def __init__(
        self,
        walkieAI,
        walkie: WalkieInterface,
        store: SceneStore,
        embedder: Embedder,
        *,
        lifter: Optional[PositionLifter] = None,
        interval: float = 2.0,
        position_timeout: float = 2.0,
        min_confidence: float = 0.0,
        caption_per_object: bool = False,
        archive_source_frame: bool = True,
        exclude_classes: Optional[Sequence[str]] = None,
        position_fallback_to_pose: bool = True,
        prune_ttl_sec: Optional[float] = None,
        prune_interval_sec: float = 30.0,
        prune_radius_m: Optional[float] = None,
        prune_max_records: Optional[int] = None,
        max_lift_distance_m: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True, name="ScenePerceptionService")
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.store = store
        self.embedder = embedder
        # Default to the SDK depth+TF lifter; callers can inject a coarser one
        # (e.g. RobotPoseLifter) when get_3d_poses is unavailable.
        self.lifter = lifter if lifter is not None else walkie.tools
        self.interval = interval
        self.position_timeout = position_timeout
        self.min_confidence = min_confidence
        self.caption_per_object = caption_per_object
        self.archive_source_frame = archive_source_frame
        self.exclude_classes = exclude_classes
        self.position_fallback_to_pose = position_fallback_to_pose
        self.prune_ttl_sec = prune_ttl_sec
        self.prune_interval_sec = prune_interval_sec
        self.prune_radius_m = prune_radius_m
        self.prune_max_records = prune_max_records
        self.max_lift_distance_m = max_lift_distance_m
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._ready = threading.Event()

    def _current_pose(self) -> Optional[tuple[float, float, float]]:
        """Robot's current planar map pose, for the prune spatial gate.

        Mirrors :class:`perception.lifters.RobotPoseLifter`: only ``x``/``y``
        are used (odometry is planar). Returns ``None`` when no pose is
        available yet, which tells the loop to skip the gated TTL sweep this
        cycle rather than evict objects on stale coordinates.
        """
        try:
            pose = self.walkie.status.get_position()
        except Exception:  # noqa: BLE001 — telemetry hiccup shouldn't crash prune
            return None
        if not pose:
            return None
        return (float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), 0.0)

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._task = loop.create_task(
            run_scene_perception(
                camera=self.walkie.camera,
                detector=self.walkieAI.object_detection,
                captioner=self.walkieAI.image_caption,
                embedder=self.embedder,
                lifter=self.lifter,
                store=self.store,
                interval_sec=self.interval,
                position_timeout=self.position_timeout,
                min_confidence=self.min_confidence,
                caption_per_object=self.caption_per_object,
                archive_source_frame=self.archive_source_frame,
                exclude_classes=self.exclude_classes,
                prune_ttl_sec=self.prune_ttl_sec,
                prune_interval_sec=self.prune_interval_sec,
                prune_radius_m=self.prune_radius_m,
                prune_max_records=self.prune_max_records,
                pose_provider=self._current_pose,
                position_fallback_to_pose=self.position_fallback_to_pose,
                max_lift_distance_m=self.max_lift_distance_m,
            )
        )
        # Signal that _loop/_task are set, so a fast stop_and_join can find them.
        self._ready.set()
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — log, don't take down the process
            _log.exception("scene perception loop crashed")
        finally:
            loop.close()

    def stop_and_join(self, timeout: float | None = None) -> None:
        # Wait briefly for run() to publish the loop/task before cancelling,
        # so an immediate stop after start() doesn't no-op.
        self._ready.wait(timeout=2.0)
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            loop.call_soon_threadsafe(task.cancel)
        self.join(timeout)
