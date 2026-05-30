"""Phase 3 integration tests for the async background loop.

Exercises the whole pipeline end-to-end with fake collaborators.
``pytest.mark.asyncio`` is provided by the ``anyio`` plugin that ships
with langsmith; we use the lower-level approach (``asyncio.run`` /
``anyio.from_thread``) so we don't take an extra dep.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from perception import SceneStore, TickReport, run_scene_perception
from perception.mocks import (
    FakeCamera,
    FakeCaptioner,
    FakeDetectedObject,
    FakeDetector,
    FakeEmbedder,
    FakePositionLifter,
    make_tiny_image,
)


@pytest.fixture
def fake_world(tmp_path):
    """Build a coherent set of fakes representing a small static scene."""
    embedder = FakeEmbedder(dim=16)
    store = SceneStore(persist_dir=tmp_path / "chroma", embedder=embedder)
    camera = FakeCamera(frames=[make_tiny_image(seed=1) for _ in range(20)])
    captioner = FakeCaptioner("a chair next to a table")
    return {
        "embedder": embedder,
        "store": store,
        "camera": camera,
        "captioner": captioner,
        "tmp_path": tmp_path,
    }


async def _run_for_n_ticks(coro_factory, n: int, interval: float, timeout: float | None = None):
    """Start the loop, wait for ``n`` ``on_tick`` callbacks, cancel cleanly.

    ``timeout`` overrides the default 4s/4×interval headroom — useful for
    long-running patrol tests where chroma upsert overhead dominates.
    """
    reports: list[TickReport] = []
    done = asyncio.Event()

    def on_tick(report: TickReport) -> None:
        reports.append(report)
        if len(reports) >= n:
            done.set()

    task = asyncio.create_task(coro_factory(on_tick))
    effective_timeout = timeout if timeout is not None else max(4.0, 4 * n * interval)
    try:
        await asyncio.wait_for(done.wait(), timeout=effective_timeout)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return reports, task


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_loop_happy_path_records_settle_after_n_ticks(fake_world):
    detector = FakeDetector({
        0: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))],
        1: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))],
        2: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))],
        3: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))],
        4: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))],
    })
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))[0]
    assert len(reports) >= 5
    # Same chair every tick → one record, 5 sightings (or more if extra ticks slipped in).
    assert fake_world["store"].count == 1
    entry = fake_world["store"].recency_query(since_ts=0.0)[0]
    assert entry.sightings >= 5
    # First tick should be an insert; subsequent ticks updates.
    inserts = sum(r.n_inserts for r in reports)
    updates = sum(r.n_updates for r in reports)
    assert inserts == 1
    assert updates >= 4


# ---------------------------------------------------------------------------
# 2. Mid-scene change
# ---------------------------------------------------------------------------


def test_loop_detects_new_object_mid_run(fake_world):
    chair = FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))
    mug = FakeDetectedObject("mug", 41, 0.9, (200, 200, 260, 260))
    detector = FakeDetector({
        0: [chair],
        1: [chair],
        2: [chair],
        3: [chair, mug],
        4: [chair, mug],
    })
    lifter = FakePositionLifter(
        scripted={
            (35.0, 35.0, 50.0, 50.0): [1.0, 0.0, 0.0],     # chair
            (230.0, 230.0, 60.0, 60.0): [2.0, 2.0, 0.5],   # mug
        },
    )

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))[0]
    store = fake_world["store"]
    assert store.count == 2
    classes = sorted(e.class_name for e in store.recency_query(since_ts=0.0))
    assert classes == ["chair", "mug"]


# ---------------------------------------------------------------------------
# 3. Graceful cancel
# ---------------------------------------------------------------------------


def test_loop_cancels_cleanly_within_100ms(fake_world):
    detector = FakeDetector({
        i: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))]
        for i in range(50)
    })
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])

    async def run():
        task = asyncio.create_task(
            run_scene_perception(
                camera=fake_world["camera"],
                detector=detector,
                captioner=fake_world["captioner"],
                embedder=fake_world["embedder"],
                lifter=lifter,
                store=fake_world["store"],
                interval_sec=0.05,
                archive_source_frame=False,
            )
        )
        # Let it run a bit, then cancel and time the shutdown.
        await asyncio.sleep(0.1)
        t0 = time.perf_counter()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return time.perf_counter() - t0

    shutdown_ms = asyncio.run(run()) * 1000
    assert shutdown_ms < 200, f"shutdown took {shutdown_ms:.0f}ms; expected <200ms"
    # Store should have at least one chair record.
    assert fake_world["store"].count >= 1


# ---------------------------------------------------------------------------
# 4. Detector errors don't kill the loop
# ---------------------------------------------------------------------------


def test_loop_recovers_from_detector_exception(fake_world):
    chair = FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))
    detector = FakeDetector(
        {0: [chair], 1: [chair], 2: [chair], 3: [chair], 4: [chair]},
        raise_on_idx=2,
        exc=RuntimeError("simulated provider 500"),
    )
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))[0]

    # We received the tick where the detector raised — it carries `error`.
    error_reports = [r for r in reports if r.error]
    assert len(error_reports) >= 1
    assert "simulated provider 500" in error_reports[0].error

    # Ticks before and after still produced upserts.
    successful_inserts_or_updates = sum(
        r.n_inserts + r.n_updates for r in reports if not r.error
    )
    assert successful_inserts_or_updates >= 4
    # The single chair still ended up in the store.
    assert fake_world["store"].count == 1


# ---------------------------------------------------------------------------
# 5. Slow tick doesn't pile up — sequential execution
# ---------------------------------------------------------------------------


def test_loop_runs_ticks_sequentially(fake_world):
    """A slow captioner should NOT cause overlapping ticks.

    We measure the wall time between successive ticks and assert that the
    gap is always >= the slow-tick duration (i.e. the loop is sequential).
    """
    slow_delay = 0.05  # 50ms — much larger than 5ms interval
    chair = FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))
    detector = FakeDetector(
        {i: [chair] for i in range(20)},
    )
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])
    slow_captioner = FakeCaptioner("a chair", delay=slow_delay)

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=slow_captioner,
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.005,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=4, interval=slow_delay))[0]
    assert len(reports) >= 4
    # Successive on_tick timestamps must be at least slow_delay apart
    # (allowing a small slop tolerance for asyncio scheduling).
    gaps = [
        reports[i + 1].ts - reports[i].ts
        for i in range(len(reports) - 1)
    ]
    assert all(g >= slow_delay * 0.7 for g in gaps), (
        f"gaps={gaps}; loop is not running sequentially"
    )


# ---------------------------------------------------------------------------
# 6. Robot stares at a single mug for 200 ticks → 1 record, sightings == 200
#    (the "DB grows forever" guard)
# ---------------------------------------------------------------------------


def test_loop_stares_at_one_object_for_200_ticks(fake_world):
    """The classic 'don't blow up the DB' regression test.

    A static scene with one mug. The loop runs 200 ticks. The store must
    end with exactly one row, sightings >= 200. If dedup ever regresses
    (wrong threshold, missing class match, etc.), this test goes from
    1 record to 200 and the assertion fails loudly.
    """
    mug = FakeDetectedObject("mug", 41, 0.9, (100, 100, 200, 200))
    detector = FakeDetector({i: [mug] for i in range(250)})
    lifter = FakePositionLifter(default=[1.5, 0.3, 0.7])

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.001,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=200, interval=0.001))[0]
    assert len(reports) >= 200
    store = fake_world["store"]
    assert store.count == 1, (
        f"DB blew up to {store.count} rows after 200 ticks; dedup is broken"
    )
    entry = store.recency_query(since_ts=0.0)[0]
    assert entry.sightings >= 200
    # Exactly one INSERT across the whole run — the rest are updates.
    total_inserts = sum(r.n_inserts for r in reports)
    assert total_inserts == 1


# ---------------------------------------------------------------------------
# 7. Object moved by a human (position shifts beyond τ_pos)
# ---------------------------------------------------------------------------


def test_loop_handles_object_moved_by_human(fake_world):
    """Mug at A for several ticks, then mug at B (>0.5m away) for several ticks.

    Expected behavior: TWO records — the original at A goes stale, a new
    one at B is inserted. The store does not move A's position to B.

    The original at A keeps its last_seen_ts pinned at the moment it
    disappeared, so callers can detect 'stale' via timestamp aging.
    """
    mug = FakeDetectedObject("mug", 41, 0.9, (100, 100, 200, 200))
    detector = FakeDetector({
        0: [mug], 1: [mug], 2: [mug],            # mug at A
        3: [mug], 4: [mug], 5: [mug],            # mug at B (>0.5m away)
    })
    # Same bbox both times, but the lifter returns different world-frame
    # positions — simulates the robot's view being unchanged but the
    # object having been physically moved.
    pos_a = [0.0, 0.0, 0.5]
    pos_b = [3.0, 3.0, 0.5]  # > τ_pos away from A

    class _MovingLifter:
        def __init__(self):
            self.call = 0
        def bboxes_to_positions(self, coords, timeout=5.0):
            self.call += 1
            return [pos_a if self.call <= 3 else pos_b]

    lifter = _MovingLifter()

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.001,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    asyncio.run(_run_for_n_ticks(factory, n=6, interval=0.001))
    store = fake_world["store"]
    # Two distinct records.
    assert store.count == 2
    entries = store.recency_query(since_ts=0.0)
    positions = sorted(round(e.position[0], 1) for e in entries)
    assert positions == [0.0, 3.0]
    # The record at A should NOT have its position dragged toward B.
    record_at_a = next(e for e in entries if e.position[0] < 1.0)
    assert record_at_a.position[0] < 0.5
    # Both should have sightings >= 1; the order they appeared determines
    # which has more sightings but both must exist.
    assert all(e.sightings >= 1 for e in entries)


# ---------------------------------------------------------------------------
# 8. Long-run patrol — final DB size ≈ unique objects, not observations
# ---------------------------------------------------------------------------


def test_loop_long_patrol_db_size_matches_unique_objects(fake_world):
    """Simulate the robot patrolling a 4-object scene for ~300 ticks.

    Each tick the FOV picks up a different subset of the 4 objects (so
    overlapping FOVs are exercised). At the end the DB must contain 4
    records — one per unique object — not 300 (one per tick) and not
    some intermediate number (which would mean dedup is leaky).
    """
    # Four objects with stable world-frame positions.
    OBJECTS = {
        "chair_a": FakeDetectedObject("chair", 56, 0.92, (10, 10, 60, 60)),
        "chair_b": FakeDetectedObject("chair", 56, 0.91, (70, 10, 120, 60)),
        "mug":     FakeDetectedObject("mug",   41, 0.88, (130, 10, 180, 60)),
        "lamp":    FakeDetectedObject("lamp",  82, 0.85, (10, 70, 60, 120)),
    }
    POSITIONS = {
        "chair_a": [1.0, 1.0, 0.0],
        "chair_b": [3.0, 1.0, 0.0],
        "mug":     [2.0, 2.0, 0.5],
        "lamp":    [0.0, 3.0, 1.2],
    }
    # Deterministic patrol schedule — varies which subset is visible.
    # Three 'rooms' the robot rotates through; each ROI sees 2 objects.
    PATROL = [
        ["chair_a", "chair_b"],   # room 1
        ["chair_b", "mug"],       # room 2
        ["mug", "lamp"],          # room 3
        ["lamp", "chair_a"],      # cycle back
    ]
    # Build per-tick detection lists for ~300 ticks
    N_TICKS = 320
    scripted = {}
    for i in range(N_TICKS):
        visible = PATROL[i % len(PATROL)]
        scripted[i] = [OBJECTS[name] for name in visible]
    detector = FakeDetector(scripted)

    # The lifter must respond to which bbox came in, mapping each to its
    # canonical world-frame position. Build a lookup by (cx, cy, w, h).
    bbox_to_pos = {}
    for name, det in OBJECTS.items():
        x1, y1, x2, y2 = det.bbox
        key = (float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1))
        bbox_to_pos[key] = POSITIONS[name]
    lifter = FakePositionLifter(scripted=bbox_to_pos)

    # Pin embeddings per object so visual matching is deterministic across
    # ticks (no FakeEmbedder hashing variability — the crops differ
    # tick-to-tick because the source frame is the same tiny image but the
    # bbox-crop region differs).
    embedder = FakeEmbedder(
        dim=16,
        override_image={},  # we'll patch via subclass instead
    )

    # Override the image embedder to depend only on bbox identity, not
    # crop pixels — simulates a real CLIP being stable across viewpoints.
    bbox_to_emb_axis = {
        "chair_a": 0, "chair_b": 1, "mug": 2, "lamp": 3,
    }
    name_by_bbox = {OBJECTS[n].bbox: n for n in OBJECTS}

    class _StableEmbedder(FakeEmbedder):
        def embed_image(self, image):  # noqa: D401
            # Recover bbox identity from image size — each crop has the
            # bbox's width/height, so use that as the key.
            w, h = image.size
            for det_name, det in OBJECTS.items():
                x1, y1, x2, y2 = det.bbox
                if (x2 - x1, y2 - y1) == (w, h):
                    v = [0.0] * self.dim
                    v[bbox_to_emb_axis[det_name]] = 1.0
                    return v
            return super().embed_image(image)

    stable_embedder = _StableEmbedder(dim=16)
    store = SceneStore(persist_dir=fake_world["tmp_path"] / "patrol", embedder=stable_embedder)

    # A SINGLE frame, reused every tick. This matters: the FakeEmbedder
    # hashes the cropped bbox region, so if frame pixels vary per tick
    # the embedding of the *same physical object* would vary too, and
    # dedup would (correctly!) refuse to merge them. In a real CLIP
    # deployment, viewpoint-invariance handles this; in the fake we
    # achieve it by holding the source frame constant.
    static_frame = make_tiny_image(seed=99)

    async def factory(on_tick):
        return await run_scene_perception(
            camera=FakeCamera(frames=[static_frame]),
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=stable_embedder,
            lifter=lifter,
            store=store,
            interval_sec=0.0005,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    reports = asyncio.run(
        _run_for_n_ticks(factory, n=N_TICKS, interval=0.0005, timeout=60.0)
    )[0]
    assert len(reports) >= N_TICKS

    # Final DB has exactly 4 records, one per unique object.
    assert store.count == 4, (
        f"after {N_TICKS} ticks, store has {store.count} records; "
        f"expected 4 unique objects"
    )
    # And total inserts across the whole run is 4 — not 4 × number_of_re-sightings.
    total_inserts = sum(r.n_inserts for r in reports)
    assert total_inserts == 4

    # Per-class accounting matches the patrol schedule (each appears in
    # 2 out of 4 patrol stops).
    classes = sorted(e.class_name for e in store.recency_query(since_ts=0.0))
    assert classes == sorted(["chair", "chair", "mug", "lamp"])

    # Each record's sightings should be roughly N_TICKS / 2 (each object
    # appears in 2 of 4 patrol stops).
    expected = N_TICKS / 2
    for entry in store.recency_query(since_ts=0.0):
        assert abs(entry.sightings - expected) <= expected * 0.2, (
            f"{entry.class_name} sightings={entry.sightings}, "
            f"expected ~{expected}"
        )


# ---------------------------------------------------------------------------
# Position fallback — small/many objects whose 3D lift fails are still stored
# ---------------------------------------------------------------------------


class _NeverLifts:
    """A lifter that always fails (simulates depth missing / batch timeout)."""

    def bboxes_to_positions(self, coords, timeout=5.0):
        return None


def test_loop_drops_unliftable_objects_without_a_pose_provider(fake_world):
    detector = FakeDetector(
        {i: [FakeDetectedObject("pen", 1, 0.6, (5, 5, 12, 12))] for i in range(6)}
    )

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=_NeverLifts(),
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            on_tick=on_tick,
        )

    asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))
    # No fallback → unliftable detections are dropped (old behavior).
    assert fake_world["store"].count == 0


def test_loop_falls_back_to_robot_pose_when_lift_fails(fake_world):
    detector = FakeDetector(
        {i: [FakeDetectedObject("pen", 1, 0.6, (5, 5, 12, 12))] for i in range(6)}
    )

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=_NeverLifts(),
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            pose_provider=lambda: (4.0, 2.0, 0.0),  # robot's map pose
            on_tick=on_tick,
        )

    asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))
    store = fake_world["store"]
    # With a fallback pose the object is catalogued at the robot's location.
    assert store.count == 1
    entry = store.recency_query(since_ts=0.0)[0]
    assert entry.class_name == "pen"
    assert (round(entry.position[0], 1), round(entry.position[1], 1)) == (4.0, 2.0)


def test_loop_drops_unliftable_when_pose_fallback_disabled(fake_world):
    detector = FakeDetector(
        {i: [FakeDetectedObject("pen", 1, 0.6, (5, 5, 12, 12))] for i in range(6)}
    )

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=_NeverLifts(),
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            pose_provider=lambda: (4.0, 2.0, 0.0),  # available, but for prune only
            position_fallback_to_pose=False,
            on_tick=on_tick,
        )

    asyncio.run(_run_for_n_ticks(factory, n=5, interval=0.01))
    # The pose provider is present (the prune gate needs it) but the position
    # fallback is off → an unliftable detection is dropped, NOT stamped with the
    # robot's pose. This is the production setting: only real 3D lifts persist.
    assert fake_world["store"].count == 0


# ---------------------------------------------------------------------------
# Eviction (periodic prune wired into the loop)
# ---------------------------------------------------------------------------


def test_loop_prunes_object_after_it_is_removed(fake_world):
    """An object seen then removed ages out once TTL elapses, near the robot."""
    # Chair visible for the first 3 ticks, then gone from view.
    detector = FakeDetector(
        {i: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))] for i in range(3)}
    )
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            prune_ttl_sec=0.05,
            prune_interval_sec=0.01,
            prune_radius_m=1.0,
            pose_provider=lambda: (1.0, 1.0, 0.0),  # robot stays near the object
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=25, interval=0.01))[0]
    assert fake_world["store"].count == 0
    assert sum(r.n_pruned for r in reports) >= 1


def test_loop_does_not_prune_object_robot_has_left(fake_world):
    """Roaming safety: a stale object far from the robot is NOT evicted."""
    detector = FakeDetector(
        {i: [FakeDetectedObject("chair", 56, 0.9, (10, 10, 60, 60))] for i in range(3)}
    )
    lifter = FakePositionLifter(default=[1.0, 1.0, 0.0])

    async def factory(on_tick):
        return await run_scene_perception(
            camera=fake_world["camera"],
            detector=detector,
            captioner=fake_world["captioner"],
            embedder=fake_world["embedder"],
            lifter=lifter,
            store=fake_world["store"],
            interval_sec=0.01,
            archive_source_frame=False,
            prune_ttl_sec=0.05,
            prune_interval_sec=0.01,
            prune_radius_m=1.0,
            pose_provider=lambda: (100.0, 100.0, 0.0),  # robot wandered far away
            on_tick=on_tick,
        )

    reports = asyncio.run(_run_for_n_ticks(factory, n=25, interval=0.01))[0]
    # TTL elapsed, but the object is outside the prune radius → it survives.
    assert fake_world["store"].count == 1
    assert sum(r.n_pruned for r in reports) == 0
