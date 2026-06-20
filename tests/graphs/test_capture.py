"""Capture: remainder lift, capture-level registration, CaptureStore lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from services.walkie_graphs import pcd_ops
from services.walkie_graphs.capture import (
    Capture,
    CaptureStore,
    Segment,
    lift_capture,
    parse_ref,
    register_capture,
)
from interfaces.perception.geometry import CameraPose, Intrinsics


def _snap(depth, *, cam=True):
    h, w = depth.shape
    return SimpleNamespace(
        ts=100.0,
        img=SimpleNamespace(size=(w, h)),
        depth=depth,
        cam=CameraPose(R=np.eye(3), t=np.zeros(3)) if cam else None,
        intr=Intrinsics(fx=500.0, fy=500.0, cx=w / 2, cy=h / 2, width=w, height=h),
        has_geometry=cam,
    )


def _flat(depth_m=2.0, h=480, w=640):
    return np.full((h, w), depth_m, dtype=np.float32)


def _box_mask(x1, y1, x2, y2, h=480, w=640):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _px_to_world_x(u, depth=2.0, cx=320.0, fx=500.0):
    return (u - cx) * depth / fx


# ---------------------------------------------------------------------------
# lift_capture — segments + background remainder
# ---------------------------------------------------------------------------
def test_lift_builds_segments_and_remainder():
    snap = _snap(_flat())
    obj_mask = _box_mask(100, 100, 200, 200)
    obj_pts = np.ones((50, 3), dtype=np.float32)
    cap = lift_capture(snap, [obj_mask], [(0, obj_pts)], bg_voxel_m=0.05)
    assert [s.ref for s in cap.segments] == [f"{cap.id}:0"]
    assert np.array_equal(cap.segments[0].points, obj_pts)
    assert len(cap.background) > 0
    assert cap.icp_accepted is False and np.array_equal(cap.correction, np.eye(4))


def test_remainder_excludes_all_masks_even_masking_only():
    """A person's mask carves their pixels out of the background despite never
    becoming a segment — that's the whole point of masking-only detections."""
    snap = _snap(_flat(2.0))
    obj_mask = _box_mask(100, 100, 200, 200)
    person_mask = _box_mask(400, 100, 500, 400)
    cap = lift_capture(
        snap, [obj_mask, person_mask], [(0, np.ones((5, 3), np.float32))],
        bg_voxel_m=0.0, bg_max_points=10**9, bg_dilate_px=4,
    )
    # World-x band of the person's mask interior (well inside the dilation rim).
    x_lo, x_hi = _px_to_world_x(410), _px_to_world_x(490)
    in_person_band = (cap.background[:, 0] > x_lo) & (cap.background[:, 0] < x_hi)
    y_lo, y_hi = _px_to_world_x(110, cx=240.0), _px_to_world_x(390, cx=240.0)
    in_person_band &= (cap.background[:, 1] > y_lo) & (cap.background[:, 1] < y_hi)
    assert not in_person_band.any()
    # And the carve actually removed area vs not knowing about the person.
    cap_blind = lift_capture(
        snap, [obj_mask], [(0, np.ones((5, 3), np.float32))],
        bg_voxel_m=0.0, bg_max_points=10**9, bg_dilate_px=4,
    )
    assert len(cap.background) < len(cap_blind.background)


def test_dilation_widens_the_carve():
    snap = _snap(_flat())
    mask = _box_mask(200, 150, 440, 330)
    thin = lift_capture(snap, [mask], [], bg_voxel_m=0.0, bg_max_points=10**9, bg_dilate_px=0)
    wide = lift_capture(snap, [mask], [], bg_voxel_m=0.0, bg_max_points=10**9, bg_dilate_px=8)
    assert len(wide.background) < len(thin.background)


def test_lift_without_geometry_has_empty_background():
    snap = _snap(_flat(), cam=False)
    cap = lift_capture(snap, [None], [(0, np.ones((5, 3), np.float32))])
    assert cap.background.shape == (0, 3)
    assert len(cap.segments) == 1  # caller-lifted points pass through regardless


def test_lift_background_respects_max_depth():
    depth = _flat(2.0)
    depth[:, 320:] = 6.0  # far half beyond the trusted ZED range
    snap = _snap(depth)
    gated = lift_capture(snap, [], [], bg_voxel_m=0.0, bg_max_points=10**9, bg_max_depth_m=4.0)
    full = lift_capture(snap, [], [], bg_voxel_m=0.0, bg_max_points=10**9)
    assert len(gated.background) < len(full.background)
    assert gated.background[:, 2].max() < 4.0
    assert full.background[:, 2].max() == pytest.approx(6.0, abs=1e-4)


# ---------------------------------------------------------------------------
# register_capture — one rigid correction for the whole capture
# ---------------------------------------------------------------------------
def _corner_cloud(n=600, seed=3):
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(0, 0.5, n), rng.uniform(0, 0.5, n), np.zeros(n)], axis=1)
    wall = np.stack([rng.uniform(0, 0.5, n), np.zeros(n), rng.uniform(0, 0.5, n)], axis=1)
    return np.vstack([floor, wall]).astype(np.float32)


def _capture_with(background, segments_pts):
    return Capture(
        id="c1-test",
        ts=0.0,
        cam=None,
        segments=[
            Segment("c1-test", i, np.asarray(p, np.float32))
            for i, p in enumerate(segments_pts)
        ],
        background=np.asarray(background, np.float32),
    )


def test_register_corrects_pose_error_on_segments_and_background():
    pytest.importorskip("open3d")
    bg_true = _corner_cloud()
    seg_true = (_corner_cloud(80, seed=9)[:120] * 0.2 + 0.1).astype(np.float32)
    # The map target holds both the background AND the object's prior cloud,
    # exactly what icp_target_near assembles — segment points need true
    # correspondences or they bias the solve.
    target = np.vstack([bg_true, seg_true])
    offset = np.array([0.05, 0.03, 0.02], dtype=np.float32)
    cap = _capture_with(bg_true + offset, [seg_true + offset])
    out = register_capture(cap, target, max_corr_dist=0.1, min_points=100)
    assert out.icp_accepted and out.icp_fitness > 0.9
    assert np.abs(out.background - bg_true).max() < 0.005
    # The SAME correction moved the segment back too.
    assert np.abs(out.segments[0].points - seg_true).max() < 0.005
    assert not np.array_equal(out.correction, np.eye(4))


def test_register_rejects_low_fitness():
    pytest.importorskip("open3d")
    target = _corner_cloud()
    bg = target + np.array([5.0, 0.0, 0.0], dtype=np.float32)  # no overlap at all
    cap = _capture_with(bg, [])
    out = register_capture(cap, target, max_corr_dist=0.1, min_points=100)
    assert not out.icp_accepted
    assert np.array_equal(out.background, bg)  # untouched


def test_register_caps_reject_oversized_corrections(monkeypatch):
    """Translation/rotation caps gate deterministically (solver stubbed out)."""
    big_t = np.eye(4)
    big_t[:3, 3] = (0.5, 0.0, 0.0)
    a = np.radians(10.0)
    big_r = np.eye(4)
    big_r[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    ok = np.eye(4)
    ok[:3, 3] = (0.05, 0.0, 0.0)

    target = _corner_cloud()
    for T, accepted in [(big_t, False), (big_r, False), (ok, True)]:
        monkeypatch.setattr(pcd_ops, "icp", lambda *a, T=T, **k: (T, 0.95))
        cap = _capture_with(target.copy(), [])
        out = register_capture(
            cap, target, max_corr_dist=0.25,
            max_trans_m=0.3, max_rot_deg=5.0, min_points=100,
        )
        assert out.icp_accepted is accepted
        assert out.icp_fitness == pytest.approx(0.95)


def test_register_skips_when_disabled_or_starved():
    target = _corner_cloud()
    cap = _capture_with(target + 0.05, [])
    assert register_capture(cap, target, max_corr_dist=0.0).icp_accepted is False
    assert register_capture(cap, None, max_corr_dist=0.1).icp_accepted is False
    assert (
        register_capture(cap, target[:10], max_corr_dist=0.1, min_points=100).icp_accepted
        is False
    )
    thin = _capture_with(target[:10] + 0.05, [])
    assert register_capture(thin, target, max_corr_dist=0.1, min_points=100).icp_accepted is False


# ---------------------------------------------------------------------------
# CaptureStore — deferred writes, reads, refcounted GC
# ---------------------------------------------------------------------------
def _stored_capture(cid="c5-abc", n_segs=2):
    return Capture(
        id=cid,
        ts=5.0,
        cam=None,
        segments=[
            Segment(cid, i, np.full((10, 3), float(i), np.float32)) for i in range(n_segs)
        ],
        background=np.zeros((0, 3), np.float32),
    )


def test_parse_ref_roundtrip():
    seg = Segment("c7-x:y", 3, np.zeros((1, 3), np.float32))
    assert parse_ref(seg.ref) == ("c7-x:y", 3)


def test_save_flush_load(tmp_path):
    store = CaptureStore(str(tmp_path))
    cap = _stored_capture()
    store.save(cap)
    # Readable before any flush (pending queue / cache)...
    assert np.array_equal(store.load_segment(f"{cap.id}:1"), cap.segments[1].points)
    assert store.flush() == 1
    assert (tmp_path / f"{cap.id}.npz").exists()
    # ...and from disk by a fresh instance after.
    fresh = CaptureStore(str(tmp_path))
    assert np.array_equal(fresh.load_segment(f"{cap.id}:0"), cap.segments[0].points)
    assert fresh.load_segment(f"{cap.id}:9") is None
    assert fresh.load_segment("c0-missing:0") is None


def test_empty_capture_not_saved(tmp_path):
    store = CaptureStore(str(tmp_path))
    store.save(_stored_capture(n_segs=0))
    assert store.flush() == 0


def test_gc_unlinks_unreferenced_only(tmp_path):
    store = CaptureStore(str(tmp_path))
    kept, dropped = _stored_capture("c1-kept"), _stored_capture("c2-dropped")
    store.save(kept)
    store.save(dropped)
    store.flush()
    store.retain("c1-kept:0")
    store.retain("c1-kept:1")
    assert store.gc() == 1
    assert (tmp_path / "c1-kept.npz").exists()
    assert not (tmp_path / "c2-dropped.npz").exists()
    # Releasing the last ref makes it collectable.
    store.release("c1-kept:0")
    store.release("c1-kept:1")
    assert store.gc() == 1
    assert not (tmp_path / "c1-kept.npz").exists()


def test_gc_drops_unreferenced_pending_writes(tmp_path):
    store = CaptureStore(str(tmp_path))
    cap = _stored_capture("c3-pend")
    store.save(cap)  # never retained, never flushed
    store.gc()
    assert store.flush() == 0
    assert not (tmp_path / "c3-pend.npz").exists()


def test_update_segment_pending(tmp_path):
    store = CaptureStore(str(tmp_path))
    cap = _stored_capture("c4-up")
    store.save(cap)  # still pending
    new_pts = np.full((7, 3), 9.0, np.float32)
    assert store.update_segment("c4-up:1", new_pts)
    assert np.array_equal(store.load_segment("c4-up:1"), new_pts)
    store.flush()
    fresh = CaptureStore(str(tmp_path))
    assert np.array_equal(fresh.load_segment("c4-up:1"), new_pts)


def test_update_segment_on_disk_preserves_siblings(tmp_path):
    store = CaptureStore(str(tmp_path))
    cap = _stored_capture("c5-up", n_segs=2)
    store.save(cap)
    store.flush()
    new_pts = np.full((4, 3), 7.0, np.float32)
    assert store.update_segment("c5-up:0", new_pts)
    store.flush()
    fresh = CaptureStore(str(tmp_path))
    assert np.array_equal(fresh.load_segment("c5-up:0"), new_pts)
    # The sibling segment in the same capture file survived the rewrite.
    assert np.array_equal(fresh.load_segment("c5-up:1"), cap.segments[1].points)


def test_update_segment_missing_returns_false(tmp_path):
    store = CaptureStore(str(tmp_path))
    assert not store.update_segment("c0-gone:0", np.ones((3, 3), np.float32))
    cap = _stored_capture("c6-up")
    store.save(cap)
    assert not store.update_segment("c6-up:9", np.ones((3, 3), np.float32))
