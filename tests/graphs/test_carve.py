"""Free-space carving: projection math (pure numpy) + GraphMemory write-through."""

from __future__ import annotations

import numpy as np
import pytest

from services.walkie_graphs.carve import (
    corrected_pose,
    free_space_mask,
    frustum_aabb,
)
from interfaces.perception.geometry import CameraPose, Intrinsics
from services.walkie_graphs.memory import GraphMemory
from tests.graphs.conftest import put_object, unit

INTR = Intrinsics(fx=500.0, fy=500.0, cx=50.0, cy=50.0, width=100, height=100)
EYE = CameraPose(R=np.eye(3), t=np.zeros(3))


def _wall_depth(z=2.0, shape=(100, 100)):
    return np.full(shape, float(z), dtype=np.float32)


def _world_at_pixel(u, v, z, intr=INTR):
    """The world point (identity pose) that projects to pixel (u, v) at depth z."""
    return np.array(
        [(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z], dtype=np.float32
    )


# ---------------------------------------------------------------------------
# free_space_mask — the core "seen straight through" test
# ---------------------------------------------------------------------------
def test_carves_point_seen_through():
    # A point claiming to be solid at z=1, but the sensor measured the wall at z=2
    # right behind it → free space → carve.
    p = _world_at_pixel(50, 50, 1.0)[None]
    mask = free_space_mask(p, _wall_depth(2.0), INTR, EYE)
    assert mask.tolist() == [True]


def test_keeps_surface_occluded_and_untestable_points():
    pts = np.stack(
        [
            _world_at_pixel(50, 50, 2.0),   # ON the measured wall → keep (margin)
            _world_at_pixel(50, 50, 3.0),   # BEHIND the wall (occluded) → keep
            _world_at_pixel(50, 50, 5.0),   # beyond max_z=4 → untestable → keep
            np.array([0.0, 0.0, -1.0]),     # behind the camera → keep
            _world_at_pixel(9999, 50, 1.0), # projects out of frame → keep
        ]
    ).astype(np.float32)
    mask = free_space_mask(pts, _wall_depth(2.0), INTR, EYE, max_z=4.0)
    assert not mask.any()


def test_nan_or_zero_depth_is_no_evidence():
    depth = _wall_depth(2.0)
    depth[50, 50] = np.nan
    p_nan = _world_at_pixel(50, 50, 1.0)[None]
    assert not free_space_mask(p_nan, depth, INTR, EYE).any()
    depth[50, 50] = 0.0
    assert not free_space_mask(p_nan, depth, INTR, EYE).any()


def test_margin_grows_with_depth():
    # A point 6 cm in front of the measured surface at z=1.
    p = _world_at_pixel(50, 50, 1.0)[None]
    depth = _wall_depth(1.06)
    # base margin 5 cm, no relative term → 6 cm gap carves.
    assert free_space_mask(p, depth, INTR, EYE, margin_base=0.05, margin_rel=0.0).all()
    # widen the base margin past the gap → kept.
    assert not free_space_mask(p, depth, INTR, EYE, margin_base=0.10, margin_rel=0.0).any()
    # relative term also widens it (0.05 + 0.02*1 = 0.07 > 0.06) → kept.
    assert not free_space_mask(p, depth, INTR, EYE, margin_base=0.05, margin_rel=0.02).any()


# ---------------------------------------------------------------------------
# frustum_aabb — bounds everything in view
# ---------------------------------------------------------------------------
def test_frustum_aabb_contains_lifted_points():
    lo, hi = frustum_aabb(INTR, EYE, (100, 100), z_min=0.05, z_max=4.0)
    for u, v, z in [(0, 0, 0.1), (99, 99, 3.9), (50, 50, 1.0), (10, 80, 2.5)]:
        p = _world_at_pixel(u, v, z)
        assert np.all(p >= lo - 1e-5) and np.all(p <= hi + 1e-5)


def test_frustum_aabb_translates_with_pose():
    pose = CameraPose(R=np.eye(3), t=np.array([10.0, 0.0, 0.0]))
    lo, hi = frustum_aabb(INTR, pose, (100, 100), z_min=0.05, z_max=4.0)
    assert lo[0] >= 10.0 - 1.0 and hi[0] <= 10.0 + 1.0  # x-shifted by the pose


# ---------------------------------------------------------------------------
# corrected_pose — fold a registration correction into the projection pose
# ---------------------------------------------------------------------------
def test_corrected_pose_identity_is_noop():
    assert corrected_pose(EYE, None) is EYE
    assert corrected_pose(EYE, np.eye(4)) is EYE


def test_corrected_pose_roundtrip():
    from services.walkie_graphs.pcd_ops import apply_transform

    rng = np.random.default_rng(0)
    # A non-trivial camera pose.
    ang = 0.3
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    cam = CameraPose(R=R, t=np.array([0.5, -0.2, 1.0]))
    T = np.eye(4)
    T[:3, 3] = (0.05, 0.03, -0.02)  # the accepted correction
    p_opt = rng.normal(size=(20, 3)).astype(np.float32)
    raw_world = p_opt @ cam.R.T + cam.t
    corrected_world = apply_transform(raw_world, T)
    pose = corrected_pose(cam, T)
    # Inverse-projecting the corrected world points through the corrected pose must
    # recover the original optical-frame points.
    recovered = (corrected_world - pose.t) @ pose.R
    assert np.allclose(recovered, p_opt, atol=1e-4)


# ---------------------------------------------------------------------------
# GraphMemory.carve_free_space — write-through to background + nodes
# ---------------------------------------------------------------------------
def _mem(tmp_path, **kw):
    from services.walkie_graphs.background import BackgroundStore
    from services.walkie_graphs.capture import CaptureStore

    return GraphMemory(
        chroma_dir=None,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
        capture_store=CaptureStore(str(tmp_path / "captures")),
        background=BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.01),
        dbscan_enabled=False, sor_k=0, voxel_m=0.001, max_points_per_obj=100000,
        **kw,
    )


def test_carve_removes_background_ghost(tmp_path):
    mem = _mem(tmp_path)
    # A grid of ghost points at z=1 (in front of the wall) + real wall points at z=2.
    us, vs = np.meshgrid(np.arange(30, 70), np.arange(30, 70))
    ghost = np.stack(
        [(us.ravel() - 50) / 500.0, (vs.ravel() - 50) / 500.0, np.ones(us.size)], axis=1
    ).astype(np.float32)
    wall = np.stack(
        [(us.ravel() - 50) * 2 / 500.0, (vs.ravel() - 50) * 2 / 500.0, np.full(us.size, 2.0)],
        axis=1,
    ).astype(np.float32)
    mem.background.add(np.vstack([ghost, wall]))
    n_before = len(mem.background)

    stats = mem.carve_free_space(_wall_depth(2.0), INTR, EYE)
    assert stats["bg_carved"] > 0
    remaining = mem.background.points()
    assert len(remaining) < n_before
    # Everything left sits at/behind the wall — the z=1 ghost is gone.
    assert remaining[:, 2].min() > 1.5


def test_carve_evicts_seen_through_node(tmp_path):
    mem = _mem(tmp_path)
    # A node whose whole cloud is a ghost at z=1, directly in view.
    us, vs = np.meshgrid(np.arange(30, 70), np.arange(30, 70))
    ghost = np.stack(
        [(us.ravel() - 50) / 500.0, (vs.ravel() - 50) / 500.0, np.ones(us.size)], axis=1
    ).astype(np.float32)
    put_object(mem, "ghost-1", "box", ghost, emb=unit(1, 0, 0))
    assert mem.count() == 1

    stats = mem.carve_free_space(_wall_depth(2.0), INTR, EYE, evict_min_points=20)
    assert stats["nodes_evicted"] == 1
    assert mem.count() == 0


def test_carve_flags_partially_carved_node_for_refine(tmp_path):
    from dataclasses import replace

    from services.walkie_graphs.capture import Capture, Segment
    from tests.graphs.conftest import make_det

    mem = _mem(tmp_path, segments_per_node=8)
    # Segment: 60 points at z=1 (carved) + 40 spread points at z=2 (kept).
    us1, vs1 = np.meshgrid(np.arange(20, 26), np.arange(20, 30))  # 60 px
    front = np.stack(
        [(us1.ravel() - 50) / 500.0, (vs1.ravel() - 50) / 500.0, np.ones(us1.size)], axis=1
    )
    us2, vs2 = np.meshgrid(np.arange(60, 70), np.arange(60, 64))  # 40 px
    back = np.stack(
        [(us2.ravel() - 50) * 2 / 500.0, (vs2.ravel() - 50) * 2 / 500.0, np.full(us2.size, 2.0)],
        axis=1,
    )
    seg_pts = np.vstack([front, back]).astype(np.float32)

    cap = Capture(id="c1-cv", ts=1.0, cam=None, segments=[Segment("c1-cv", 0, seg_pts)])
    mem.capture_store.save(cap)
    det = replace(
        make_det(class_name="box", emb=unit(1, 0, 0)),
        points_world=seg_pts, segment_ref="c1-cv:0",
    )
    node = mem.upsert(det)
    mem.capture_store.flush()

    stats = mem.carve_free_space(
        _wall_depth(2.0), INTR, EYE, evict_min_points=10, refine_frac=0.5
    )
    assert stats["nodes_carved"] == 1
    assert node.id in mem._refine_pending
    # The carved segment on disk now holds only the z=2 remainder.
    fresh_seg = mem.capture_store.load_segment("c1-cv:0")
    assert fresh_seg is not None
    assert fresh_seg[:, 2].min() > 1.5


def test_carve_no_geometry_is_noop(tmp_path):
    mem = _mem(tmp_path)
    put_object(mem, "n1", "box", _world_at_pixel(50, 50, 1.0)[None], emb=unit(1, 0, 0))
    stats = mem.carve_free_space(None, INTR, EYE)
    assert stats == {"bg_carved": 0, "nodes_carved": 0, "nodes_evicted": 0}
    assert mem.count() == 1
