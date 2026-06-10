"""Unit tests for services.walkie_graphs.geometry — optical-frame deprojection, no robot."""

from __future__ import annotations

import numpy as np
import pytest

from services.walkie_graphs.geometry import (
    CameraPose,
    Intrinsics,
    depth_discontinuity_mask,
    deproject_mask,
    voxel_downsample,
)


def _intr(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
def test_intrinsics_scaled_to_noop_when_same_res():
    intr = _intr(640, 480)
    assert intr.scaled_to(640, 480) is intr


def test_intrinsics_scaled_to_halves():
    intr = _intr(640, 480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    half = intr.scaled_to(320, 240)
    assert (half.fx, half.fy, half.cx, half.cy) == pytest.approx((250.0, 250.0, 160.0, 120.0))
    assert (half.width, half.height) == (320, 240)


def test_intrinsics_scaled_to_unknown_native_res_is_noop():
    intr = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=0, height=0)
    assert intr.scaled_to(640, 480) is intr


# ---------------------------------------------------------------------------
# Deprojection (optical frame -> map)
# ---------------------------------------------------------------------------
def test_deproject_identity_pose_is_optical_plus_translation():
    # R = I, t = (1,2,3): a center pixel at depth 2 → optical (0,0,2) → map (1,2,5).
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[240, 320] = 1  # the principal point
    pts = deproject_mask(mask, depth, intr, pose)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx((1.0, 2.0, 5.0), abs=1e-5)


def test_deproject_applies_rotation():
    # 90° yaw about map-z: optical (0,0,z) maps to map (0, 0, ...)? Check a known point.
    # Place a single off-center pixel so optical x is non-zero, then rotate.
    intr = _intr()
    R = _rot_z(np.pi / 2)  # maps (x,y,z) -> (-y, x, z)
    pose = CameraPose(R=R, t=np.zeros(3))
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[240, 420] = 1  # x offset of +100 px → optical Xc = 100*4/500 = 0.8
    pts = deproject_mask(mask, depth, intr, pose)
    # optical point ≈ (0.8, 0, 4); after Rz(90°): (-0, 0.8, 4)
    assert pts[0] == pytest.approx((0.0, 0.8, 4.0), abs=1e-4)


def test_deproject_constant_plane_count_and_value():
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:260, 300:380] = 1
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) == 60 * 80
    assert np.allclose(pts[:, 2], 2.0)  # frontoparallel plane → constant optical z = map z


def test_deproject_drops_nan_and_zero():
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 1.5, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[0, 0:4] = 1  # nan, zero, and two valid
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) == 2


def test_deproject_resizes_mask_to_depth():
    intr = _intr(1280, 720)
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    mask = np.zeros((360, 640), dtype=np.uint8)  # half-res mask
    mask[100:200, 100:200] = 1
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) > 0


def test_deproject_empty_mask():
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 1.5, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    assert deproject_mask(mask, depth, intr, pose).shape == (0, 3)


def test_deproject_voxel_and_cap():
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 100:400] = 1
    capped = deproject_mask(mask, depth, intr, pose, max_points=500)
    assert len(capped) == 500


def test_voxel_downsample_reduces_count():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 0.05, size=(1000, 3)).astype(np.float32)  # all in one 10cm voxel
    out = voxel_downsample(pts, 0.1)
    assert len(out) == 1
    assert out[0] == pytest.approx(pts.mean(axis=0), abs=1e-4)


# ---------------------------------------------------------------------------
# Flying-pixel (depth-bleed) cleanup
# ---------------------------------------------------------------------------
def test_depth_discontinuity_flags_step_edge():
    depth = np.full((10, 10), 1.0, dtype=np.float32)
    depth[:, 5:] = 1.5  # a 0.5 m step between columns 4 and 5
    edge = depth_discontinuity_mask(depth, thresh=0.1)
    # both columns straddling the jump are flagged...
    assert edge[:, 4].all() and edge[:, 5].all()
    # ...and a flat interior column is not.
    assert not edge[:, 0].any() and not edge[:, 9].any()


def test_depth_discontinuity_ignores_small_steps_and_disabled():
    depth = np.full((10, 10), 1.0, dtype=np.float32)
    depth[:, 5:] = 1.02  # 2 cm step, below threshold
    assert not depth_discontinuity_mask(depth, thresh=0.1).any()
    assert depth_discontinuity_mask(depth, thresh=0.0) is None


def test_depth_discontinuity_ignores_nan():
    depth = np.full((6, 6), 1.0, dtype=np.float32)
    depth[:, 3:] = np.nan  # invalid region, not a real surface jump
    edge = depth_discontinuity_mask(depth, thresh=0.1)
    assert not edge.any()


def test_deproject_edge_mask_drops_flying_pixels():
    # Object plane at 1.0 m on the left, wall at 1.5 m on the right; a vertical mask
    # spanning the boundary. The edge filter should drop the silhouette columns.
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 1.0, dtype=np.float32)
    depth[:, 320:] = 1.5
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:260, 315:325] = 1  # straddles the depth step at column 320
    edge = depth_discontinuity_mask(depth, thresh=0.1)
    clean = deproject_mask(mask, depth, intr, pose, edge_mask=edge)
    raw = deproject_mask(mask, depth, intr, pose)
    assert len(clean) < len(raw)  # boundary pixels removed
    # no surviving point sits at the bled intermediate band (depths are exactly 1.0/1.5)
    assert set(np.round(np.unique(clean[:, 2]), 3)).issubset({1.0, 1.5})


def test_deproject_erode_shrinks_cloud():
    intr = _intr()
    pose = CameraPose(R=np.eye(3), t=np.zeros(3))
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:240, 300:340] = 1  # 40x40 = 1600 px block
    full = deproject_mask(mask, depth, intr, pose)
    eroded = deproject_mask(mask, depth, intr, pose, erode_px=2)
    assert len(eroded) < len(full)
    # eroding a 40x40 block by 2 px → ~36x36 = 1296
    assert len(eroded) == 36 * 36
