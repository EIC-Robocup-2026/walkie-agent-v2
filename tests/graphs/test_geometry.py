"""Unit tests for walkie_graphs.geometry — optical-frame deprojection, no robot."""

from __future__ import annotations

import numpy as np
import pytest

from walkie_graphs.geometry import (
    CameraPose,
    Intrinsics,
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
