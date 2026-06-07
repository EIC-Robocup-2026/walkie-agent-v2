"""Unit tests for walkie_graphs.geometry — the camera math, no robot/server."""

from __future__ import annotations

import math

import numpy as np
import pytest

from walkie_graphs.geometry import (
    DEFAULT_LIFT_TO_HEAD,
    DEFAULT_PIVOT_TO_OPTIC,
    CameraPose,
    Intrinsics,
    compute_camera_pose,
    deproject_mask,
    pixel_to_world,
    rot_y,
    voxel_downsample,
)


def _project_world_to_pixel(p_world, intr: Intrinsics, pose: CameraPose):
    """Inverse of pixel_to_world, used to test the round-trip."""
    p_local = pose.R.T @ (np.asarray(p_world, dtype=float) - pose.t)
    Zc = p_local[0]
    Xc = -p_local[1]
    Yc = -p_local[2]
    u = Xc * intr.fx / Zc + intr.cx
    v = Yc * intr.fy / Zc + intr.cy
    return u, v, Zc


# ---------------------------------------------------------------------------
# Camera pose composition
# ---------------------------------------------------------------------------
def test_lift_cm_to_m_conversion():
    # 37 cm of lift must add exactly 0.37 m to the camera height, not 37 m.
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=37.0, tilt_rad=0.0)
    expected_z = 0.37 + DEFAULT_LIFT_TO_HEAD[2]  # + pivot z (0 at tilt 0)
    assert pose.t[2] == pytest.approx(expected_z, abs=1e-9)


def test_pose_at_zero():
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    expected = np.array(
        [
            DEFAULT_LIFT_TO_HEAD[0] + DEFAULT_PIVOT_TO_OPTIC[0],  # 0.265 + 0.065
            0.0,
            DEFAULT_LIFT_TO_HEAD[2],  # 0.422
        ]
    )
    assert pose.t == pytest.approx(expected, abs=1e-9)
    assert pose.R == pytest.approx(np.eye(3), abs=1e-9)


def test_pose_tilt_down_lowers_and_advances_optical_center():
    t = math.pi / 4  # 45° down
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=t)
    # x = 0.265 + 0.065*cos(t); z = 0.422 - 0.065*sin(t)
    assert pose.t[0] == pytest.approx(0.265 + 0.065 * math.cos(t), abs=1e-9)
    assert pose.t[2] == pytest.approx(0.422 - 0.065 * math.sin(t), abs=1e-9)
    assert pose.R == pytest.approx(rot_y(t), abs=1e-9)


def test_tilt_sign_points_camera_down():
    # Positive tilt must aim the optical axis (local forward +x) below horizontal.
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.3)
    forward_world = pose.R @ np.array([1.0, 0.0, 0.0])
    assert forward_world[2] < 0.0  # looking down


def test_heading_rotates_offset_into_world():
    # Facing +y (heading 90°): the forward mount offset should land along world +y.
    pose = compute_camera_pose(0.0, 0.0, math.pi / 2, lift_cm=0.0, tilt_rad=0.0)
    assert pose.t[0] == pytest.approx(0.0, abs=1e-9)
    assert pose.t[1] == pytest.approx(0.265 + 0.065, abs=1e-9)


# ---------------------------------------------------------------------------
# Deprojection
# ---------------------------------------------------------------------------
def test_pixel_to_world_center_pixel():
    intr = Intrinsics.from_hfov(1280, 720, 110.0)
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    x, y, z = pixel_to_world(intr.cx, intr.cy, 2.0, intr, pose)
    # 2 m straight ahead + the camera's own forward/height offset.
    assert (x, y, z) == pytest.approx((2.0 + 0.330, 0.0, 0.422), abs=1e-6)


def test_pixel_to_world_round_trip():
    intr = Intrinsics.from_hfov(1280, 720, 110.0)
    pose = compute_camera_pose(1.2, -0.4, 0.3, lift_cm=20.0, tilt_rad=0.2)
    target = np.array([2.5, 0.7, 0.9])
    u, v, depth = _project_world_to_pixel(target, intr, pose)
    back = pixel_to_world(u, v, depth, intr, pose)
    assert back == pytest.approx(tuple(target), abs=1e-6)


def test_deproject_mask_constant_plane():
    intr = Intrinsics.from_hfov(1280, 720, 110.0)
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[300:360, 600:680] = 1
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) == 60 * 80
    # Frontoparallel plane at Zc=2, identity rotation → all share world x = 2.33.
    assert np.allclose(pts[:, 0], 2.0 + 0.330, atol=1e-4)


def test_deproject_mask_drops_nan_and_zero():
    intr = Intrinsics.from_hfov(640, 480, 110.0)
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    depth = np.full((480, 640), 1.5, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[0, 0:4] = 1  # 4 pixels: nan, zero, and two valid
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) == 2


def test_deproject_mask_resizes_to_depth_shape():
    intr = Intrinsics.from_hfov(1280, 720, 110.0)
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    depth = np.full((720, 1280), 2.0, dtype=np.float32)
    mask = np.zeros((360, 640), dtype=np.uint8)  # half-res mask
    mask[100:200, 100:200] = 1
    pts = deproject_mask(mask, depth, intr, pose)
    assert len(pts) > 0  # resized without crashing


def test_deproject_mask_empty():
    intr = Intrinsics.from_hfov(640, 480, 110.0)
    pose = compute_camera_pose(0.0, 0.0, 0.0, lift_cm=0.0, tilt_rad=0.0)
    depth = np.full((480, 640), 1.5, dtype=np.float32)
    mask = np.zeros((480, 640), dtype=np.uint8)
    pts = deproject_mask(mask, depth, intr, pose)
    assert pts.shape == (0, 3)


def test_voxel_downsample_reduces_count():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 0.05, size=(1000, 3)).astype(np.float32)  # all in one 10cm voxel
    out = voxel_downsample(pts, 0.1)
    assert len(out) == 1
    assert out[0] == pytest.approx(pts.mean(axis=0), abs=1e-4)


def test_intrinsics_overrides():
    intr = Intrinsics.from_hfov(1280, 720, 110.0, fx=500.0, cx=600.0)
    assert intr.fx == 500.0
    assert intr.fy == 500.0  # fy defaults to fx
    assert intr.cx == 600.0
    assert intr.cy == 360.0  # derived
