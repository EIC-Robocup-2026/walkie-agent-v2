"""Offline tests for the live-scan/live-approach pure helpers (no robot/network).

The live find+approach path (RESTAURANT_LIVE_SCAN) keys everything off two pure pieces
of geometry: the absolute BEARING of a person from their torso-centre pixel, and the
CLOSE/FAR classification by depth validity. Both are unit-testable without hardware, and
the bearing SIGN in particular is easy to get backwards — this locks the repo's convention
(a person LEFT of frame centre needs a LARGER, CCW heading to face, matching the
FaceTracker rotate in tasks/skills/people.py).
"""

from __future__ import annotations

import math

from interfaces.perception.geometry import Intrinsics
from tasks.Restaurant.skills import (
    Caller,
    _bearing_from_pixel_u,
    _bearing_is_dup,
    _caller_is_close,
    _ray_point,
    _wrap,
)


class _Img:
    def __init__(self, w: int, h: int):
        self.size = (w, h)


class _Snap:
    """Minimal CameraSnapshot stand-in for the bearing helper."""

    def __init__(self, intr: Intrinsics, heading: float, rgb_w: int, rgb_h: int):
        self.intr = intr
        self.img = _Img(rgb_w, rgb_h)
        self.robot_pose = {"heading": heading}


def _intr(width=640, height=480, fx=600.0, cx=320.0):
    return Intrinsics(fx=fx, fy=fx, cx=cx, cy=240.0, width=width, height=height)


# --- bearing sign (the discriminator) --------------------------------------
def test_bearing_centre_pixel_equals_heading():
    snap = _Snap(_intr(), heading=0.7, rgb_w=640, rgb_h=480)
    assert _bearing_from_pixel_u(320.0, snap) == math.atan2(math.sin(0.7), math.cos(0.7))


def test_bearing_left_of_centre_is_larger_ccw():
    # u < cx => person is to the LEFT => robot must turn CCW (heading increases).
    snap = _Snap(_intr(), heading=0.0, rgb_w=640, rgb_h=480)
    left = _bearing_from_pixel_u(160.0, snap)
    assert left > 0.0


def test_bearing_right_of_centre_is_smaller_cw():
    snap = _Snap(_intr(), heading=0.0, rgb_w=640, rgb_h=480)
    right = _bearing_from_pixel_u(480.0, snap)
    assert right < 0.0


def test_bearing_adds_capture_heading():
    snap = _Snap(_intr(), heading=1.0, rgb_w=640, rgb_h=480)
    assert math.isclose(_bearing_from_pixel_u(320.0, snap), 1.0, abs_tol=1e-9)


def test_bearing_scales_rgb_column_to_depth_resolution():
    # intr at 640 wide, image at 1280 wide: rgb centre 640 -> depth centre 320 (=cx).
    snap = _Snap(_intr(width=640, cx=320.0), heading=0.0, rgb_w=1280, rgb_h=960)
    assert math.isclose(_bearing_from_pixel_u(640.0, snap), 0.0, abs_tol=1e-9)


def test_bearing_none_without_intrinsics():
    class _NoIntr:
        intr = None
        img = _Img(640, 480)
        robot_pose = {"heading": 0.0}

    assert _bearing_from_pixel_u(100.0, _NoIntr()) is None


def test_bearing_none_without_capture_heading():
    snap = _Snap(_intr(), heading=0.0, rgb_w=640, rgb_h=480)
    snap.robot_pose = None
    assert _bearing_from_pixel_u(320.0, snap) is None


# --- bearing dedup ----------------------------------------------------------
def test_bearing_dedup_collapses_within_bucket():
    kept = [math.radians(10.0)]
    assert _bearing_is_dup(math.radians(15.0), kept, dedup_deg=8.0)  # 5deg apart


def test_bearing_dedup_keeps_distinct_beyond_bucket():
    kept = [math.radians(10.0)]
    assert not _bearing_is_dup(math.radians(30.0), kept, dedup_deg=8.0)  # 20deg apart


def test_bearing_dedup_handles_wrap():
    kept = [math.radians(179.0)]
    assert _bearing_is_dup(math.radians(-179.0), kept, dedup_deg=8.0)  # 2deg across the seam


# --- close/far classification by depth validity -----------------------------
def test_far_when_no_depth_fix():
    pose = {"x": 0.0, "y": 0.0}
    assert not _caller_is_close(None, pose, reliable_m=3.5)


def test_close_when_valid_fix_in_band():
    pose = {"x": 0.0, "y": 0.0}
    assert _caller_is_close((2.0, 0.0), pose, reliable_m=3.5)


def test_far_when_valid_fix_beyond_band():
    pose = {"x": 0.0, "y": 0.0}
    assert not _caller_is_close((10.0, 0.0), pose, reliable_m=3.5)


# --- misc -------------------------------------------------------------------
def test_ray_point_projects_along_bearing():
    p = _ray_point({"x": 1.0, "y": 2.0}, bearing=0.0, dist=2.0)
    assert math.isclose(p[0], 3.0) and math.isclose(p[1], 2.0)
    p2 = _ray_point({"x": 0.0, "y": 0.0}, bearing=math.pi / 2, dist=2.0)
    assert math.isclose(p2[0], 0.0, abs_tol=1e-9) and math.isclose(p2[1], 2.0)


def test_caller_accepts_none_world_xy():
    c = Caller(world_xy=None, bearing=0.3, bbox_xyxy=(0.0, 0.0, 1.0, 1.0), confidence=0.8)
    assert c.world_xy is None and c.bearing == 0.3


def test_wrap_normalizes_to_pi_interval():
    assert math.isclose(_wrap(3 * math.pi), math.pi, abs_tol=1e-9) or \
           math.isclose(_wrap(3 * math.pi), -math.pi, abs_tol=1e-9)
    assert math.isclose(_wrap(0.0), 0.0)
