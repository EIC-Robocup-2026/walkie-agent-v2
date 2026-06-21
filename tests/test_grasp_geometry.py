"""Pure-geometry unit tests for the grasp-execution helpers.

No robot, no AI server — just the map->base transform, the lateral dead-zone
check, the look-down tilt math, and the face-object heading, exercised against
fake TaskContext stand-ins (only ``current_pose``/``rotate_to``/``transform`` are
touched). The arm/nav-heavy paths (execute_grasp, approach_object, pick_object)
are integration-level and validated on-robot.
"""

import math

import pytest

from tasks.skills.grasp import (
    _arm_sides,
    _look_down_tilt,
    _world_to_base,
    face_object_with_arm,
    in_arm_deadzone,
    _HEAD_TILT_MAX,
    _HEAD_TILT_MIN,
)


class FakeCtx:
    """Minimal stand-in: the pure helpers only touch current_pose()."""

    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self._pose = {"x": x, "y": y, "heading": heading}

    def current_pose(self):
        return self._pose


# --- _arm_sides -------------------------------------------------------------
def test_arm_sides_left():
    assert _arm_sides("left") == ("left_arm_lift", "left_arm", "left_gripper")


def test_arm_sides_right():
    assert _arm_sides("right") == ("right_arm_lift", "right_arm", "right_gripper")


def test_arm_sides_unknown_defaults_left():
    assert _arm_sides("bogus") == ("left_arm_lift", "left_arm", "left_gripper")
    assert _arm_sides("") == ("left_arm_lift", "left_arm", "left_gripper")


# --- _world_to_base ---------------------------------------------------------
def test_world_to_base_identity():
    # Robot at origin facing +x: base frame == map frame.
    assert _world_to_base(FakeCtx(), (1.0, 2.0, 0.5)) == pytest.approx((1.0, 2.0, 0.5))


def test_world_to_base_rotation_90deg():
    # Robot facing +y: a map point 2 m straight ahead lands at base x=+2, y=0.
    bx, by, bz = _world_to_base(FakeCtx(0.0, 0.0, math.pi / 2), (0.0, 2.0, 0.8))
    assert (bx, by, bz) == pytest.approx((2.0, 0.0, 0.8), abs=1e-9)


# --- in_arm_deadzone --------------------------------------------------------
def test_deadzone_dead_centre():
    # Robot at origin facing +x; object straight ahead -> lateral y == 0 -> in zone.
    assert in_arm_deadzone(FakeCtx(), (0.6, 0.0, 0.8)) is True


def test_deadzone_left_outside():
    # 0.30 m to the left is beyond the 0.20 m half-width -> reachable.
    assert in_arm_deadzone(FakeCtx(), (0.6, 0.30, 0.8)) is False


def test_deadzone_left_inside():
    # 0.10 m to the left is within the band -> still in the dead-zone.
    assert in_arm_deadzone(FakeCtx(), (0.6, 0.10, 0.8)) is True


def test_deadzone_respects_heading():
    # Robot facing +y: an object dead ahead (0, 0.6) has lateral y == 0 -> in zone,
    # exercising the rotation term (not just raw map y).
    assert in_arm_deadzone(FakeCtx(0.0, 0.0, math.pi / 2), (0.0, 0.6, 0.8)) is True


def test_deadzone_custom_half_width():
    assert in_arm_deadzone(FakeCtx(), (0.6, 0.30, 0.8), half_width_m=0.40) is True


# --- _look_down_tilt --------------------------------------------------------
def test_look_down_tilt_object_below_is_positive():
    # Camera at 1.2 m, object 0.6 m ahead at 0.7 m high -> look DOWN (positive).
    tilt = _look_down_tilt((0.0, 0.0, 1.2), (0.6, 0.0, 0.7))
    assert tilt == pytest.approx(math.atan2(0.5, 0.6))
    assert tilt > 0


def test_look_down_tilt_object_above_is_negative():
    # Object higher than the camera -> look UP (negative).
    tilt = _look_down_tilt((0.0, 0.0, 1.0), (0.5, 0.0, 1.4))
    assert tilt < 0


def test_look_down_tilt_clamped_down():
    # Object directly below the camera would be ~90deg down -> clamp to the max.
    assert _look_down_tilt((0.0, 0.0, 1.2), (0.001, 0.0, 0.0)) == pytest.approx(_HEAD_TILT_MAX)


def test_look_down_tilt_clamped_up():
    # Object far above, almost overhead -> clamp to the min.
    assert _look_down_tilt((0.0, 0.0, 0.5), (0.001, 0.0, 3.0)) == pytest.approx(_HEAD_TILT_MIN)


# --- face_object_with_arm ---------------------------------------------------
class FakeTransform:
    def __init__(self, link_xyz):
        self._link = (
            None if link_xyz is None
            else {"x": link_xyz[0], "y": link_xyz[1], "z": link_xyz[2]}
        )
        self.last_lookup = None

    def lookup(self, source, target, timeout=5.0):
        self.last_lookup = (source, target)
        if self._link is None:
            return None
        return {"position": self._link, "quaternion": {"x": 0, "y": 0, "z": 0, "w": 1}}


class FakeRobot:
    def __init__(self, transform):
        self.transform = transform


class FakeWalkie:
    def __init__(self, transform):
        self.robot = FakeRobot(transform)


class FaceCtx:
    """Records the heading passed to rotate_to; serves a fake transform tree."""

    def __init__(self, link_xyz, x=0.0, y=0.0, heading=0.0):
        self._pose = {"x": x, "y": y, "heading": heading}
        self.transform = FakeTransform(link_xyz)
        self.walkie = FakeWalkie(self.transform)
        self.rotated_to = None

    def current_pose(self):
        return self._pose

    def rotate_to(self, heading_rad):
        self.rotated_to = heading_rad
        return True


def test_face_object_heading_math():
    # Left shoulder at map (0.1, 0.2); object at (1.0, 0.2) -> due +x -> heading 0.
    ctx = FaceCtx(link_xyz=(0.1, 0.2, 0.9))
    assert face_object_with_arm(ctx, (1.0, 0.2, 0.8), arm="left") is True
    assert ctx.transform.last_lookup == ("map", "openarm_left_link3")
    assert ctx.rotated_to == pytest.approx(math.atan2(0.2 - 0.2, 1.0 - 0.1))


def test_face_object_heading_offset():
    ctx = FaceCtx(link_xyz=(0.0, 0.0, 0.9))
    face_object_with_arm(ctx, (1.0, 1.0, 0.8), arm="right")
    assert ctx.transform.last_lookup == ("map", "openarm_right_link3")
    assert ctx.rotated_to == pytest.approx(math.atan2(1.0, 1.0))  # 45deg


def test_face_object_no_transform_returns_false():
    ctx = FaceCtx(link_xyz=None)
    assert face_object_with_arm(ctx, (1.0, 0.0, 0.8)) is False
    assert ctx.rotated_to is None  # never rotated when the lookup fails
