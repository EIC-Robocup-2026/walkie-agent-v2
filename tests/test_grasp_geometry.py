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
    align_arm_to_object,
    face_object,
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


# --- face_object ------------------------------------------------------------
def test_face_object_points_heading_at_object():
    # Robot at origin; object at (1, 1) -> heading 45deg toward it.
    ctx = FaceCtx(link_xyz=(0.0, 0.0, 0.9))  # link unused by face_object
    assert face_object(ctx, (1.0, 1.0, 0.8)) is True
    assert ctx.rotated_to == pytest.approx(math.atan2(1.0, 1.0))


# --- align_arm_to_object ----------------------------------------------------
class AlignNav:
    """Records the absolute go_to target (a pure strafe keeps the heading)."""

    def __init__(self):
        self.last_goto = None

    def go_to(self, x, y, heading=None, blocking=True):
        self.last_goto = (x, y, heading)
        return "SUCCEEDED"


class AlignTransform:
    """Serves base_footprint -> openarm_{side}_link3 with a fixed lateral offset."""

    def __init__(self, arm_left):
        self._arm_left = arm_left

    def lookup(self, source, target, timeout=5.0):
        return {"position": {"x": 0.1, "y": self._arm_left, "z": 0.9}}


class AlignRobot:
    def __init__(self, transform):
        self.transform = transform


class AlignWalkie:
    def __init__(self, transform, nav):
        self.robot = AlignRobot(transform)
        self.nav = nav


class AlignCtx:
    def __init__(self, arm_left, x=0.0, y=0.0, heading=0.0):
        self._pose = {"x": x, "y": y, "heading": heading}
        self.nav = AlignNav()
        self.walkie = AlignWalkie(AlignTransform(arm_left), self.nav)

    def current_pose(self):
        return self._pose


def test_align_arm_strafes_to_put_object_in_front_of_arm():
    # Robot facing +x; left arm mounted +0.2 m to the left. Object dead ahead
    # (0.6, 0) -> obj_left 0, so the base must strafe -0.2 m (to the right) to
    # line the arm up: target = origin + (-0.2) * base_left_axis(0, 1) = (0, -0.2).
    ctx = AlignCtx(arm_left=0.2)
    assert align_arm_to_object(ctx, (0.6, 0.0, 0.8), arm="left") is True
    x, y, heading = ctx.nav.last_goto
    assert (x, y) == pytest.approx((0.0, -0.2), abs=1e-9)
    assert heading == pytest.approx(0.0)  # pure strafe: heading unchanged


def test_align_arm_respects_heading():
    # Robot facing +y; base +left axis is -x. Object at (0, 0.6) -> obj_left 0,
    # arm_left 0.2 -> strafe -0.2 along (-1, 0) -> target (+0.2, 0).
    ctx = AlignCtx(arm_left=0.2, heading=math.pi / 2)
    align_arm_to_object(ctx, (0.0, 0.6, 0.8), arm="left")
    x, y, heading = ctx.nav.last_goto
    assert (x, y) == pytest.approx((0.2, 0.0), abs=1e-9)
    assert heading == pytest.approx(math.pi / 2)


def test_align_arm_ignores_nav_failure():
    # A nav that raises must not propagate — the strafe is best-effort.
    ctx = AlignCtx(arm_left=0.2)

    def boom(*a, **k):
        raise RuntimeError("nav down")

    ctx.nav.go_to = boom
    assert align_arm_to_object(ctx, (0.6, 0.0, 0.8), arm="left") is True
