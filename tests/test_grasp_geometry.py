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
# align_arm_to_object now issues a pure lateral body-frame creep via
# creep_base_relative (direct cmd_vel, NOT nav.go_to), so the tests capture the
# (forward_m, left_m) it asks for rather than an absolute map goal. The strafe is
# body-frame and therefore heading-independent by construction.


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
    def __init__(self, transform):
        self.robot = AlignRobot(transform)


class AlignCtx:
    def __init__(self, arm_left, x=0.0, y=0.0, heading=0.0):
        self._pose = {"x": x, "y": y, "heading": heading}
        self.walkie = AlignWalkie(AlignTransform(arm_left))

    def current_pose(self):
        return self._pose


@pytest.fixture
def captured_creep(monkeypatch):
    """Capture (forward_m, left_m) handed to creep_base_relative; stub the drive out."""
    calls = []

    def fake_creep(ctx, forward_m, left_m=0.0, **kwargs):
        calls.append((forward_m, left_m))
        return True

    monkeypatch.setattr("tasks.skills.grasp.creep_base_relative", fake_creep)
    return calls


def test_align_arm_strafes_to_put_object_in_front_of_arm(captured_creep):
    # Robot facing +x; left arm mounted +0.2 m to the left. Object dead ahead
    # (0.6, 0) -> obj_left 0, so the base must strafe -0.2 m (to the right, i.e.
    # left_m = -0.2) to line the arm up, with no forward component.
    ctx = AlignCtx(arm_left=0.2)
    assert align_arm_to_object(ctx, (0.6, 0.0, 0.8), arm="left") is True
    forward_m, left_m = captured_creep[-1]
    assert forward_m == pytest.approx(0.0, abs=1e-9)  # pure strafe, no creep
    assert left_m == pytest.approx(-0.2, abs=1e-9)


def test_align_arm_strafe_is_heading_independent(captured_creep):
    # The strafe is a body-frame lateral creep, so the SAME object geometry
    # relative to the arm yields the same strafe whatever the map heading. Robot
    # facing +y with the object straight ahead (0, 0.6) -> obj_left 0, arm_left 0.2
    # -> left_m -0.2, identical to the +x-facing case above.
    ctx = AlignCtx(arm_left=0.2, heading=math.pi / 2)
    align_arm_to_object(ctx, (0.0, 0.6, 0.8), arm="left")
    forward_m, left_m = captured_creep[-1]
    assert forward_m == pytest.approx(0.0, abs=1e-9)
    assert left_m == pytest.approx(-0.2, abs=1e-9)


def test_align_arm_ignores_creep_failure(captured_creep, monkeypatch):
    # A blocked/refused strafe (creep returns False) is best-effort — align still
    # reports True so the grasp proceeds from wherever the base got to.
    monkeypatch.setattr("tasks.skills.grasp.creep_base_relative",
                        lambda *a, **k: False)
    ctx = AlignCtx(arm_left=0.2)
    assert align_arm_to_object(ctx, (0.6, 0.0, 0.8), arm="left") is True


# --- _grasp_cloud_multi_tilt (dedup + fuse + reframe) -----------------------
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from tasks.skills import grasp as grasp_mod  # noqa: E402
from tasks.skills.grasp import ObjectLocation, _grasp_cloud_multi_tilt  # noqa: E402


class _Cam:
    def __init__(self, R, t):
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float)


class _Snap:
    def __init__(self, R, t):
        self.cam = _Cam(R, t)


def _world_object(seed=0, center=(1.5, 2.2, 0.85)):
    return np.random.RandomState(seed).randn(200, 3) * 0.05 + np.asarray(center)


def _loc_for(world, R, t, conf):
    """Build an ObjectLocation as if a camera at (R, t) saw *world*."""
    snap = _Snap(R, t)
    cloud_opt = (world - snap.cam.t) @ snap.cam.R  # p_opt = (p_map - t) @ R
    xyz = tuple(np.median(world, axis=0))
    return ObjectLocation(xyz_map=xyz, cloud_optical=cloud_opt, snap=snap,
                          range_m=float(np.median(np.linalg.norm(cloud_opt, axis=1))),
                          confidence=conf)


def _patch_two_views(monkeypatch, locs):
    """Make tilt_head a no-op and locate_object yield *locs* in order."""
    monkeypatch.setattr(grasp_mod, "tilt_head", lambda *a, **k: None)
    it = iter(locs)
    monkeypatch.setattr(grasp_mod, "locate_object", lambda *a, **k: next(it, None))


def test_multi_tilt_fuses_same_object_into_chosen_optical_frame(monkeypatch):
    # Two cameras see the SAME world object; the fused cloud, reframed into the
    # first view's optical frame, must reproject back to the true world points.
    world = _world_object()
    RA = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    RB = Rotation.from_euler("xyz", [-0.05, 0.15, 1.2]).as_matrix()
    a = _loc_for(world, RA, [1.0, 2.0, 0.7], conf=0.9)
    b = _loc_for(world, RB, [1.3, 1.8, 0.72], conf=0.8)
    _patch_two_views(monkeypatch, [a, b])

    out = _grasp_cloud_multi_tilt(FakeCtx(), ["can"], merge_voxel=1e-6)
    assert out is not None
    cloud, snap, ref = out
    assert snap is a.snap  # chosen = first view
    back = cloud @ snap.cam.R.T + snap.cam.t  # optical -> map via chosen pose
    # Every fused point lies on the original object (within the tiny merge voxel).
    d = np.linalg.norm(back[:, None, :] - world[None, :, :], axis=2).min(axis=1)
    assert d.max() < 1e-3
    # ref_optical reprojects to the chosen view's map-frame centroid.
    assert ref.shape == (3,)
    assert np.allclose(ref @ snap.cam.R.T + snap.cam.t, a.xyz_map, atol=1e-5)


def test_multi_tilt_mismatch_keeps_higher_confidence_view(monkeypatch):
    # Centroids far apart (> dedup radius) -> different objects -> don't fuse;
    # keep the higher-confidence view's cloud/snap unchanged.
    R = np.eye(3)
    a = _loc_for(_world_object(seed=1, center=(1.0, 0.0, 0.85)), R, [0, 0, 0], conf=0.4)
    b = _loc_for(_world_object(seed=2, center=(3.0, 0.0, 0.85)), R, [0, 0, 0], conf=0.95)
    _patch_two_views(monkeypatch, [a, b])

    out = _grasp_cloud_multi_tilt(FakeCtx(), ["can"], dedup_radius_m=0.10)
    assert out is not None
    cloud, snap, ref = out
    assert snap is b.snap
    assert np.array_equal(cloud, b.cloud_optical)
    assert ref.shape == (3,)


def test_multi_tilt_single_view_degrades_gracefully(monkeypatch):
    # Only one tilt yields geometry -> return that view alone (no fuse).
    a = _loc_for(_world_object(), np.eye(3), [0, 0, 0], conf=0.5)
    _patch_two_views(monkeypatch, [a, None])

    out = _grasp_cloud_multi_tilt(FakeCtx(), ["can"])
    assert out is not None
    cloud, snap, ref = out
    assert snap is a.snap and np.array_equal(cloud, a.cloud_optical)
    assert ref.shape == (3,)


def test_multi_tilt_no_views_returns_none(monkeypatch):
    _patch_two_views(monkeypatch, [None, None])
    assert _grasp_cloud_multi_tilt(FakeCtx(), ["can"]) is None
