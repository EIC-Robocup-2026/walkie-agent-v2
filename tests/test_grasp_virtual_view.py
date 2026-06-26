"""Pure-geometry unit tests for the virtual-viewpoint grasp transform.

No robot, no AI server — exercises the rigid rotation that reorients a lifted
optical cloud into a virtual "side"/"top" view before GraspNet, and the inverse
that maps the returned grasp back into the true optical frame. The grasp-quality
question (does the transform actually help GraspNet?) is an on-robot experiment —
see manual_tests/grasp_virtual_view.py; here we only pin the math: round-trip
exactness and the target viewing axis.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from client.grasp import GraspPose
from tasks.skills.grasp import (
    _apply_virtual_view,
    _grasp_to_optical,
    _invert_grasp_virtual,
    _resolve_virtual_view,
    _rotation_between,
    _virtual_view_rotation,
)


def _is_proper_rotation(R: np.ndarray) -> bool:
    R = np.asarray(R, dtype=float)
    return (
        R.shape == (3, 3)
        and np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        and abs(np.linalg.det(R) - 1.0) < 1e-9
    )


def _tilted_camera_up_opt(theta: float) -> np.ndarray:
    """World-up expressed in a camera-optical frame pitched *theta* rad below level.

    Level optical axes in the map frame: X=[0,-1,0], Y=[0,0,-1], Z=[1,0,0] (a camera
    looking along map +X, image-down = map -Z). Pitch down rotates about optical X.
    ``up_opt = R.T @ [0,0,1]`` is what the production code derives from snap.cam.R.
    """
    R_level = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    R = R_level @ Rotation.from_euler("x", theta).as_matrix()
    return R.T @ np.array([0.0, 0.0, 1.0])


# --- _rotation_between -------------------------------------------------------
def test_rotation_between_maps_a_onto_b():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.3, -0.7, 0.5])
    b = b / np.linalg.norm(b)
    R = _rotation_between(a, b)
    assert _is_proper_rotation(R)
    assert np.allclose(R @ a, b, atol=1e-9)


def test_rotation_between_parallel_is_identity():
    a = np.array([0.0, 0.0, 1.0])
    assert np.allclose(_rotation_between(a, a), np.eye(3), atol=1e-12)


def test_rotation_between_antiparallel_flips():
    a = np.array([0.0, 0.0, 1.0])
    R = _rotation_between(a, -a)
    assert _is_proper_rotation(R)
    assert np.allclose(R @ a, -a, atol=1e-9)


# --- _virtual_view_rotation -------------------------------------------------
def test_virtual_view_none_is_identity_about_median():
    cloud = np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.5], [-0.1, 0.05, 0.45]])
    R, c, c_out = _virtual_view_rotation(cloud, _tilted_camera_up_opt(0.6), "none")
    assert np.allclose(R, np.eye(3))
    assert np.allclose(c, np.median(cloud, axis=0))
    assert np.allclose(c_out, c)  # center_xy defaults off -> pivot unchanged


def test_virtual_view_side_makes_viewing_axis_horizontal():
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    cloud = np.random.default_rng(0).uniform(-0.05, 0.05, size=(200, 3)) + [0, 0, 0.5]
    R, _, _ = _virtual_view_rotation(cloud, up, "side")
    assert _is_proper_rotation(R)
    # Rotating the cloud by R is equivalent to moving the camera by R.T, so the virtual
    # viewing axis (in optical coords) is R.T @ forward. For "side" it must be horizontal
    # == perpendicular to gravity (up).
    view_axis = R.T @ np.array([0.0, 0.0, 1.0])
    assert abs(float(view_axis @ (up / np.linalg.norm(up)))) < 1e-9


def test_virtual_view_top_makes_viewing_axis_point_down():
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    up_unit = up / np.linalg.norm(up)
    cloud = np.random.default_rng(1).uniform(-0.05, 0.05, size=(200, 3)) + [0, 0, 0.5]
    R, _, _ = _virtual_view_rotation(cloud, up, "top")
    assert _is_proper_rotation(R)
    view_axis = R.T @ np.array([0.0, 0.0, 1.0])  # virtual viewing axis, optical frame
    assert np.allclose(view_axis, -up_unit, atol=1e-9)  # looks straight down


@pytest.mark.parametrize("deg", [15.0, 35.0, 45.0, 70.0])
def test_rotated_up_is_minus_y_for_side_minus_z_for_top(deg):
    """The `up` handed to GraspNet after the transform (v_rot @ up_opt) must be -Y for a
    side view and -Z for a top view, pose-independently (no-roll camera)."""
    up = _tilted_camera_up_opt(np.deg2rad(deg))
    cloud = np.full((5, 3), [0.0, 0.0, 0.5])
    R_side, _, _ = _virtual_view_rotation(cloud, up, "side")
    R_top, _, _ = _virtual_view_rotation(cloud, up, "top")
    assert np.allclose(R_side @ up, [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(R_top @ up, [0.0, 0.0, -1.0], atol=1e-9)


def test_virtual_view_top_faces_top_surface_at_camera():
    """The object's top-surface normal (world-up) must map onto virtual -Z (toward cam)."""
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    up_unit = up / np.linalg.norm(up)
    cloud = np.random.default_rng(5).uniform(-0.05, 0.05, size=(200, 3)) + [0, 0, 0.5]
    R, _, _ = _virtual_view_rotation(cloud, up, "top")
    assert np.allclose(R @ up_unit, [0.0, 0.0, -1.0], atol=1e-9)


def test_virtual_view_zero_up_is_identity():
    cloud = np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.5]])
    R, _, _ = _virtual_view_rotation(cloud, np.zeros(3), "side")
    assert np.allclose(R, np.eye(3))


def test_virtual_view_bad_mode_raises():
    with pytest.raises(ValueError):
        _virtual_view_rotation(np.zeros((3, 3)), np.array([0, -1, 0.0]), "diagonal")


# --- round-trip: apply then invert is exact ---------------------------------
@pytest.mark.parametrize("mode", ["side", "top"])
def test_apply_then_invert_is_identity(mode):
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    cloud = np.random.default_rng(2).uniform(-0.05, 0.05, size=(300, 3)) + [0, 0, 0.55]
    R, c, _ = _virtual_view_rotation(cloud, up, mode)
    cloud_v = _apply_virtual_view(cloud, R, c)
    # Inverting a virtual-frame *point* (identity rotation) must recover the original.
    for p_v, p0 in zip(cloud_v, cloud):
        _, back = _invert_grasp_virtual(np.eye(3), p_v, R, c)
        assert np.allclose(back, p0, atol=1e-9)


def test_grasp_to_optical_preserves_geometry_and_inverts():
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    cloud = np.random.default_rng(3).uniform(-0.05, 0.05, size=(200, 3)) + [0, 0, 0.5]
    R, c, _ = _virtual_view_rotation(cloud, up, "side")

    # A grasp as GraspNet would return it in the *virtual* frame.
    rot_v = Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_matrix()
    t_v = np.array([0.01, -0.02, 0.5])
    g_v = GraspPose(translation=tuple(t_v), rotation=rot_v, width=0.05, score=0.9)

    g_opt = _grasp_to_optical(g_v, R, c)
    # Re-expressed, not re-shaped: still a proper rotation, width/score untouched.
    assert _is_proper_rotation(g_opt.rotation)
    assert g_opt.width == g_v.width and g_opt.score == g_v.score
    # The approach axis (column 2) transforms by R.T (the inverse rotation).
    assert np.allclose(g_opt.rotation[:, 2], R.T @ rot_v[:, 2], atol=1e-9)
    # Pushing the optical grasp point forward into the virtual frame recovers t_v.
    fwd = _apply_virtual_view(np.asarray(g_opt.translation)[None, :], R, c)[0]
    assert np.allclose(fwd, t_v, atol=1e-9)


# --- center_xy: lateral recentring (depth kept) -----------------------------
def test_center_xy_zeroes_lateral_offset_keeps_depth():
    """center_xy drops the cloud's XY centroid onto the optical axis, preserving depth."""
    rng = np.random.default_rng(7)
    cloud = rng.uniform(-0.05, 0.05, size=(300, 3)) + [0.18, -0.09, 0.55]  # off-axis
    R, c_in, c_out = _virtual_view_rotation(cloud, np.array([0.0, -1.0, 0.0]), "none",
                                            center_xy=True)
    assert np.allclose(R, np.eye(3))
    assert np.allclose(c_in, np.median(cloud, axis=0))
    assert np.allclose(c_out[:2], 0.0, atol=1e-12)        # XY zeroed
    assert np.isclose(c_out[2], c_in[2], atol=1e-12)      # depth kept
    cloud_v = _apply_virtual_view(cloud, R, c_in, c_out)
    med = np.median(cloud_v, axis=0)
    assert np.allclose(med[:2], 0.0, atol=1e-9)           # recentred laterally
    assert np.isclose(med[2], np.median(cloud, axis=0)[2], atol=1e-9)  # same range


def test_center_xy_is_pure_translation_no_rotation():
    """With mode 'none', recentring must not rotate — it only shifts the cloud rigidly."""
    rng = np.random.default_rng(8)
    cloud = rng.uniform(-0.05, 0.05, size=(200, 3)) + [0.2, 0.1, 0.5]
    R, c_in, c_out = _virtual_view_rotation(cloud, np.array([0.0, -1.0, 0.0]), "none",
                                            center_xy=True)
    cloud_v = _apply_virtual_view(cloud, R, c_in, c_out)
    shift = cloud_v - cloud
    assert np.allclose(shift, shift[0], atol=1e-9)        # identical per-point => translation
    assert np.allclose(shift[0], [-c_in[0], -c_in[1], 0.0], atol=1e-9)


@pytest.mark.parametrize("mode", ["none", "side", "top"])
def test_center_xy_grasp_roundtrip_inverts(mode):
    """A grasp in the centred/rotated frame maps back to true optical exactly."""
    up = _tilted_camera_up_opt(np.deg2rad(35.0))
    rng = np.random.default_rng(9)
    cloud = rng.uniform(-0.05, 0.05, size=(200, 3)) + [0.15, -0.07, 0.55]
    R, c_in, c_out = _virtual_view_rotation(cloud, up, mode, center_xy=True)

    rot_v = Rotation.from_euler("xyz", [0.1, -0.3, 0.2]).as_matrix()
    t_v = np.array([0.005, -0.01, 0.55])
    g_v = GraspPose(translation=tuple(t_v), rotation=rot_v, width=0.05, score=0.9)

    g_opt = _grasp_to_optical(g_v, R, c_in, c_out)
    assert _is_proper_rotation(g_opt.rotation)
    assert np.allclose(g_opt.rotation, R.T @ rot_v, atol=1e-9)  # translation never rotates
    # Pushing the optical grasp point forward through the SAME transform recovers t_v.
    fwd = _apply_virtual_view(np.asarray(g_opt.translation)[None, :], R, c_in, c_out)[0]
    assert np.allclose(fwd, t_v, atol=1e-9)


def test_center_xy_off_matches_default_path_byte_for_byte():
    """center_xy=False must reproduce the legacy (c_out is None) behaviour exactly."""
    up = _tilted_camera_up_opt(np.deg2rad(40.0))
    rng = np.random.default_rng(10)
    cloud = rng.uniform(-0.05, 0.05, size=(150, 3)) + [0.2, -0.1, 0.6]
    R, c_in, c_out = _virtual_view_rotation(cloud, up, "top", center_xy=False)
    assert np.array_equal(c_out, c_in)
    legacy = _apply_virtual_view(cloud, R, c_in)            # c_out defaults to c_in
    explicit = _apply_virtual_view(cloud, R, c_in, c_out)
    assert np.array_equal(legacy, explicit)


# --- _resolve_virtual_view (the "auto" coupling) ----------------------------
@pytest.mark.parametrize("setting,pref,expected", [
    ("none", "none", "none"),
    ("none", "side", "none"),     # explicit none wins regardless of preference
    ("side", "none", "side"),     # explicit setting passes through
    ("top", "side", "top"),       # explicit setting ignores preference
    ("auto", "none", "none"),     # auto with no preference -> off
    ("auto", "side", "side"),     # auto follows preference
    ("auto", "top", "top"),
    ("auto", "garbage", "none"),  # auto with unknown preference -> off
    ("AUTO", "SIDE", "side"),     # case-insensitive
])
def test_resolve_virtual_view(setting, pref, expected):
    assert _resolve_virtual_view(setting, pref) == expected


def test_grasp_to_optical_none_mode_is_noop():
    cloud = np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.5], [-0.1, 0.05, 0.45]])
    R, c, _ = _virtual_view_rotation(cloud, np.array([0.0, -1.0, 0.0]), "none")
    rot = Rotation.from_euler("z", 0.3).as_matrix()
    g = GraspPose(translation=(0.02, 0.0, 0.45), rotation=rot, width=0.04, score=0.8)
    g2 = _grasp_to_optical(g, R, c)
    assert np.allclose(g2.rotation, g.rotation, atol=1e-12)
    assert np.allclose(g2.translation, g.translation, atol=1e-12)
