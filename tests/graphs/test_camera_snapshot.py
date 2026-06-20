"""Unit tests for interfaces.devices.camera.CameraSnapshot — synthetic geometry, no robot."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from interfaces.devices.camera import CameraSnapshot
from interfaces.perception.geometry import CameraPose, Intrinsics


def _intr(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _identity_pose(t=(0.0, 0.0, 0.0)):
    return CameraPose(R=np.eye(3), t=np.array(t, dtype=float))


def _snap(depth, *, cam=None, intr=None, img_size=None):
    h, w = depth.shape if depth is not None else (480, 640)
    img = SimpleNamespace(size=img_size or (w, h))
    return CameraSnapshot(ts=0.0, img=img, depth=depth, cam=cam, intr=intr, robot_pose=None)


def _flat(depth_m=2.0, h=480, w=640):
    return np.full((h, w), depth_m, dtype=np.float32)


# ---------------------------------------------------------------------------
# bbox lifting
# ---------------------------------------------------------------------------
def test_bbox_world_point_lands_on_central_ray():
    # Flat plane at 2 m, identity pose: the bbox around the principal point
    # lifts to (~0, ~0, 2) in optical coords (x right, y down, z forward).
    snap = _snap(_flat(2.0), cam=_identity_pose(), intr=_intr())
    p = snap.bbox_world_point((300, 220, 340, 260), sor_k=0)
    assert p is not None
    assert p[0] == pytest.approx(0.0, abs=0.02)
    assert p[1] == pytest.approx(0.0, abs=0.02)
    assert p[2] == pytest.approx(2.0, abs=0.01)


def test_bbox_world_point_applies_translation():
    snap = _snap(_flat(1.0), cam=_identity_pose(t=(10.0, -3.0, 0.5)), intr=_intr())
    p = snap.bbox_world_point((300, 220, 340, 260), sor_k=0)
    assert p == pytest.approx((10.0, -3.0, 1.5), abs=0.05)


def test_bbox_shrink_narrows_the_pixel_set():
    snap = _snap(_flat(2.0), cam=_identity_pose(), intr=_intr())
    full = snap.bbox_to_points((200, 150, 400, 350), shrink=1.0, voxel=0, max_points=10**9)
    small = snap.bbox_to_points((200, 150, 400, 350), shrink=0.5, voxel=0, max_points=10**9)
    assert 0 < len(small) < len(full)
    # central 50% per axis -> about a quarter of the pixels
    assert len(small) == pytest.approx(len(full) * 0.25, rel=0.1)


def test_bbox_median_ignores_background_rim():
    # Object plane at 1 m covers the majority of the bbox pixels; the rim sees
    # the far wall at 5 m. The median point must stay on the near object — and
    # because bbox lifting skips voxelization, the wall's larger WORLD footprint
    # per pixel cannot outvote the object's pixel majority.
    depth = _flat(5.0)
    depth[195:285, 275:365] = 1.0  # 90x90 object inside the 120x120 bbox (~56% of pixels)
    snap = _snap(depth, cam=_identity_pose(), intr=_intr())
    p = snap.bbox_world_point((260, 180, 380, 300), shrink=1.0, sor_k=0, use_edge_filter=False)
    assert p is not None
    assert p[2] == pytest.approx(1.0, abs=0.05)


def test_bbox_coords_scale_to_depth_resolution():
    # RGB image is 1280x960 but depth is 640x480: bbox pixels (in RGB coords)
    # must be halved before masking the depth.
    snap = _snap(_flat(2.0), cam=_identity_pose(), intr=_intr(), img_size=(1280, 960))
    p = snap.bbox_world_point((600, 440, 680, 520), sor_k=0)  # centered in RGB coords
    assert p is not None
    assert p[0] == pytest.approx(0.0, abs=0.02)
    assert p[1] == pytest.approx(0.0, abs=0.02)


def test_mask_to_points_max_depth_env_default(monkeypatch):
    depth = _flat(2.0)
    depth[:, 320:] = 6.0
    snap = _snap(depth, cam=_identity_pose(), intr=_intr())
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[200:280, 280:360] = 1  # straddles the near/far split
    monkeypatch.setenv("WALKIE_GRAPHS_MAX_DEPTH_M", "4.0")
    gated = snap.mask_to_points(mask, voxel=0, max_points=10**9, erode_px=0)
    assert gated[:, 2].max() < 4.0
    # explicit override beats the env default
    full = snap.mask_to_points(mask, voxel=0, max_points=10**9, erode_px=0, max_depth=0.0,
                               use_edge_filter=False)
    assert full[:, 2].max() == pytest.approx(6.0, abs=1e-4)


# ---------------------------------------------------------------------------
# degraded snapshots
# ---------------------------------------------------------------------------
def test_no_geometry_lifts_to_nothing():
    snap = _snap(_flat(2.0), cam=None, intr=_intr())
    assert not snap.has_geometry
    assert snap.mask_to_points(np.ones((480, 640), dtype=np.uint8)).shape == (0, 3)
    assert snap.bbox_to_points((0, 0, 10, 10)).shape == (0, 3)
    assert snap.bbox_world_point((0, 0, 10, 10)) is None
    assert snap.bbox_world_xy((0, 0, 10, 10)) is None


def test_degenerate_bbox_is_empty():
    snap = _snap(_flat(2.0), cam=_identity_pose(), intr=_intr())
    assert snap.bbox_to_points((50, 50, 50, 50)).shape == (0, 3)
    assert snap.bbox_world_point((50, 50, 50, 50)) is None


# ---------------------------------------------------------------------------
# capture() against a stub walkie
# ---------------------------------------------------------------------------
def _stub_walkie(*, depth, tf, intrinsics, pose=None):
    return SimpleNamespace(
        robot=SimpleNamespace(
            camera=SimpleNamespace(
                get_depth=lambda: depth,
                get_intrinsics=lambda: intrinsics,
            ),
            transform=SimpleNamespace(lookup=lambda *a, **k: tf),
        ),
        camera=SimpleNamespace(capture_pil=lambda: SimpleNamespace(size=(640, 480))),
        status=SimpleNamespace(get_position=lambda: pose),
    )


_TF = {
    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
    "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}
_INTR_RAW = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}


def test_capture_populates_all_fields():
    w = _stub_walkie(depth=_flat(2.0), tf=_TF, intrinsics=_INTR_RAW, pose={"x": 0, "y": 0, "heading": 0})
    snap = CameraSnapshot.capture(w)
    assert snap is not None and snap.has_geometry
    assert np.allclose(snap.cam.t, [1.0, 2.0, 3.0])
    assert snap.intr.fx == pytest.approx(500.0)
    assert snap.robot_pose == {"x": 0, "y": 0, "heading": 0}
    # and the captured geometry lifts end to end
    assert snap.bbox_world_point((300, 220, 340, 260), sor_k=0) is not None


def test_capture_requires_depth_and_image():
    assert CameraSnapshot.capture(_stub_walkie(depth=None, tf=_TF, intrinsics=_INTR_RAW)) is None

    w = _stub_walkie(depth=_flat(2.0), tf=_TF, intrinsics=_INTR_RAW)
    w.camera = SimpleNamespace(capture_pil=lambda: (_ for _ in ()).throw(RuntimeError("no cam")))
    assert CameraSnapshot.capture(w) is None


def test_capture_degrades_to_no_geometry_without_tf():
    w = _stub_walkie(depth=_flat(2.0), tf=None, intrinsics=_INTR_RAW)
    snap = CameraSnapshot.capture(w)
    assert snap is not None and not snap.has_geometry
    assert snap.bbox_world_point((0, 0, 10, 10)) is None
