"""Service-level detection filters + maintenance cadence (stubs, no robot/server)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from walkie_graphs.service import WalkieGraphsService


class _StubMemory:
    """Records which periodic-maintenance methods fired and when."""

    def __init__(self):
        self.calls = []

    def derive_relations(self):
        self.calls.append("relations")

    def prune(self):
        self.calls.append("prune")

    def denoise_nodes(self):
        self.calls.append("denoise")

    def merge_overlapping_nodes(self):
        self.calls.append("merge")

    def evict_stale_provisional(self, now_ts):
        self.calls.append("evict")


@pytest.fixture
def svc():
    s = WalkieGraphsService(walkieAI=None, walkie=None, memory=_StubMemory(), verbose=False)
    s.viz = None
    return s


def _det(mask_area, bbox):
    mask = np.zeros((100, 100), dtype=bool)
    mask.flat[:mask_area] = True
    return SimpleNamespace(mask=mask, bbox=bbox)


def test_size_filters_noop_by_default(svc):
    # defaults: max_bbox_area_ratio=1.0, min_mask_area_px=0 → keep everything
    assert svc._passes_size_filters(_det(5, (0, 0, 100, 100)), img_area=10000)


def test_rejects_whole_frame_box(svc):
    svc.max_bbox_area_ratio = 0.9
    big = _det(5000, (0, 0, 100, 100))  # bbox area 10000 == whole image
    small = _det(5000, (0, 0, 50, 50))  # bbox area 2500
    assert not svc._passes_size_filters(big, img_area=10000)
    assert svc._passes_size_filters(small, img_area=10000)


def test_rejects_tiny_mask(svc):
    svc.min_mask_area_px = 64
    assert not svc._passes_size_filters(_det(10, (0, 0, 50, 50)), img_area=10000)
    assert svc._passes_size_filters(_det(200, (0, 0, 50, 50)), img_area=10000)


def test_maintenance_cadence_staggered(svc):
    svc.relation_every_n = 5
    svc.denoise_every_n = 20
    svc.merge_every_n = 20
    svc.ghost_every_n = 20
    for _ in range(45):
        svc._maybe_tick(True)
    calls = svc.memory.calls
    # relations+prune at 5,10,...,45 (9 times each)
    assert calls.count("relations") == 9
    # denoise at tick 20, 40 (t % 20 == 0)
    assert calls.count("denoise") == 2
    # merge at 21, 41 (t % 20 == 1)
    assert calls.count("merge") == 2
    # evict at 22, 42 (t % 20 == 2)
    assert calls.count("evict") == 2


def test_tick_false_runs_nothing(svc):
    svc._maybe_tick(False)
    assert svc.memory.calls == []


def test_mask_subtract_and_crop_margin_defaults(svc):
    # mask subtraction defaults ON (CG always applies it); crop margin matches CG's 20px.
    assert svc.mask_subtract is True
    assert svc.crop_margin_px == 20


def test_flying_pixel_cleanup_defaults(svc):
    # depth-bleed cleanup defaults ON: erode 2px + reject depth jumps over 5cm.
    assert svc.mask_erode_px == 2
    assert svc.depth_edge_thresh_m == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# walkie-sdk integration: intrinsics from get_intrinsics, pose from optical TF
# ---------------------------------------------------------------------------
class _FakeCamera:
    def get_intrinsics(self):
        return {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}


class _FakeTransform:
    def __init__(self, tf):
        self.tf = tf
        self.calls = []

    def lookup(self, source, target, timeout=1.0):
        self.calls.append((source, target))
        return self.tf


class _FakeRobot:
    def __init__(self, tf):
        self.camera = _FakeCamera()
        self.transform = _FakeTransform(tf)


class _FakeWalkie:
    def __init__(self, tf):
        self.robot = _FakeRobot(tf)


def _svc_with_walkie(tf):
    w = _FakeWalkie(tf)
    s = WalkieGraphsService(walkieAI=None, walkie=w, memory=_StubMemory(), verbose=False)
    return s, w


def test_intrinsics_from_sdk_scaled_to_depth():
    s, _ = _svc_with_walkie(None)
    intr = s._intrinsics(640, 480)
    assert (intr.fx, intr.cx, intr.cy) == pytest.approx((500.0, 320.0, 240.0))
    # different depth resolution → intrinsics rescaled
    half = s._intrinsics(320, 240)
    assert (half.fx, half.cx) == pytest.approx((250.0, 160.0))


def test_camera_pose_from_optical_tf():
    tf = {
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},  # identity
    }
    s, w = _svc_with_walkie(tf)
    pose = s._camera_pose()
    assert np.allclose(pose.R, np.eye(3))
    assert np.allclose(pose.t, [1.0, 2.0, 3.0])
    # it looked up MAP_FRAME -> the optical camera frame
    assert w.robot.transform.calls == [("map", "zed_head_left_camera_frame_optical")]


def test_camera_pose_none_when_lookup_fails():
    s, _ = _svc_with_walkie(None)  # lookup returns None
    assert s._camera_pose() is None


def test_default_camera_frame_is_optical():
    s, _ = _svc_with_walkie(None)
    assert s._tf_cam_frame == "zed_head_left_camera_frame_optical"
