"""Unit tests for masked detection + lift in tasks.skills.grasp._detect_and_lift.

No robot, no AI server: the detector is a fake walkieAI client returning canned
detections, and the snapshot is a real CameraSnapshot over a synthetic flat-depth
array (so mask_to_points lifts genuine optical-frame points). The custom detector
boxes the target directly, so the selection rule is plain nearest-wins: among the
masks that lift to enough points, the closest-to-camera cloud is chosen. We also
pin the confidence / min_points filters and that prompts pass through unexpanded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from interfaces.devices.camera import CameraSnapshot
from interfaces.perception.geometry import CameraPose, Intrinsics
from tasks.skills.grasp import _detect_and_lift


# ---------------------------------------------------------------------------
# Snapshot + fake-client scaffolding
# ---------------------------------------------------------------------------
def _intr(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _snap(depth):
    h, w = depth.shape
    img = SimpleNamespace(size=(w, h))
    return CameraSnapshot(
        ts=0.0, img=img, depth=depth,
        cam=CameraPose(R=np.eye(3), t=np.zeros(3)), intr=_intr(), robot_pose=None,
    )


def _two_object_depth():
    """Flat scene at 4 m with a NEAR object (1 m) and a FAR object (2.5 m)."""
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    depth[200:280, 100:180] = 1.0   # near object, left of centre
    depth[200:280, 460:540] = 2.5   # far object, right of centre
    return depth


def _mask(rows, cols):
    m = np.zeros((480, 640), dtype=np.uint8)
    m[rows[0]:rows[1], cols[0]:cols[1]] = 1
    return m


def _det(mask, *, confidence=0.9):
    return SimpleNamespace(mask=mask, confidence=confidence)


class _FakeImage:
    """Stand-in for ctx.walkieAI.image with a detect() call recorder."""

    def __init__(self, dets):
        self._dets = dets
        self.detect_calls: list[list[str]] = []

    def detect(self, img, *, prompts=None, return_mask=False):
        self.detect_calls.append(list(prompts or []))
        return list(self._dets)


def _ctx(image):
    return SimpleNamespace(walkieAI=SimpleNamespace(image=image))


# ---------------------------------------------------------------------------
# _detect_and_lift — plain nearest-wins masked detection
# ---------------------------------------------------------------------------
def test_detect_and_lift_picks_nearest():
    near = _det(_mask((200, 280), (100, 180)))   # near object, 1 m
    far = _det(_mask((200, 280), (460, 540)))    # far object, 2.5 m
    img = _FakeImage([far, near])                # order shouldn't matter
    snap = _snap(_two_object_depth())

    out = _detect_and_lift(
        _ctx(img), snap, ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is not None
    _cloud, range_m, _conf = out
    assert range_m < 1.5                         # nearest-wins selected the NEAR box


def test_detect_and_lift_passes_prompts_through_unexpanded():
    near = _det(_mask((200, 280), (100, 180)))
    img = _FakeImage([near])
    _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert img.detect_calls == [["coke"]]        # no descriptor expansion


def test_detect_and_lift_no_detections_returns_none():
    img = _FakeImage([])
    out = _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is None


def test_detect_and_lift_drops_below_confidence():
    near = _det(_mask((200, 280), (100, 180)), confidence=0.1)
    img = _FakeImage([near])
    out = _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is None                           # only detection filtered by confidence


def test_detect_and_lift_skips_too_sparse():
    # A tiny mask lifts to fewer than min_points -> skipped; the fuller near mask wins.
    tiny = _det(_mask((200, 201), (100, 101)))
    near = _det(_mask((200, 280), (100, 180)))
    img = _FakeImage([tiny, near])
    out = _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is not None
    _cloud, range_m, _conf = out
    assert range_m < 1.5
