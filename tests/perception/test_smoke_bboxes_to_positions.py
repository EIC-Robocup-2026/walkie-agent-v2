"""Smoke: walkie_sdk Tools.bboxes_to_positions contract.

The perception loop will call ``walkie.tools.bboxes_to_positions(coords, timeout)``
and trust the return value as a list of map-frame ``[x, y, z]`` positions
aligned with the input bbox order.

Inputs:  ``coords: List[List[float]]`` — each ``[cx, cy, w, h]`` (image pixels).
Output:  ``Optional[List[List[float]]]`` — list of ``[x, y, z]`` or ``None`` on timeout.

Under the hood (see walkie_sdk/modules/tools.py) this is a publish/subscribe
request-reply:

    publish    /yolo/detections_2d   (vision_msgs/Detection2DArray)
    subscribe  /ob_detection/poses   (geometry_msgs/PoseArray)

The 3D output frame is whatever the upstream YOLO 3D fusion node publishes —
typically ``map`` for a Nav2-aware deployment. The perception loop should NOT
assume the returned positions are camera-relative.

This test does NOT import ``walkie_sdk`` (it's a git dependency that may not
be installed in CI). Instead it asserts the converter helpers used by the
SDK behave as documented, so when we wire the SDK call in for real we can
trust the message shape.
"""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("walkie_sdk") is None,
    reason="walkie_sdk is a git dependency; install via `uv sync` to run this smoke test",
)


def test_converters_roundtrip_bboxes():
    """[cx, cy, w, h] → Detection2DArray dict round-trips to the same numbers."""
    from walkie_sdk.utils import converters

    msg = converters.convert_bboxes_to_detection_array([[320, 240, 50, 80]])
    assert msg["header"]["frame_id"] == "camera_frame"
    assert len(msg["detections"]) == 1
    bbox = msg["detections"][0]["bbox"]
    assert bbox["center"]["position"]["x"] == 320
    assert bbox["center"]["position"]["y"] == 240
    assert bbox["size_x"] == 50
    assert bbox["size_y"] == 80


def test_converters_extract_xyz_from_pose_array():
    """PoseArray → list of [x, y, z] in the order the poses were published."""
    from walkie_sdk.utils import converters

    pose_array_dict = {
        "poses": [
            {"position": {"x": 1.1, "y": 2.2, "z": 0.0}, "orientation": {}},
            {"position": {"x": -0.4, "y": 1.0, "z": 0.3}, "orientation": {}},
        ]
    }
    out = converters.convert_poses_to_array(pose_array_dict)
    assert out == [[1.1, 2.2, 0.0], [-0.4, 1.0, 0.3]]


def test_bboxes_to_positions_timeout_returns_none():
    """If no /ob_detection/poses response arrives within timeout, returns None."""
    from walkie_sdk.modules.tools import Tools

    class _FakeTransport:
        is_connected = True
        def subscribe(self, topic, msg_type, callback, **_):
            return object()
        def unsubscribe(self, _sub): pass
        def publish(self, *a, **kw): pass

    tools = Tools(_FakeTransport())
    out = tools.bboxes_to_positions([[10, 10, 20, 20]], timeout=0.05)
    assert out is None


def test_bboxes_to_positions_returns_xyz_on_response():
    """When the subscribe callback fires with a PoseArray, we get [[x,y,z], ...]."""
    from walkie_sdk.modules.tools import Tools

    captured_callback = {}

    class _FakeTransport:
        is_connected = True
        def subscribe(self, topic, msg_type, callback, **_):
            captured_callback["cb"] = callback
            return object()
        def unsubscribe(self, _sub): pass
        def publish(self, *a, **kw):
            captured_callback["cb"]({
                "poses": [{"position": {"x": 1.0, "y": 2.0, "z": 0.5}, "orientation": {}}]
            })

    tools = Tools(_FakeTransport())
    out = tools.bboxes_to_positions([[10, 10, 20, 20]], timeout=1.0)
    assert out == [[1.0, 2.0, 0.5]]
