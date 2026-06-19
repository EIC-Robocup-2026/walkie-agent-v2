"""Build a ROS ``PointCloud2`` message dict from a numpy ``(N, 3)`` XYZ cloud.

The SDK ships :func:`walkie_sdk.utils.converters.parse_point_cloud_xyz`
(PointCloud2 -> numpy) but no inverse. The DB-driven grasp path needs the
inverse: the object's stored world cloud (``GraphMemory.load_pcd``) is a numpy
``(N, 3)`` array and ``walkie.robot.grasp.from_cloud`` wants a PointCloud2 dict.

The wire encoding of the ``data`` field differs by transport — zenoh wants raw
bytes, rosbridge wants a base64 string (or a uint8 list). Default to ``bytes``;
override with ``WALKIE_PC2_DATA_ENCODING`` (bytes|base64|list) if the grasp
server can't parse what we send (see the calibration note in the design).
"""

from __future__ import annotations

import base64
import os

import numpy as np

# sensor_msgs/PointField datatype enum: 7 = FLOAT32 (4 bytes).
_FLOAT32 = 7
_POINT_STEP = 12  # 3 * float32


def _xyz_fields() -> list[dict]:
    return [
        {"name": "x", "offset": 0, "datatype": _FLOAT32, "count": 1},
        {"name": "y", "offset": 4, "datatype": _FLOAT32, "count": 1},
        {"name": "z", "offset": 8, "datatype": _FLOAT32, "count": 1},
    ]


def _encode(raw: bytes) -> object:
    enc = os.getenv("WALKIE_PC2_DATA_ENCODING", "bytes").strip().lower()
    if enc == "base64":
        return base64.b64encode(raw).decode("ascii")
    if enc == "list":
        return list(raw)
    return raw  # "bytes" (default) — zenoh transport


def numpy_to_pointcloud2(points_xyz: np.ndarray, frame_id: str = "map") -> dict:
    """Pack an ``(N, 3)`` float array into an unorganized XYZ PointCloud2 dict.

    Args:
        points_xyz: ``(N, 3)`` array of XYZ points in *frame_id*.
        frame_id: the cloud's reference frame (e.g. ``"map"``). The grasp server
            transforms it into the planning frame.

    Returns:
        A PointCloud2 message dict round-trippable through
        :func:`walkie_sdk.utils.converters.parse_point_cloud_xyz`.
    """
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n = int(pts.shape[0])
    raw = pts.tobytes()  # row-major: x0,y0,z0, x1,y1,z1, ... (little-endian f32)
    return {
        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": frame_id},
        "height": 1,
        "width": n,
        "fields": _xyz_fields(),
        "is_bigendian": False,
        "point_step": _POINT_STEP,
        "row_step": _POINT_STEP * n,
        "data": _encode(raw),
        "is_dense": True,
    }
