"""Position lifters — turn 2D detections into a 3D map-frame position.

The canonical lifter is the SDK's ``walkie.tools.bboxes_to_positions``
(depth + TF via the robot's ``get_3d_poses`` service). When that service is
unavailable or unreliable, :class:`RobotPoseLifter` is a coarse stand-in: it
ignores bbox geometry and reports the robot's *own* current map-frame pose
for every detection — i.e. "this object was seen from roughly here". Good
enough for a spatial memory that only needs to send the robot back to the
right area; swap back to the SDK lifter once ``get_3d_poses`` is trustworthy.

Both satisfy the :class:`perception.types.PositionLifter` protocol.
"""

from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("perception.lifters")


class RobotPoseLifter:
    """A :class:`~perception.types.PositionLifter` backed by robot odometry.

    Returns the robot's current ``[x, y, z]`` (``z`` is always 0 — odometry
    is planar) for *every* bbox, regardless of where in the frame it sits.
    Returns ``None`` when no odometry is available yet, mirroring the SDK
    lifter's timeout contract so the pipeline skips the tick rather than
    storing a bogus origin position.

    Args:
        status: the SDK telemetry module (``walkie.status``). Only
            ``get_position()`` is used; it must return a dict with ``x``/``y``
            (as ``WalkieRobot.status`` does) or ``None``.
    """

    def __init__(self, status) -> None:
        self._status = status

    def bboxes_to_positions(
        self,
        coords: list[list[float]],
        timeout: float = 5.0,
    ) -> Optional[list[list[float]]]:
        pose = self._status.get_position()
        if not pose:
            _log.info("RobotPoseLifter: no odometry yet; skipping tick (None)")
            return None
        xyz = [float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), 0.0]
        # Same position for each detection — they were all seen from here.
        return [list(xyz) for _ in coords]
