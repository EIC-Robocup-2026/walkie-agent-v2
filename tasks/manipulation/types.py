"""Shared manipulation types + small config/arm helpers (no robot I/O at import).

Moved out of the old single-file ``tasks/manipulation.py`` when it grew into a
package. Holds the data carriers every manipulation submodule passes around plus
the tiny env/arm-group helpers they all need.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tasks.base import TaskContext

BBox = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass
class DetectedObject:
    """One object perceived on a surface, with its map-frame position when lifted."""

    bbox_xyxy: BBox
    class_name: str
    confidence: float
    world_xy: tuple[float, float] | None = None
    world_xyz: Vec3 | None = None  # full map-frame centroid (x, y, z); z = height
    node_id: str | None = None  # walkie_graphs node this came from, when DB-resolved


@dataclass
class GraspPlan:
    """Where the hand must go to grasp/place — output of the grasp planner.

    ``position``/``rotation`` are an end-effector target (metres / RPY radians)
    expressed in ``frame_id`` (``base_footprint`` or ``map``). ``approach`` is the
    coarse strategy used by the stub's heuristic pre-grasp offset.

    The GraspNet path additionally fills ``quaternion`` (the EE orientation as a
    quaternion, preferred over ``rotation`` when present) and ``approach_position``
    (the pre-grasp/pre-place waypoint GraspNet returns). The executor uses
    ``go_to_pose_quat`` + ``approach_position`` whenever ``quaternion`` is set, and
    falls back to ``rotation`` + a heuristic offset for the stub path.
    """

    position: Vec3
    rotation: Vec3
    frame_id: str = "base_footprint"
    approach: str = "top_down"  # "top_down" | "front"
    quaternion: Quat | None = None
    approach_position: Vec3 | None = None
    width: float | None = None  # GraspNet gripper opening (m), when known
    score: float | None = None  # GraspNet quality, when known


# --- config helpers ---------------------------------------------------------
def _parse3(s: str) -> Vec3:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'a,b,c', got {s!r}")
    a, b, c = (float(p) for p in parts)
    return a, b, c


def _parse6(s: str) -> tuple[float, float, float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"expected 'x,y,z,roll,pitch,yaw', got {s!r}")
    x, y, z, r, p, yw = (float(v) for v in parts)
    return x, y, z, r, p, yw


def _arm_group(ctx: TaskContext):
    """The ArmGroup (arm.left / arm.right) selected by WALKIE_ARM."""
    side = os.getenv("WALKIE_ARM", "left").strip().lower()
    arm = ctx.walkie.robot.arm
    return arm.right if side == "right" else arm.left


def _arm_frame() -> str:
    return os.getenv("WALKIE_ARM_FRAME", "base_footprint").strip() or "base_footprint"


def world_to_base(ctx: TaskContext, xyz_map: Vec3) -> Vec3:
    """Convert a map-frame point to the robot's base_footprint frame.

    Map and base_footprint share the gravity (z) axis and base_footprint sits on
    the floor under the robot, so z passes straight through; x/y are rotated by
    the robot's heading and translated by its position (from current odometry).
    """
    ox, oy, oz = xyz_map
    pose = ctx.current_pose()
    rx, ry, th = pose["x"], pose["y"], pose["heading"]
    dx, dy = ox - rx, oy - ry
    cos_t, sin_t = math.cos(th), math.sin(th)
    bx = cos_t * dx + sin_t * dy
    by = -sin_t * dx + cos_t * dy
    return bx, by, oz
