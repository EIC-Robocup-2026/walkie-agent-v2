"""Base + camera positioning for a grasp: drive to a standoff, frame the object.

Reuses the shared navigation skills (``approach_point``/``face_point``) to stop a
configured distance short of the object and face it, and the lift + head servo to
tilt the camera down onto the surface so the perception/grasp sees the object.
All best-effort: a positioning failure logs and returns False so the caller can
still attempt the grasp (or degrade).
"""

from __future__ import annotations

import os

from tasks.base import TaskContext
from tasks.skills.navigation import approach_point, face_point, tilt_head

from .types import Vec3


def _envf(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def drive_to_object(ctx: TaskContext, target_xy: tuple[float, float], standoff: float) -> bool:
    """Drive to *standoff* metres short of a map-frame point, facing it.

    *target_xy* is the object (or surface) centroid in the map frame. Returns
    False on odometry/nav failure (the caller degrades).
    """
    x, y = target_xy
    return approach_point(ctx, x, y, stop_distance=standoff, blocking=True)


def aim_camera_at_object(ctx: TaskContext, target_xyz: Vec3 | None = None) -> None:
    """Raise the lift and tilt the head down so the camera frames the object.

    Uses fixed config presets (``WALKIE_PICK_LIFT_NORM`` / ``WALKIE_PICK_HEAD_TILT_RAD``)
    that must be calibrated on the robot for the working table height. *target_xyz*
    is accepted for a future height-aware tilt; today only the surface presets are
    applied. Best-effort — never raises (an off-robot stub may lack lift/head).
    """
    lift_norm = _envf("WALKIE_PICK_LIFT_NORM", "0.5")
    tilt_rad = _envf("WALKIE_PICK_HEAD_TILT_RAD", "0.5")
    settle = _envf("WALKIE_PICK_CAMERA_SETTLE_SEC", "1.0")
    try:
        ctx.walkie.robot.lift.set(lift_norm, blocking=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.approach] lift.set({lift_norm}) failed ({exc})")
    # POSITIVE tilt = camera down (walkie-sdk convention); settle before capture.
    tilt_head(ctx, tilt_rad, settle=settle)


def refine_approach(ctx: TaskContext, target_xy: tuple[float, float], standoff: float) -> bool:
    """Creep in to the close grasp standoff once the grasp is planned.

    Same geometry as :func:`drive_to_object` with a smaller stop distance, so the
    object is within the arm's workspace. Returns False on nav/odometry failure.
    """
    x, y = target_xy
    ok = approach_point(ctx, x, y, stop_distance=standoff, blocking=True)
    if not ok:
        # Last resort: at least face the object so a front grasp is reachable.
        face_point(ctx, x, y)
    return ok
