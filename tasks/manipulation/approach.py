"""Base + camera positioning for a grasp: drive to a standoff, frame the object.

Drives to the standoff via the SDK's NavigateToObject mode (``nav.go_to`` with
``heading=None``): nav_commander edge-fits the approach pose to the object/surface
and stops a configured distance from the nearest edge, so the robot squares up to
the surface instead of stopping at an arbitrary line-of-sight angle. The lift +
head servo then tilt the camera down so the perception/grasp sees the object. All
best-effort: a positioning failure logs and returns False so the caller can still
attempt the grasp (or degrade); ``refine_approach`` additionally falls back to
just facing the object (``face_point``) when the drive fails.
"""

from __future__ import annotations

import os

from tasks.base import TaskContext
from tasks.skills.navigation import face_point, tilt_head

from .types import Vec3

# go_to status strings that count as "we got there" (CLOSE_ENOUGH needs a
# goal_tolerance, but accept it defensively in case the server promotes one).
_NAV_OK = {"SUCCEEDED", "CLOSE_ENOUGH"}


def _envf(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _align_method() -> str:
    """NavigateToObject alignment: ``nearest_edge`` (default) or ``face_target``."""
    return os.getenv("WALKIE_PICK_ALIGN_METHOD", "nearest_edge").strip() or "nearest_edge"


def _nav_to_object(ctx: TaskContext, target_xy: tuple[float, float], standoff: float) -> bool:
    """Edge-aligned standoff drive via the SDK NavigateToObject (``heading=None``).

    nav_commander PCA edge-fits the approach pose (``align_method``) and stops
    *standoff* metres from the fitted edge. *target_xy* is the object (or surface)
    centroid in the map frame (sent as ``obj_x``/``obj_y``). Returns True only on a
    ``SUCCEEDED``/``CLOSE_ENOUGH`` status; False (logging the status) otherwise, so
    the caller degrades.
    """
    x, y = target_xy
    try:
        status = ctx.walkie.nav.go_to(
            x, y, heading=None, standoff=standoff,
            align_method=_align_method(), blocking=True,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never raise
        print(f"[manipulation.approach] NavigateToObject failed ({exc})")
        return False
    ok = str(status).upper() in _NAV_OK
    if not ok:
        print(f"[manipulation.approach] NavigateToObject -> {status!r}")
    return ok


def drive_to_object(ctx: TaskContext, target_xy: tuple[float, float], standoff: float) -> bool:
    """Drive to *standoff* metres from a map-frame point, edge-aligned to it.

    *target_xy* is the object (or surface) centroid in the map frame. Uses the SDK
    NavigateToObject (nearest_edge) drive; returns False on nav failure (the caller
    degrades).
    """
    return _nav_to_object(ctx, target_xy, standoff)


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

    Same NavigateToObject drive as :func:`drive_to_object` with a smaller standoff,
    so the object is within the arm's workspace. Returns False on nav failure; on
    failure it at least faces the object (``face_point``) so a front grasp stays
    reachable.
    """
    ok = _nav_to_object(ctx, target_xy, standoff)
    if not ok:
        # Last resort: at least face the object so a front grasp is reachable.
        x, y = target_xy
        face_point(ctx, x, y)
    return ok


def viz_nav_target(
    ctx: TaskContext,
    target_xy: tuple[float, float],
    standoff: float,
    *,
    label: str = "nav target",
    ns: str = "manip/nav_target",
    marker_id: int = 400,
    z: float = 0.0,
) -> None:
    """Publish an RViz marker at the map-frame point sent to ``nav.go_to``.

    Lets a tester see where the base is being sent BEFORE the confirm gate fires.
    Draws an orange sphere at *target_xy* (the object/surface centroid passed as
    ``obj_x``/``obj_y``) plus a floating text label with the standoff. The actual
    stop pose is edge-fit by nav_commander, so this marks the TARGET we send, not
    the final base pose. Published on ``walkie/viz_markers`` (add a MarkerArray
    display). Gated by ``WALKIE_MANIP_VIZ`` (default on); best-effort — a viz or
    transport failure logs and returns without affecting the drive.
    """
    if os.getenv("WALKIE_MANIP_VIZ", "1").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        from walkie_sdk import SPHERE, TEXT_VIEW_FACING

        viz = ctx.walkie.robot.viz
        x, y = float(target_xy[0]), float(target_xy[1])
        viz.draw_marker(
            position=[x, y, z], frame_id="map", marker_type=SPHERE,
            scale=[0.10, 0.10, 0.10], color=[1.0, 0.55, 0.0, 0.9],
            marker_id=marker_id, ns=ns,
        )
        viz.draw_marker(
            position=[x, y, z + 0.20], frame_id="map", marker_type=TEXT_VIEW_FACING,
            scale=[0.0, 0.0, 0.07], color=[1.0, 1.0, 1.0, 1.0],
            marker_id=marker_id + 1, ns=ns, text=f"{label} (standoff={standoff:.2f}m)",
        )
        print(f"[manipulation.approach] nav target marker {label!r} at map "
              f"({x:.2f},{y:.2f}) standoff={standoff:.2f}m on walkie/viz_markers")
    except Exception as exc:  # noqa: BLE001 — viz must never disturb the drive
        print(f"[manipulation.approach] nav target viz failed ({exc})")
