"""Reusable perception / manipulation skills for Pick and Place (rulebook 5.2).

Plain functions over a TaskContext, mirroring tasks/HRI/skills.py.

Manipulation here follows the agreed stub boundary: the *grasp planner*
(``plan_grasp`` / ``plan_place``) is a heuristic — given a detected object's 3D
position it returns the position + rotation the hand must move to — but the
robot really commands the arm to that pose and really opens/closes the gripper
("command the hand as if picking something up"). The exact grasp orientations
and the destination drop poses are geometry-specific placeholders read from
config and **need on-robot calibration** before a real run.

Anything that turns out generic enough (a real pick/place) should graduate to
tasks/base.py so the other manipulation tasks (Laundry, Restaurant) can lift it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tasks.base import TaskContext

from . import prompts

BBox = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]


@dataclass
class DetectedObject:
    """One object perceived on a surface, with its map-frame position when lifted."""

    bbox_xyxy: BBox
    class_name: str
    confidence: float
    world_xy: tuple[float, float] | None = None
    world_xyz: Vec3 | None = None  # full map-frame centroid (x, y, z); z = height


@dataclass
class GraspPlan:
    """Where the hand must go to grasp/place — the output of the grasp stub.

    ``position``/``rotation`` are an end-effector target (metres / RPY radians)
    expressed in ``frame_id`` (``base_footprint`` or ``map``). ``approach`` tells
    the executor how to back off for the pre-grasp/pre-place waypoint.
    """

    position: Vec3
    rotation: Vec3
    frame_id: str = "base_footprint"
    approach: str = "top_down"  # "top_down" | "front"


# --- config helpers ---------------------------------------------------------
def _classes(env: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env, default).split(",") if c.strip()]


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
    """The ArmGroup (arm.left / arm.right) selected by PNP_ARM."""
    side = os.getenv("PNP_ARM", "left").strip().lower()
    arm = ctx.walkie.robot.arm
    return arm.right if side == "right" else arm.left


def _arm_frame() -> str:
    return os.getenv("PNP_ARM_FRAME", "base_footprint").strip() or "base_footprint"


# --- perception -------------------------------------------------------------
def perceive_surface(ctx: TaskContext, classes: list[str] | None = None) -> list[DetectedObject]:
    """Detect objects on the surface in front of the robot (open-vocab).

    Returns DetectedObjects with bboxes; world_xy/world_xyz are lifted against
    the snapshot geometry when available. Empty list on any capture/detection
    failure (the task degrades, never raises). The rulebook requires the robot
    to *communicate its perception* to the referee — the caller should announce
    the result.
    """
    snap = ctx.snapshot()
    if snap is None:
        return []
    prompts_list = classes or _classes(
        "PNP_TABLE_CLASSES", "cup,mug,plate,fork,knife,spoon,bottle,box,can,bowl"
    )
    try:
        detections = ctx.walkieAI.object_detection.detect(snap.img, prompts=prompts_list)
    except Exception as exc:
        print(f"[pnp.skills] detection failed ({exc})")
        return []
    out: list[DetectedObject] = []
    for det in detections:
        world_xyz = None
        if getattr(snap, "has_geometry", False):
            try:
                world_xyz = snap.bbox_world_point(det.bbox)
            except Exception:
                world_xyz = None
        world_xy = tuple(world_xyz[:2]) if world_xyz is not None else None
        out.append(
            DetectedObject(
                bbox_xyxy=tuple(det.bbox),
                class_name=det.class_name or "object",
                confidence=det.confidence or 0.0,
                world_xy=world_xy,
                world_xyz=tuple(world_xyz) if world_xyz is not None else None,
            )
        )
    return out


def sort_object(ctx: TaskContext, obj: DetectedObject) -> prompts.ObjectSort | None:
    """LLM decision: dishwasher vs trash vs cabinet (+ group) for one object.

    Real glue — the categorisation is pure reasoning over the class name and the
    run's designated trash category, so it works today. Returns None on
    extraction failure (caller should default to the cabinet).
    """
    trash_category = os.getenv("PNP_TRASH_CATEGORY", "").strip() or "(announced at setup)"
    text = (
        f"Object class: {obj.class_name}. "
        f"Designated trash category for this run: {trash_category}."
    )
    return ctx.extract(prompts.ObjectSort, prompts.SORT_OBJECT_INSTRUCTIONS, text)


# --- frame transform --------------------------------------------------------
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


# --- grasp planning (THE STUB) ----------------------------------------------
def plan_grasp(ctx: TaskContext, obj: DetectedObject) -> GraspPlan | None:
    """Heuristic grasp planner: object 3D position -> hand target pose.

    This is the deliberately-stubbed part. It does NOT run a learned
    grasp-quality network; it places the end-effector at the object's centroid
    with a configured orientation (top-down by default, or a front/horizontal
    approach). Returns None when the object was never lifted to 3D, so the caller
    can degrade gracefully. The orientations come from config and are
    robot-specific — calibrate PNP_GRASP_RPY_* on the real arm.
    """
    if obj.world_xyz is None:
        return None
    frame = _arm_frame()
    centroid = obj.world_xyz if frame == "map" else world_to_base(ctx, obj.world_xyz)
    cx, cy, cz = centroid
    approach = os.getenv("PNP_GRASP_APPROACH", "top_down").strip().lower()
    z_off = float(os.getenv("PNP_GRASP_Z_OFFSET_M", "0.0"))
    if approach == "front":
        rot = _parse3(os.getenv("PNP_GRASP_RPY_FRONT", "-0.8,0.0,-1.5708"))
        return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="front")
    rot = _parse3(os.getenv("PNP_GRASP_RPY_TOPDOWN", "-2.623,-0.033,-1.468"))
    return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="top_down")


def plan_place(ctx: TaskContext, destination: str, *, group: str | None = None) -> GraspPlan | None:
    """Drop pose for a destination ('dishwasher'|'trash'|'cabinet').

    We have no 3D detection of furniture interiors, so the place pose is read
    from config (PNP_PLACE_POSE_<DEST> = 'x,y,z,roll,pitch,yaw' in the arm
    frame), falling back to a generic forward reach (PNP_PLACE_POSE_DEFAULT).
    Returns None only when neither is set. Calibrate on the real arena.
    """
    env_key = {
        "dishwasher": "PNP_PLACE_POSE_DISHWASHER",
        "trash": "PNP_PLACE_POSE_TRASH",
        "cabinet": "PNP_PLACE_POSE_CABINET",
    }.get(destination)
    raw = (os.getenv(env_key, "").strip() if env_key else "") or os.getenv(
        "PNP_PLACE_POSE_DEFAULT", ""
    ).strip()
    if not raw:
        return None
    x, y, z, r, p, yw = _parse6(raw)
    return GraspPlan((x, y, z), (r, p, yw), frame_id=_arm_frame(), approach="top_down")


# --- motion execution -------------------------------------------------------
def _pregrasp(position: Vec3, approach: str) -> Vec3:
    """Back the grasp/place point off to a safe pre-approach waypoint."""
    x, y, z = position
    off = float(os.getenv("PNP_PREGRASP_OFFSET_M", "0.10"))
    if approach == "front":
        return x - off, y, z  # pull back toward the robot (+x is forward)
    return x, y, z + off  # top-down: come down from above


def _carry_arm(ctx: TaskContext, group) -> None:
    """Tuck the arm into a carry/standby pose so the base can nav safely."""
    pose_name = os.getenv("PNP_CARRY_POSE", "standby").strip()
    try:
        group.go_to_home(pose_name=pose_name, blocking=True)
    except Exception as exc:
        print(f"[pnp.skills] carry/home ({pose_name}) failed ({exc})")


def _execute_grasp(ctx: TaskContext, group, plan: GraspPlan) -> bool:
    """Open -> pre-grasp -> grasp pose -> close -> lift -> carry. Returns grasped."""
    group.gripper(1.0, blocking=True)  # open, ready to receive
    group.go_to_pose(
        _pregrasp(plan.position, plan.approach), plan.rotation,
        frame_id=plan.frame_id, blocking=True,
    )
    group.go_to_pose(
        plan.position, plan.rotation,
        frame_id=plan.frame_id, cartesian_path=True, blocking=True,
    )
    grasped = True
    try:
        result = group.grasp()  # close on the object; judge by 'grasped'
        grasped = bool(result.get("grasped", True))
    except Exception as exc:
        print(f"[pnp.skills] grasp() failed ({exc}); closing gripper directly")
        group.gripper(0.0, blocking=True)
    lift = float(os.getenv("PNP_LIFT_HEIGHT_M", "0.15"))
    group.go_to_pose_relative([0.0, 0.0, lift], [0.0, 0.0, 0.0], blocking=True)
    _carry_arm(ctx, group)
    return grasped


def _execute_place(ctx: TaskContext, group, plan: GraspPlan) -> bool:
    """Pre-place -> place pose -> open (release) -> carry -> close. Returns True."""
    group.go_to_pose(
        _pregrasp(plan.position, plan.approach), plan.rotation,
        frame_id=plan.frame_id, blocking=True,
    )
    group.go_to_pose(
        plan.position, plan.rotation,
        frame_id=plan.frame_id, cartesian_path=True, blocking=True,
    )
    group.gripper(1.0, blocking=True)  # open: release the object
    _carry_arm(ctx, group)
    group.gripper(0.0, blocking=False)  # close so the gripper isn't left hanging open
    return True


# --- public manipulation primitives -----------------------------------------
def pick_object(ctx: TaskContext, obj: DetectedObject) -> bool:
    """Grasp *obj* and lift it for transport. Real arm motion; planner is the stub.

    Plans a grasp pose, then commands the arm (open -> pre-grasp -> grasp ->
    close -> lift -> carry). Degrades to False (announces, never raises) when the
    object has no 3D position or any arm command fails, so the caller can
    score-degrade gracefully (partial scoring is allowed).
    """
    plan = plan_grasp(ctx, obj)
    if plan is None:
        ctx.say(prompts.PICK_NOT_AVAILABLE.format(obj=obj.class_name))
        print(f"[pnp.skills] pick_object({obj.class_name}) — no 3D grasp plan")
        return False
    ctx.say(prompts.PICKING.format(obj=obj.class_name))  # also the perception signal
    print(f"[pnp.skills] grasp plan for {obj.class_name}: pos={plan.position} "
          f"rpy={plan.rotation} frame={plan.frame_id} approach={plan.approach}")
    try:
        return _execute_grasp(ctx, _arm_group(ctx), plan)
    except Exception as exc:
        print(f"[pnp.skills] pick_object({obj.class_name}) arm motion failed ({exc})")
        _carry_arm(ctx, _arm_group(ctx))
        return False


def place_object(ctx: TaskContext, destination: str, *, group: str | None = None) -> bool:
    """Navigate to *destination* furniture and release the held object there.

    Navigation to the furniture waypoint is real (config poses); the place pose
    is the stub (config PNP_PLACE_POSE_<DEST>). Returns False if no place pose is
    configured (opens the gripper to drop in front as a fallback) or on motion
    failure.
    """
    pose_key = {
        "dishwasher": "PNP_DISHWASHER_POSE",
        "trash": "PNP_TRASH_BIN_POSE",
        "cabinet": "PNP_CABINET_POSE",
    }.get(destination)
    if pose_key and os.getenv(pose_key):
        from .subtasks import _pose  # local import: shared pose parser

        x, y, h = _pose(pose_key)
        ctx.goto(x, y, h)
    plan = plan_place(ctx, destination, group=group)
    arm = _arm_group(ctx)
    if plan is None:
        print(f"[pnp.skills] place_object({destination}) — no place pose; dropping in front")
        arm.gripper(1.0, blocking=True)  # release best-effort
        _carry_arm(ctx, arm)
        return False
    try:
        ok = _execute_place(ctx, arm, plan)
        ctx.say(prompts.PLACED.format(destination=destination))
        return ok
    except Exception as exc:
        print(f"[pnp.skills] place_object({destination}) arm motion failed ({exc})")
        _carry_arm(ctx, arm)
        return False


def place_at(ctx: TaskContext, pose6: str) -> bool:
    """Release the held object at a fixed arm-frame pose 'x,y,z,roll,pitch,yaw'.

    Used by ServeBreakfast to drop each item at its laid-out slot. Returns False
    on a bad/empty pose string or motion failure (best-effort, never raises).
    """
    raw = (pose6 or "").strip()
    arm = _arm_group(ctx)
    if not raw:
        print("[pnp.skills] place_at — empty pose; dropping in front")
        arm.gripper(1.0, blocking=True)
        _carry_arm(ctx, arm)
        return False
    try:
        x, y, z, r, p, yw = _parse6(raw)
    except ValueError as exc:
        print(f"[pnp.skills] place_at — bad pose ({exc})")
        return False
    plan = GraspPlan((x, y, z), (r, p, yw), frame_id=_arm_frame())
    try:
        return _execute_place(ctx, arm, plan)
    except Exception as exc:
        print(f"[pnp.skills] place_at arm motion failed ({exc})")
        _carry_arm(ctx, arm)
        return False
