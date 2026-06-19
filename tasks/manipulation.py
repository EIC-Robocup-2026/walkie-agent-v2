"""Shared manipulation primitives for the on-robot tasks (Pick and Place, Restaurant, ...).

Plain functions over a TaskContext — no state. This is the "graduated" home for
the grasp/pick/place logic that started in tasks/PickAndPlace and is now reused
across manipulation tasks.

Stub boundary (agreed with the team): the *grasp planner* (`plan_grasp`) is a
heuristic — given a detected object's 3D position it returns the position +
rotation the hand must move to — but the robot really commands the arm to that
pose and really opens/closes the gripper ("command the hand as if picking
something up"). The grasp orientations and any destination drop poses are
geometry-specific placeholders read from config and **need on-robot
calibration** before a real run.

Robot-wide knobs live in the root config.toml under `[manipulation]`
(WALKIE_*); task-specific bits (which classes to detect, where to place) are
passed in by the caller.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tasks.base import TaskContext

BBox = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]

# Generic narration owned here so every task picks the same way. Callers narrate
# their own place/serve lines (place_at_pose is silent).
PICKING = "I am picking up the {obj}."
PICK_NO_PLAN = "I cannot work out how to grasp the {obj}, so I will skip it for now."


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


# --- perception -------------------------------------------------------------
def perceive_surface(ctx: TaskContext, classes: list[str]) -> list[DetectedObject]:
    """Detect objects on the surface in front of the robot (open-vocab).

    *classes* are the open-vocabulary detector prompts (the caller supplies the
    task-specific list). Returns DetectedObjects with bboxes; world_xy/world_xyz
    are lifted against the snapshot geometry when available. Empty list on any
    capture/detection failure (the task degrades, never raises).
    """
    snap = ctx.snapshot()
    if snap is None:
        return []
    try:
        detections = ctx.walkieAI.image.detect(snap.img, prompts=classes)
    except Exception as exc:
        print(f"[manipulation] detection failed ({exc})")
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
    robot-specific — calibrate WALKIE_GRASP_RPY_* on the real arm.
    """
    if obj.world_xyz is None:
        return None
    frame = _arm_frame()
    centroid = obj.world_xyz if frame == "map" else world_to_base(ctx, obj.world_xyz)
    cx, cy, cz = centroid
    approach = os.getenv("WALKIE_GRASP_APPROACH", "top_down").strip().lower()
    z_off = float(os.getenv("WALKIE_GRASP_Z_OFFSET_M", "0.0"))
    if approach == "front":
        rot = _parse3(os.getenv("WALKIE_GRASP_RPY_FRONT", "-0.8,0.0,-1.5708"))
        return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="front")
    rot = _parse3(os.getenv("WALKIE_GRASP_RPY_TOPDOWN", "-2.623,-0.033,-1.468"))
    return GraspPlan((cx, cy, cz + z_off), rot, frame_id=frame, approach="top_down")


# --- motion execution -------------------------------------------------------
def _pregrasp(position: Vec3, approach: str) -> Vec3:
    """Back the grasp/place point off to a safe pre-approach waypoint."""
    x, y, z = position
    off = float(os.getenv("WALKIE_PREGRASP_OFFSET_M", "0.10"))
    if approach == "front":
        return x - off, y, z  # pull back toward the robot (+x is forward)
    return x, y, z + off  # top-down: come down from above


def _carry_arm(ctx: TaskContext, group) -> None:
    """Tuck the arm into a carry/standby pose so the base can nav safely."""
    pose_name = os.getenv("WALKIE_CARRY_POSE", "standby").strip()
    try:
        group.go_to_home(pose_name=pose_name, blocking=True)
    except Exception as exc:
        print(f"[manipulation] carry/home ({pose_name}) failed ({exc})")


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
        print(f"[manipulation] grasp() failed ({exc}); closing gripper directly")
        group.gripper(0.0, blocking=True)
    lift = float(os.getenv("WALKIE_LIFT_HEIGHT_M", "0.15"))
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
        ctx.say(PICK_NO_PLAN.format(obj=obj.class_name))
        print(f"[manipulation] pick_object({obj.class_name}) — no 3D grasp plan")
        return False
    ctx.say(PICKING.format(obj=obj.class_name))  # also the perception signal
    print(f"[manipulation] grasp plan for {obj.class_name}: pos={plan.position} "
          f"rpy={plan.rotation} frame={plan.frame_id} approach={plan.approach}")
    try:
        return _execute_grasp(ctx, _arm_group(ctx), plan)
    except Exception as exc:
        print(f"[manipulation] pick_object({obj.class_name}) arm motion failed ({exc})")
        _carry_arm(ctx, _arm_group(ctx))
        return False


def place_at_pose(ctx: TaskContext, pose6: str) -> bool:
    """Release the held object at a fixed arm-frame pose 'x,y,z,roll,pitch,yaw'.

    Silent (the caller narrates). Returns False on a bad/empty pose string —
    falling back to release_in_front so the object isn't stuck in the gripper —
    or on motion failure (best-effort, never raises).
    """
    raw = (pose6 or "").strip()
    if not raw:
        print("[manipulation] place_at_pose — empty pose; releasing in front")
        return release_in_front(ctx)
    try:
        x, y, z, r, p, yw = _parse6(raw)
    except ValueError as exc:
        print(f"[manipulation] place_at_pose — bad pose ({exc})")
        return False
    plan = GraspPlan((x, y, z), (r, p, yw), frame_id=_arm_frame())
    arm = _arm_group(ctx)
    try:
        return _execute_place(ctx, arm, plan)
    except Exception as exc:
        print(f"[manipulation] place_at_pose arm motion failed ({exc})")
        _carry_arm(ctx, arm)
        return False


def release_in_front(ctx: TaskContext) -> bool:
    """Fallback drop: open the gripper to release in front, then tuck the arm.

    Used when no place pose is configured. Returns False so the caller knows the
    placement was not a deliberate, located put-down.
    """
    arm = _arm_group(ctx)
    try:
        arm.gripper(1.0, blocking=True)  # release best-effort
    except Exception as exc:
        print(f"[manipulation] release_in_front gripper failed ({exc})")
    _carry_arm(ctx, arm)
    return False
