"""Perception lift + the real pick/place motion sequences.

``pick_object`` / ``place_at_pose`` command the real arm + gripper. On the
GraspNet planner they run the full collision-aware sequence (standoff approach,
camera aim, table collision box, GraspNet pose, scene attach/detach, octomap
allowance); on the stub planner they behave exactly as before (centroid + fixed
RPY, no scene/approach extras) so offline arm testing is unchanged.

Error philosophy matches the task framework: every step degrades (announces /
logs, never raises) so a partial failure scores partially instead of aborting.
"""

from __future__ import annotations

import math
import os

from tasks.base import TaskContext

from . import db, scene
from .approach import aim_camera_at_object, drive_to_object, refine_approach, viz_nav_target
from .grasp import plan_grasp
from .types import (
    DetectedObject,
    GraspPlan,
    Quat,
    Vec3,
    _arm_frame,
    _arm_group,
    _parse6,
)
from walkie_sdk.utils.converters import euler_to_quaternion, quaternion_multiply

# Generic narration owned here so every task picks the same way. Callers narrate
# their own place/serve lines (place_at_pose is silent).
PICKING = "I am picking up the {obj}."
PICK_NO_PLAN = "I cannot work out how to grasp the {obj}, so I will skip it for now."


def _envf(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _rich() -> bool:
    """True when the full collision-aware sequence should run (GraspNet planner)."""
    return os.getenv("WALKIE_GRASP_PLANNER", "graspnet").strip().lower() != "stub"


def _nav_label(verb: str, xy: tuple[float, float], standoff: float) -> str:
    """Confirm-gate label naming the map-frame ``go_to`` target + standoff.

    So a tester gating the drive sees where the base is being sent even without
    RViz open (it pairs with :func:`viz_nav_target`'s marker).
    """
    return f"{verb} @ map({xy[0]:.2f},{xy[1]:.2f}) standoff={standoff:.2f}m"


# --- tester confirmation gate -----------------------------------------------
class _AbortManipulation(Exception):
    """Raised by the confirm gate when the tester aborts the sequence ('q')."""


def _confirm(label: str) -> bool:
    """Tester gate before a robot action (enabled by WALKIE_MANIP_CONFIRM=1).

    Prints what is about to happen and waits for the tester: Enter performs the
    action, ``s`` skips just this action, ``q`` aborts the whole pick/place. A
    no-op (always True) when the gate is off, so normal runs are unaffected.
    """
    if os.getenv("WALKIE_MANIP_CONFIRM", "0").strip().lower() not in ("1", "true", "yes"):
        return True
    try:
        ans = input(f"[confirm] about to {label} — Enter=do, s=skip, q=abort: ").strip().lower()
    except EOFError:
        return True
    if ans == "q":
        raise _AbortManipulation(label)
    if ans == "s":
        print(f"[manipulation] tester skipped: {label}")
        return False
    return True


def _gated(label: str, fn, *args, **kwargs):
    """Run *fn* only if the tester confirms *label*; otherwise skip it."""
    if _confirm(label):
        return fn(*args, **kwargs)
    return None


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
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation] detection failed ({exc})")
        return []
    out: list[DetectedObject] = []
    for det in detections:
        world_xyz = None
        if getattr(snap, "has_geometry", False):
            try:
                world_xyz = snap.bbox_world_point(det.bbox)
            except Exception:  # noqa: BLE001
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


# --- motion helpers ---------------------------------------------------------
def _move_ee(group, position: Vec3, plan: GraspPlan, *, cartesian: bool = False) -> str:
    """Send the EE to *position* with the plan's orientation (quat preferred)."""
    if plan.quaternion is not None:
        return group.go_to_pose_quat(
            position, plan.quaternion,
            frame_id=plan.frame_id, cartesian_path=cartesian, blocking=True,
        )
    return group.go_to_pose(
        position, plan.rotation,
        frame_id=plan.frame_id, cartesian_path=cartesian, blocking=True,
    )


# --- grasp orientation: flip-retry + fixed (robot-frame) override -----------
# GraspNet's returned EE orientation is frequently IK-unsolvable on OpenArm. Two
# escape hatches, both grasp-only (the place path is untouched):
#   * flip-retry — a parallel-jaw grasp is physically identical with the gripper
#     rotated 180° about its approach axis (fingers swapped). On a FAILED move we
#     retry that flipped pose, which is often IK-solvable. Decided once at the
#     free-planning pre-grasp move and locked for the cartesian grasp descent, so
#     the wrist never twists mid-approach.
#   * fixed override (WALKIE_GRASP_FIX_QUAT) — a known-good wrist quaternion given
#     in the ROBOT (base_footprint) frame and converted into the plan frame via the
#     robot heading, so it tracks the body instead of a map-frame constant.
def _flip_enabled() -> bool:
    return os.getenv("WALKIE_GRASP_TRY_FLIP", "1").strip().lower() in ("1", "true", "yes")


def _move_ok(result) -> bool:
    """True when an SDK move result indicates success ('SUCCEEDED')."""
    if isinstance(result, str):
        return result.strip().upper() == "SUCCEEDED"
    if isinstance(result, dict):
        status = str(result.get("status", "")).upper()
        return bool(result.get("success")) or status == "SUCCEEDED"
    return bool(result)


def _fix_quat() -> Quat | None:
    """Parse WALKIE_GRASP_FIX_QUAT='qx,qy,qz,qw' (base_footprint frame); None if unset/bad."""
    raw = os.getenv("WALKIE_GRASP_FIX_QUAT", "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        print(f"[manipulation] WALKIE_GRASP_FIX_QUAT ignored — expected 'qx,qy,qz,qw', got {raw!r}")
        return None
    try:
        qx, qy, qz, qw = (float(p) for p in parts)
    except ValueError:
        print(f"[manipulation] WALKIE_GRASP_FIX_QUAT ignored — non-numeric {raw!r}")
        return None
    return qx, qy, qz, qw


def _orient_in_frame(ctx: TaskContext, q_base: Quat, frame: str) -> Quat:
    """Express a base_footprint EE orientation in *frame* (identity for base_footprint).

    map and base_footprint differ only by the robot heading (yaw about shared
    gravity-z), so base->map is a left-multiply by the heading quaternion.
    """
    if frame == "base_footprint":
        return q_base
    heading = float(ctx.current_pose().get("heading", 0.0))
    q_yaw = euler_to_quaternion(0.0, 0.0, heading)  # base -> map orientation
    return quaternion_multiply(q_yaw, q_base)


def _approach_axis_world(plan: GraspPlan) -> Vec3:
    """Unit approach direction in the plan frame: from approach_position toward the grasp.

    Falls back to the coarse approach strategy (front -> +x, top_down -> -z) when
    GraspNet gave no approach waypoint, matching :func:`_pregrasp`'s geometry.
    """
    if plan.approach_position is not None:
        px, py, pz = plan.position
        ax, ay, az = plan.approach_position
        ux, uy, uz = px - ax, py - ay, pz - az
        norm = math.sqrt(ux * ux + uy * uy + uz * uz)
        if norm > 1e-9:
            return ux / norm, uy / norm, uz / norm
    return (1.0, 0.0, 0.0) if plan.approach == "front" else (0.0, 0.0, -1.0)


def _flip_quat_about_approach(q: Quat, plan: GraspPlan) -> Quat:
    """Rotate the gripper 180° about its approach axis (fingers swapped, same grasp).

    The 180° flip about unit axis ``u`` is the pure quaternion ``(ux,uy,uz,0)``;
    world pre-multiply leaves the approach direction unchanged.
    """
    ux, uy, uz = _approach_axis_world(plan)
    return quaternion_multiply((ux, uy, uz, 0.0), q)


def _base_orientation(ctx: TaskContext, plan: GraspPlan) -> Quat | None:
    """The first orientation to try: the fixed override (converted) or GraspNet's own."""
    q_fix = _fix_quat()
    if q_fix is not None:
        q = _orient_in_frame(ctx, q_fix, plan.frame_id)
        print(f"[manipulation] fixed EE orientation override (base_footprint) {q_fix} "
              f"-> {plan.frame_id} {q}")
        return q
    return plan.quaternion


def _orient_candidates(ctx: TaskContext, plan: GraspPlan) -> list[Quat | None]:
    """Ordered orientations to try: [base] then the flipped gripper when enabled.

    Returns ``[None]`` for the stub/RPY path (orientation comes from plan.rotation).
    """
    base = _base_orientation(ctx, plan)
    if base is None:
        return [None]
    candidates: list[Quat | None] = [base]
    if _flip_enabled():
        candidates.append(_flip_quat_about_approach(base, plan))
    return candidates


def _move_ee_oriented(group, position: Vec3, plan: GraspPlan,
                      candidates: list[Quat | None], *, cartesian: bool = False,
                      label: str = "move") -> str | None:
    """Try each orientation candidate in turn; lock plan.quaternion to the first that moves.

    On total failure the first candidate is restored so the (failed) orientation
    stays consistent for the subsequent grasp descent.
    """
    result = None
    multi = len(candidates) > 1
    for i, q in enumerate(candidates):
        plan.quaternion = q
        tag = "graspnet" if i == 0 else f"flip#{i}"
        if multi:
            print(f"[manipulation] {label}: trying orientation [{tag}] {q}")
        result = _move_ee(group, position, plan, cartesian=cartesian)
        if _move_ok(result):
            if i > 0:
                print(f"[manipulation] {label}: orientation [{tag}] SUCCEEDED (flipped gripper)")
            return result
        if multi:
            print(f"[manipulation] {label}: orientation [{tag}] result={result!r}")
    plan.quaternion = candidates[0]  # keep the locked orientation consistent on failure
    return result


def _pregrasp(plan: GraspPlan) -> Vec3:
    """Pre-grasp/pre-place waypoint: GraspNet's approach pose, else a heuristic offset."""
    if plan.approach_position is not None:
        return plan.approach_position
    x, y, z = plan.position
    off = _envf("WALKIE_PREGRASP_OFFSET_M", "0.10")
    if plan.approach == "front":
        return x - off, y, z  # pull back toward the robot (+x is forward)
    return x, y, z + off  # top-down: come down from above


def _carry_arm(ctx: TaskContext, group) -> None:
    """Tuck the arm into a carry/standby pose so the base can nav safely."""
    pose_name = os.getenv("WALKIE_CARRY_POSE", "standby").strip()
    try:
        group.go_to_home(pose_name=pose_name, blocking=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation] carry/home ({pose_name}) failed ({exc})")


# --- collision-scene staging (rich path only) -------------------------------
def _stage_table(ctx: TaskContext, obj: DetectedObject) -> None:
    """Add the surface collision box from a DB surface node, when one is resolvable.

    Gated by ``WALKIE_STAGE_TABLE`` (default off): the table box frequently
    swallows the grasp pose, so MoveIt rejects it as in-collision and can't plan.
    With it off the arm plans against the octomap alone. Re-enable
    (``WALKIE_STAGE_TABLE=1``) once the box is calibrated to sit below the grasp.
    """
    if os.getenv("WALKIE_STAGE_TABLE", "0").strip().lower() not in ("1", "true", "yes"):
        print("[manipulation] table staging disabled (WALKIE_STAGE_TABLE=0); "
              "planning against octomap only")
        return
    graphs = getattr(ctx, "graphs", None)
    node = None
    surface_query = os.getenv("WALKIE_SURFACE_CLASS", "table").strip()
    if graphs is not None and surface_query:
        node = db.resolve_surface_node(graphs, surface_query, near=obj.world_xy)
    scene.add_surface_collision(ctx, node=node)


# --- grasp execution --------------------------------------------------------
def _execute_grasp(ctx: TaskContext, group, plan: GraspPlan, *, rich: bool) -> bool:
    """Open -> pre-grasp -> grasp -> close(+attach) -> lift -> carry. Returns grasped."""
    _gated("open gripper", group.gripper, 1.0, blocking=True)  # ready to receive
    # Decide the wrist orientation at the free-planning pre-grasp move (try
    # GraspNet's pose, then the flipped gripper / fixed override) and lock the
    # winner into plan.quaternion for the cartesian grasp descent below.
    candidates = _orient_candidates(ctx, plan)
    _gated("move to pre-grasp", _move_ee_oriented,
           group, _pregrasp(plan), plan, candidates, label="pre-grasp")
    if rich:
        scene.allow_gripper_vs_octomap(ctx, True)  # grasp inside sensed voxels
    _gated("move to grasp pose", _move_ee, group, plan.position, plan, cartesian=True)
    if rich:
        scene.attach_grasped_object(ctx)  # next close attaches the box to the hand
    grasped = True
    if _confirm("close gripper to grasp"):
        try:
            result = group.grasp()  # close on the object; judge by 'grasped'
            grasped = bool(result.get("grasped", True))
        except Exception as exc:  # noqa: BLE001
            print(f"[manipulation] grasp() failed ({exc}); closing gripper directly")
            group.gripper(0.0, blocking=True)
    if rich:
        scene.allow_gripper_vs_octomap(ctx, False)  # re-enforce once grasped
    lift = _envf("WALKIE_LIFT_HEIGHT_M", "0.15")
    _gated(f"lift object {lift:.2f} m", group.go_to_pose_relative,
           [0.0, 0.0, lift], [0.0, 0.0, 0.0], blocking=True)
    if _confirm("tuck arm to carry"):
        _carry_arm(ctx, group)
    return grasped


def _execute_place(ctx: TaskContext, group, plan: GraspPlan, *, rich: bool) -> bool:
    """Pre-place -> (hover) -> place -> open(+detach) -> carry -> close. Returns True."""
    _gated("move to pre-place", _move_ee, group, _pregrasp(plan), plan)
    if rich:
        clearance = _envf("WALKIE_PLACE_Z_CLEARANCE_M", "0.10")
        x, y, z = plan.position
        _gated(f"hover {clearance:.2f} m above surface", _move_ee,
               group, (x, y, z + clearance), plan)
    _gated("move to place pose", _move_ee, group, plan.position, plan, cartesian=True)
    if rich:
        scene.release_object_scene(ctx)  # next open detaches + removes the box
    _gated("open gripper to release", group.gripper, 1.0, blocking=True)
    if _confirm("tuck arm to carry"):
        _carry_arm(ctx, group)
    group.gripper(0.0, blocking=False)  # close so the gripper isn't left hanging open
    return True


# --- public manipulation primitives -----------------------------------------
def pick_object(ctx: TaskContext, obj: DetectedObject) -> bool:
    """Grasp *obj* and lift it for transport. Real arm motion.

    GraspNet planner: standoff approach -> camera aim -> plan -> standby -> table
    box -> refine approach -> collision-aware grasp. Stub planner: plan -> grasp
    (no scene/approach extras), unchanged from before. Degrades to False
    (announces, never raises) when the object can't be planned or any arm command
    fails, so the caller can score-degrade gracefully.
    """
    rich = _rich()
    group = _arm_group(ctx)
    try:
        if rich and obj.world_xy is not None:
            far = _envf("WALKIE_PICK_STANDOFF_FAR_M", "0.35")
            # Show the go_to target in RViz BEFORE the confirm gate, so the tester
            # sees where the base is headed and can accept/skip/abort it.
            viz_nav_target(ctx, obj.world_xy, far, label="far standoff",
                           ns="pick/far_standoff", marker_id=400)
            _gated(_nav_label("drive to far standoff", obj.world_xy, far),
                   drive_to_object, ctx, obj.world_xy, far)
            _gated("aim camera (lift + head tilt)", aim_camera_at_object, ctx, obj.world_xyz)
        plan = plan_grasp(ctx, obj)
        if plan is None:
            ctx.say(PICK_NO_PLAN.format(obj=obj.class_name))
            print(f"[manipulation] pick_object({obj.class_name}) — no grasp plan")
            return False
        ctx.say(PICKING.format(obj=obj.class_name))  # also the perception signal
        print(f"[manipulation] grasp plan for {obj.class_name}: pos={plan.position} "
              f"quat={plan.quaternion} rpy={plan.rotation} frame={plan.frame_id} "
              f"approach={plan.approach} score={plan.score}")
        if rich:
            if _confirm("tuck arm to standby"):
                _carry_arm(ctx, group)  # arm to standby before staging the scene
            _stage_table(ctx, obj)
            if obj.world_xy is not None:
                near = _envf("WALKIE_PICK_STANDOFF_NEAR_M", "0.10")
                viz_nav_target(ctx, obj.world_xy, near, label="near standoff",
                               ns="pick/near_standoff", marker_id=402)
                _gated(_nav_label("refine approach to near standoff", obj.world_xy, near),
                       refine_approach, ctx, obj.world_xy, near)
        return _execute_grasp(ctx, group, plan, rich=rich)
    except _AbortManipulation as ab:
        print(f"[manipulation] pick_object({obj.class_name}) aborted by tester at: {ab}")
        _carry_arm(ctx, group)
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation] pick_object({obj.class_name}) arm motion failed ({exc})")
        _carry_arm(ctx, group)
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
    group = _arm_group(ctx)
    try:
        return _execute_place(ctx, group, plan, rich=_rich())
    except _AbortManipulation as ab:
        print(f"[manipulation] place_at_pose aborted by tester at: {ab}")
        _carry_arm(ctx, group)
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation] place_at_pose arm motion failed ({exc})")
        _carry_arm(ctx, group)
        return False


def release_in_front(ctx: TaskContext) -> bool:
    """Fallback drop: open the gripper to release in front, then tuck the arm.

    Used when no place pose is configured. Returns False so the caller knows the
    placement was not a deliberate, located put-down.
    """
    group = _arm_group(ctx)
    if _rich():
        scene.release_object_scene(ctx)  # detach the carried box on the open
    try:
        if _confirm("open gripper to release in front"):
            group.gripper(1.0, blocking=True)  # release best-effort
    except _AbortManipulation as ab:
        print(f"[manipulation] release_in_front aborted by tester at: {ab}")
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation] release_in_front gripper failed ({exc})")
    _carry_arm(ctx, group)
    return False
