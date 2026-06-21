"""Best-of-N grasp selection — snap the camera a few times, keep the best grasp.

GraspNet only ever sees a camera-**optical** cloud (X-right, Y-down, Z-forward),
which is what keeps it in-distribution. This skill maps the winning grasp back to
the **map frame** using the snapshot's frozen capture-time pose, so callers get a
map-frame grasp point (and a backed-off pre-grasp point) ready to hand to the arm.

    cand = grasp_object(ctx, ["red can"], attempts=5, approach_preference="side")
    if cand:
        ctx.goto_pregrasp(cand.pregrasp_xyz)   # caller's arm/nav logic
        ...
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from interfaces.devices.camera import camera_pose
from tasks.base import TaskContext

Vec3 = tuple[float, float, float]

# Head tilt limits (walkie_sdk.modules.head): 0 = level, +down. head.tilt RAISES
# outside this band, so we clamp locally before commanding.
_HEAD_TILT_MIN = -math.pi / 4  # -45deg, look up
_HEAD_TILT_MAX = math.pi / 3  # +60deg, look down


@dataclass
class GraspCandidate:
    """The best grasp found, expressed in the **map frame**.

    ``grasp_xyz`` is the gripper-closing centre; ``pregrasp_xyz`` is that point
    backed off ``standoff_m`` along the *negative* approach direction — where the
    gripper should arrive before driving straight in. ``rotation`` is the 3x3 grasp
    frame in the map frame (column 0 = approach/travel, column 1 = closing/spread);
    ``approach`` is its unit approach axis. ``score`` is GraspNet's quality and
    ``width`` the gripper opening in metres.
    """

    grasp_xyz: tuple[float, float, float]
    pregrasp_xyz: tuple[float, float, float]
    rotation: np.ndarray  # (3, 3) in the map frame
    approach: np.ndarray  # (3,) unit approach direction in the map frame
    width: float
    score: float


def _to_map_frame(snap, g, standoff_m: float) -> GraspCandidate:
    """Lift one GraspNet pose (optical frame) into the snapshot's map frame."""
    R_cam, t_cam = snap.cam.R, snap.cam.t
    grasp = R_cam @ np.asarray(g.translation, dtype=float) + t_cam
    R_map = R_cam @ g.rotation
    approach = R_map[:, 2]  # unit travel direction toward the object
    pregrasp = grasp - approach * standoff_m
    return GraspCandidate(
        grasp_xyz=(float(grasp[0]), float(grasp[1]), float(grasp[2])),
        pregrasp_xyz=(float(pregrasp[0]), float(pregrasp[1]), float(pregrasp[2])),
        rotation=R_map,
        approach=approach,
        width=float(g.width),
        score=float(g.score),
    )


def get_object_grasp_pos(
    ctx: TaskContext,
    prompts: list[str],
    *,
    attempts: int = 5,
    standoff_m: float = 0.10,
    voxel: float = 0.005,
    erode_px: int = 5,
    min_points: int = 50,
    min_confidence: float = 0.3,
    antipodal: bool = True,
    approach_preference: str = "none",
    approach_weight: float | None = None,
) -> GraspCandidate | None:
    """Best-of-N grasp for the nearest object matching *prompts*, in the map frame.

    Captures up to *attempts* snapshots; on each it runs masked open-vocab
    detection for *prompts*, drops detections below *min_confidence*, lifts the
    **nearest** surviving detection's mask to a camera-optical cloud, and asks
    GraspNet for the single best grasp. The highest-scoring grasp
    across all attempts wins, mapped to the map frame against the geometry of the
    very snapshot it came from (accurate even after detection/GraspNet latency).

    Args:
        ctx: Task context (camera, AI client).
        prompts: Open-vocab detector prompts for the target (e.g. ``["red can"]``).
        attempts: How many snapshots to take and score (best-of-N).
        standoff_m: Pre-grasp back-off distance along the approach axis (metres).
        voxel: Voxel-downsample size for the lifted object cloud (metres).
        erode_px: Mask erosion before lifting, to shed rim/background pixels.
        min_points: Skip a detection whose lifted cloud is smaller than this.
        min_confidence: Drop detections whose detector confidence is below this
            (detections with no confidence reported are kept). Among the survivors,
            the one closest to the camera is grasped.
        antipodal: Run GraspNet's antipodal surface-normal validation.
        approach_preference: Bias grasp selection by approach direction relative to
            gravity: ``"side"`` favours horizontal approaches (e.g. grabbing a can
            around its side under a high fixed camera), ``"top"`` favours approaches
            pointing straight down (e.g. a spoon lying flat), ``"none"`` leaves
            GraspNet's ranking untouched. The "up" reference is derived
            automatically from each snapshot's pose (the map frame's +Z gravity axis,
            expressed in the cloud's optical frame), so the caller need not supply it.
        approach_weight: How strongly the preference outranks GraspNet's own score
            (server default ~1.0; higher favours the preferred approach harder). Only
            used when ``approach_preference`` is set; ``None`` keeps the server default.

    Returns:
        The winning :class:`GraspCandidate` (with ``grasp_xyz`` and
        ``pregrasp_xyz`` in the map frame), or ``None`` if no attempt produced a
        graspable detection.
    """
    best: GraspCandidate | None = None

    for i in range(attempts):
        tag = f"attempt {i + 1}/{attempts}"
        snap = ctx.snapshot()
        if snap is None or not snap.has_geometry:
            print(f"[grasp] {tag}: no snapshot geometry (is the ZED running?)")
            continue

        detections = ctx.walkieAI.image.detect(snap.img, prompts=prompts, return_mask=True)
        detections = [
            d for d in detections
            if d.mask is not None
            and (d.confidence is None or d.confidence >= min_confidence)
        ]
        if not detections:
            print(f"[grasp] {tag}: no masked detections for {prompts} "
                  f"(confidence >= {min_confidence})")
            continue

        # Lift every surviving detection and keep the one closest to the camera.
        # The cloud is in the optical frame (camera at the origin, looking down
        # +Z), so the median point range is the object's distance; the nearest is
        # easiest to reach and least likely to be a far-away false positive.
        cloud: np.ndarray | None = None
        nearest_range = float("inf")
        for det in detections:
            pts = snap.mask_to_points(det.mask, voxel=voxel, frame="optical", erode_px=erode_px)
            if pts.shape[0] < min_points:
                continue
            rng = float(np.median(np.linalg.norm(pts, axis=1)))
            if rng < nearest_range:
                nearest_range, cloud = rng, pts
        if cloud is None:
            print(f"[grasp] {tag}: no detection lifted >= {min_points} pts — too far/occluded?")
            continue

        infer_kwargs: dict = {"antipodal": antipodal, "max_grasps": 1}
        if approach_preference != "none":
            # World-up = the map frame's +Z (gravity) axis, expressed in the
            # camera-optical frame the cloud lives in, so the server can bias
            # side/top approaches against gravity.
            infer_kwargs["approach_preference"] = approach_preference
            infer_kwargs["up"] = snap.cam.R.T @ np.array([0.0, 0.0, 1.0])
            if approach_weight is not None:
                infer_kwargs["approach_weight"] = approach_weight
        grasps = ctx.walkieAI.grasp.infer(cloud, **infer_kwargs)
        if not grasps:
            print(f"[grasp] {tag}: GraspNet returned nothing")
            continue

        g = grasps[0]
        if best is not None and g.score <= best.score:
            print(f"[grasp] {tag}: score {g.score:.3f} (keeping best {best.score:.3f})")
            continue

        best = _to_map_frame(snap, g, standoff_m)
        gx, gy, gz = best.grasp_xyz
        print(f"[grasp] {tag}: new best score {best.score:.3f} "
              f"grasp=({gx:+.3f},{gy:+.3f},{gz:+.3f}) width={best.width * 100:.1f}cm")

    if best is None:
        print(f"[grasp] no graspable detection for {prompts} in {attempts} attempt(s)")
    return best


# ---------------------------------------------------------------------------
# Grasp execution: aim, approach, de-deadzone, and command the arm.
#
# The planner above only finds a map-frame grasp pose. The helpers below drive
# the robot to actually take it, handling Walkie's real constraints: the arms
# can't reach across the body centreline (a lateral dead-zone), the lift gives
# extra reach when commanded via the "*_arm_lift" groups, and objects too far /
# too low / off-camera need the base + head to reposition first.
# ---------------------------------------------------------------------------
def _arm_sides(arm: str) -> tuple[str, str, str]:
    """(motion_group, home_group, gripper_group) for an arm side.

    ``go_to_pose`` uses the lift group ("left_arm_lift") so MoveIt can solve the
    lift joint for extra reach; ``go_to_home`` uses the plain arm group, where the
    SRDF named poses (e.g. "hands_up") live. Unknown side -> warn + default left.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        print(f"[grasp] unknown arm {arm!r}; defaulting to left")
        side = "left"
    return f"{side}_arm_lift", f"{side}_arm", f"{side}_gripper"


def _world_to_base(ctx: TaskContext, xyz_map: Vec3) -> Vec3:
    """Map-frame point -> base_footprint (forward, left, z). Mirrors
    tasks.manipulation.world_to_base; inlined to keep grasp.py dependency-light."""
    ox, oy, oz = xyz_map
    p = ctx.current_pose()
    dx, dy = ox - p["x"], oy - p["y"]
    c, s = math.cos(p["heading"]), math.sin(p["heading"])
    return c * dx + s * dy, -s * dx + c * dy, oz


def _xy_dist(ctx: TaskContext, xyz_map: Vec3) -> float:
    """Planar distance from the robot base to a map-frame point (metres)."""
    p = ctx.current_pose()
    return math.hypot(xyz_map[0] - p["x"], xyz_map[1] - p["y"])


def _look_down_tilt(cam_t, xyz_map: Vec3) -> float:
    """Head tilt (rad, +down) that points the camera at *xyz_map*, clamped.

    ``cam_t`` is the camera optical-centre in the map frame. The pitch below the
    horizon is atan2(height_drop, horizontal_distance); positive when the object
    sits below the camera (the usual table-top case).
    """
    dx, dy = xyz_map[0] - cam_t[0], xyz_map[1] - cam_t[1]
    horiz = math.hypot(dx, dy)
    tilt = math.atan2(cam_t[2] - xyz_map[2], horiz)
    return max(_HEAD_TILT_MIN, min(_HEAD_TILT_MAX, tilt))


def _draw_grasp_viz(ctx: TaskContext, candidate: GraspCandidate) -> None:
    """Best-effort: drop the planned grasp markers into the shared viewer."""
    if getattr(ctx, "viz", None) is None:
        return
    try:
        ctx.viz.clear("grasp", recursive=True)
        ctx.viz.axes("grasp/ee", candidate.grasp_xyz, rotation=candidate.rotation,
                     length=0.10, labels=True)
        ctx.viz.points("grasp/pregrasp", [list(candidate.pregrasp_xyz)], radii=[0.02],
                       colors=[(255, 180, 0)], labels=["pregrasp"])
        approach = (np.asarray(candidate.grasp_xyz) - np.asarray(candidate.pregrasp_xyz)).tolist()
        ctx.viz.arrow("grasp/approach", candidate.pregrasp_xyz, approach, color=(255, 180, 0))
    except Exception as exc:  # noqa: BLE001 — viz is never load-bearing
        print(f"[grasp] viz failed ({exc})")


def look_at_object(ctx: TaskContext, xyz_map: Vec3) -> bool:
    """Tilt the head so the camera points at the map-frame object. Best-effort.

    Reads the live camera pose (cheap TF lookup, no RGB-D grab), computes the
    clamped look-down tilt, and commands the head servo. Returns False (never
    raises) when the camera pose is unavailable or the servo command fails.
    """
    cam = camera_pose(ctx.walkie)
    if cam is None:
        print("[grasp] look_at_object: no camera pose")
        return False
    tilt = _look_down_tilt(cam.t, xyz_map)
    try:
        ctx.walkie.robot.head.tilt(tilt)
    except Exception as exc:  # noqa: BLE001 — off-robot stub may lack robot.head
        print(f"[grasp] look_at_object: head tilt failed ({exc})")
        return False
    return True


def approach_object(
    ctx: TaskContext,
    xyz_map: Vec3,
    *,
    standoff_m: float = 0.60,
    trigger_m: float = 0.70,
    track: bool = True,
    tick_sec: float = 0.2,
    timeout_sec: float = 30.0,
) -> str:
    """Drive to a standoff facing the object, tilting the head to keep it in view.

    Uses ``nav.go_to`` with the heading omitted (NavigateToObject) and
    ``align_method="face_target"`` so the base ends up facing the object at
    *standoff_m* metres — short of a table edge. With *track* the drive is issued
    non-blocking and the head is re-aimed every *tick_sec* as the robot closes in
    (the object is static, so tracking just re-tilts as the distance shrinks).

    Returns ``"CLOSE"`` (already within *trigger_m*, no move — head still aimed),
    ``"MOVED"`` (drove and the nav goal succeeded / was close enough), or
    ``"FAILED"`` (nav refused, timed out, or errored).
    """
    if _xy_dist(ctx, xyz_map) <= trigger_m:
        look_at_object(ctx, xyz_map)
        return "CLOSE"

    ox, oy = float(xyz_map[0]), float(xyz_map[1])
    if not track:
        try:
            res = ctx.walkie.nav.go_to(
                x=ox, y=oy, blocking=True, standoff=standoff_m, align_method="face_target",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[grasp] approach_object: nav raised ({exc})")
            return "FAILED"
        look_at_object(ctx, xyz_map)
        print(f"[grasp] approach_object: nav -> {res}")
        return "MOVED" if res in ("SUCCEEDED", "CLOSE_ENOUGH") else "FAILED"

    try:
        ctx.walkie.nav.go_to(
            x=ox, y=oy, blocking=False, standoff=standoff_m, align_method="face_target",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] approach_object: nav raised ({exc})")
        return "FAILED"
    deadline = time.monotonic() + timeout_sec
    while ctx.walkie.nav.is_navigating and time.monotonic() < deadline:
        look_at_object(ctx, xyz_map)
        time.sleep(tick_sec)
    if ctx.walkie.nav.is_navigating:  # timed out while still driving
        print(f"[grasp] approach_object: timed out after {timeout_sec:.0f}s; cancelling")
        ctx.walkie.nav.cancel()
        return "FAILED"
    status = ctx.walkie.nav.status
    look_at_object(ctx, xyz_map)
    print(f"[grasp] approach_object: nav -> {status}")
    return "MOVED" if status in ("SUCCEEDED", "CLOSE_ENOUGH") else "FAILED"


def in_arm_deadzone(ctx: TaskContext, xyz_map: Vec3, *, half_width_m: float = 0.20) -> bool:
    """Whether the object sits in the central lateral dead-zone the arms can't reach.

    The arms can't rotate toward the robot centreline, so an object whose
    base_footprint lateral offset |y| is within *half_width_m* is unreachable
    even when dead ahead — the base must rotate first (see face_object_with_arm).
    """
    _, left, _ = _world_to_base(ctx, xyz_map)
    return -half_width_m <= left <= half_width_m


def face_object_with_arm(ctx: TaskContext, xyz_map: Vec3, *, arm: str = "left") -> bool:
    """Rotate the base so *arm* faces the object, lifting it out of the dead-zone.

    Looks up the arm's shoulder link (``openarm_{side}_link3``) in the map frame
    and rotates the base until the shoulder->object bearing is the robot heading
    (the arm's forward direction is taken to be the robot's heading). One-shot
    approximation: rotating the base also swings the shoulder on a small arc, so
    the post-turn aim is a few degrees off — but enough to move the object off the
    centreline and in front of that arm. Best-effort; False on no transform/odom.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        side = "left"
    frame = f"openarm_{side}_link3"
    try:
        tf = ctx.walkie.robot.transform.lookup("map", frame, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] face_object_with_arm: transform.lookup({frame}) raised ({exc})")
        return False
    if not tf or "position" not in tf:
        print(f"[grasp] face_object_with_arm: no transform for {frame}; cannot face")
        return False
    link = tf["position"]
    desired = math.atan2(xyz_map[1] - link["y"], xyz_map[0] - link["x"])
    print(f"[grasp] face_object_with_arm[{side}]: link=({link['x']:+.2f},{link['y']:+.2f}) "
          f"obj=({xyz_map[0]:+.2f},{xyz_map[1]:+.2f}) -> heading {math.degrees(desired):+.0f}deg")
    return ctx.rotate_to(desired)


def execute_grasp(
    ctx: TaskContext,
    candidate: GraspCandidate,
    *,
    arm: str = "left",
    home_pose: str = "hands_up",
    tuck_on_abort: bool = True,
    viz: bool = True,
) -> bool:
    """Command *arm* to take the map-frame *candidate*: open -> pre-grasp -> grasp -> close.

    Drives the arm to the pre-grasp then grasp pose (both map-frame, via the
    ``*_arm_lift`` MoveIt group so the lift solves for reach), **checking each
    move's result string** and aborting on anything but ``"SUCCEEDED"``. Always
    re-enables gripper collision; tucks the arm home on abort. Returns True only
    when both moves succeeded and the gripper closed. Never raises.
    """
    motion_group, home_group, gripper_group = _arm_sides(arm)
    side = motion_group.split("_")[0]
    hand = getattr(ctx.walkie.arm, side)

    try:
        roll, pitch, yaw = Rotation.from_matrix(candidate.rotation).as_euler("xyz")
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: bad rotation matrix ({exc})")
        return False
    print(f"[grasp] execute_grasp[{side}]: RPY=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) "
          f"grasp={candidate.grasp_xyz} pregrasp={candidate.pregrasp_xyz}")
    if viz:
        _draw_grasp_viz(ctx, candidate)

    succeeded = False
    collision_disabled = False
    try:
        hand.gripper(1.0, blocking=True)  # open, ready to receive
        home_res = ctx.walkie.arm.go_to_home(group_name=home_group, pose_name=home_pose, blocking=True)
        if home_res != "SUCCEEDED":  # staging move: warn but press on
            print(f"[grasp] execute_grasp[{side}]: stage home -> {home_res} (continuing)")

        ctx.walkie.arm.toggle_gripper_collision(gripper_group, False)
        collision_disabled = True

        res = ctx.walkie.arm.go_to_pose(
            *candidate.pregrasp_xyz, roll, pitch, yaw,
            group_name=motion_group, frame_id="map", blocking=True,
        )
        print(f"[grasp] execute_grasp[{side}]: pregrasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: pregrasp move failed; aborting")
            return False

        res = ctx.walkie.arm.go_to_pose(
            *candidate.grasp_xyz, roll, pitch, yaw,
            group_name=motion_group, frame_id="map", blocking=True,
        )
        print(f"[grasp] execute_grasp[{side}]: grasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: grasp move failed; aborting")
            return False

        hand.gripper(0.0, blocking=True)  # close on the object
        ctx.walkie.arm.go_to_home(group_name=home_group, pose_name=home_pose, blocking=True)
        succeeded = True
        print(f"[grasp] execute_grasp[{side}]: success")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: hardware error ({exc})")
        return False
    finally:
        if collision_disabled:
            try:
                ctx.walkie.arm.toggle_gripper_collision(gripper_group, True)
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: re-enable gripper collision failed ({exc})")
        if tuck_on_abort and not succeeded:
            try:
                ctx.walkie.arm.go_to_home(group_name=home_group, pose_name=home_pose, blocking=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: tuck-on-abort home failed ({exc})")


def pick_object(
    ctx: TaskContext,
    prompts: list[str],
    *,
    arm: str = "auto",
    attempts: int = 5,
    pregrasp_standoff_m: float = 0.10,
    approach_preference: str = "none",
    approach_weight: float | None = None,
    optimal_standoff_m: float = 0.60,
    approach_trigger_m: float = 0.70,
    max_reach_xy_m: float = 0.75,
    min_grasp_z_m: float = 0.70,
    deadzone_half_m: float = 0.20,
    default_arm: str = "left",
    track: bool = True,
    viz: bool = True,
) -> bool:
    """Full pick for the nearest object matching *prompts*: detect -> approach -> de-deadzone -> grasp.

    Sequences :func:`get_object_grasp_pos` (best-of-N planning) with the base/head
    repositioning the robot needs to actually reach the grasp:

    1. Plan a grasp; bail if the object is below *min_grasp_z_m* (no remedy).
    2. If it's farther than *approach_trigger_m* (XY), drive to *optimal_standoff_m*
       facing it (head tracking), then re-plan from the new viewpoint.
    3. Pick the arm (``"auto"`` -> object's side, dead-centre -> *default_arm*).
    4. If it's in the lateral dead-zone, rotate the base so the arm faces it
       (:func:`face_object_with_arm`), re-aim the head, re-plan, re-check once.
    5. Execute the grasp on the chosen arm, checking each arm move.

    Returns True only when the grasp executed cleanly. Degrades to False (never
    raises) at any failing step.

    Note: distinct from :func:`tasks.manipulation.pick_object` (which takes a
    pre-detected ``DetectedObject``) — import this one from ``tasks.skills``.
    """
    def _grasp() -> GraspCandidate | None:
        return get_object_grasp_pos(
            ctx, prompts, attempts=attempts, standoff_m=pregrasp_standoff_m,
            approach_preference=approach_preference, approach_weight=approach_weight,
        )

    cand = _grasp()
    if cand is None:
        print(f"[grasp] pick_object: no grasp for {prompts}")
        return False
    if cand.grasp_xyz[2] < min_grasp_z_m:
        print(f"[grasp] pick_object: object too low (z={cand.grasp_xyz[2]:.2f}m < "
              f"{min_grasp_z_m:.2f}m); cannot reach")
        return False

    # 2. Approach to the optimal standoff if too far, tracking with the head.
    status = approach_object(
        ctx, cand.grasp_xyz, standoff_m=optimal_standoff_m,
        trigger_m=approach_trigger_m, track=track,
    )
    if status == "FAILED":
        print("[grasp] pick_object: approach failed; aborting")
        return False
    if status == "MOVED":
        cand = _grasp()  # the old candidate is stale once the base moved
        if cand is None:
            print("[grasp] pick_object: lost the object after approaching")
            return False
        if cand.grasp_xyz[2] < min_grasp_z_m:
            print(f"[grasp] pick_object: object too low after approach "
                  f"(z={cand.grasp_xyz[2]:.2f}m); cannot reach")
            return False

    # 3. Pick the arm by which side the object is on (dead-centre -> default).
    in_zone = in_arm_deadzone(ctx, cand.grasp_xyz, half_width_m=deadzone_half_m)
    if arm == "auto":
        left = _world_to_base(ctx, cand.grasp_xyz)[1]
        if in_zone or left == 0:
            chosen = default_arm
        else:
            chosen = "left" if left > 0 else "right"
    else:
        chosen = (arm or default_arm).strip().lower()
        if chosen not in ("left", "right"):
            print(f"[grasp] pick_object: bad arm {arm!r}; using {default_arm}")
            chosen = default_arm
    print(f"[grasp] pick_object: chosen arm = {chosen}")

    # 4. If central, rotate so the arm faces the object, re-aim, re-plan, re-check.
    if in_zone:
        print("[grasp] pick_object: object in dead-zone; rotating to face it")
        if face_object_with_arm(ctx, cand.grasp_xyz, arm=chosen):
            look_at_object(ctx, cand.grasp_xyz)
            cand = _grasp()
            if cand is None:
                print("[grasp] pick_object: lost the object after facing")
                return False
            if in_arm_deadzone(ctx, cand.grasp_xyz, half_width_m=deadzone_half_m):
                left = _world_to_base(ctx, cand.grasp_xyz)[1]
                print(f"[grasp] pick_object: still in dead-zone after facing "
                      f"(y={left:+.2f}m); attempting grasp anyway")
        else:
            print("[grasp] pick_object: could not face object; continuing best-effort")

    # 5. Final radial reach guard, then execute.
    reach = _xy_dist(ctx, cand.grasp_xyz)
    if reach > max_reach_xy_m:
        print(f"[grasp] pick_object: out of reach (xy={reach:.2f}m > {max_reach_xy_m:.2f}m); aborting")
        return False
    return execute_grasp(ctx, cand, arm=chosen, viz=viz)
