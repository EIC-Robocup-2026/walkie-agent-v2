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
import os
import time
from dataclasses import dataclass, replace

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
    # Filled in after best-of-N selection (see get_object_grasp_pos): the surface the
    # grasped object was sitting on, and the grasp height above it — remembered so the
    # object can be placed back at the same relative height on another surface. ``None``
    # when no support surface was found (or detection was skipped).
    support_surface_z: float | None = None
    grasp_to_surface_offset: float | None = None
    object_footprint_m: float | None = None  # map-frame XY span of the grasped cloud


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
    voxel: float = 0.002,
    erode_px: int = 5,
    min_points: int = 50,
    min_confidence: float = 0.3,
    antipodal: bool = True,
    approach_preference: str = "none",
    approach_weight: float | None = None,
    compute_support: bool = True,
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
    best_snap = None  # the snapshot the winning grasp came from (for surface lookup)
    best_cloud: np.ndarray | None = None  # the winning object cloud (optical frame)

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
        best_snap, best_cloud = snap, cloud
        gx, gy, gz = best.grasp_xyz
        print(f"[grasp] {tag}: new best score {best.score:.3f} "
              f"grasp=({gx:+.3f},{gy:+.3f},{gz:+.3f}) width={best.width * 100:.1f}cm")

    if best is None:
        print(f"[grasp] no graspable detection for {prompts} in {attempts} attempt(s)")
        return None

    # Remember the support surface + object footprint from the winning snapshot, so the
    # object can be placed back later (tasks.skills.place). Computed against the RAW
    # grasp z, BEFORE the side-grasp nudge below, so the stored offset is the true
    # grasp-to-surface height (not inflated by the nudge).
    if compute_support and best_snap is not None:
        try:
            from interfaces.perception.surfaces import (
                detect_horizontal_surfaces,
                support_surface_for,
            )
            from tasks.skills.place import _full_scene_cloud, _surface_kwargs

            scene = _full_scene_cloud(best_snap)
            surfaces = detect_horizontal_surfaces(scene, **_surface_kwargs())
            gx, gy, gz = best.grasp_xyz
            sup = support_surface_for(surfaces, gx, gy, gz)
            if sup is not None:
                best.support_surface_z = sup.z
                best.grasp_to_surface_offset = gz - sup.z
                print(f"[grasp] support surface z={sup.z:.2f}m "
                      f"(grasp {gz:.2f}m, offset {best.grasp_to_surface_offset:+.2f}m)")
            else:
                print("[grasp] no support surface found under the grasp")
        except Exception as exc:  # noqa: BLE001 — surface lookup is best-effort
            print(f"[grasp] support-surface lookup failed ({exc})")

    if best_cloud is not None and best_snap is not None and best_snap.cam is not None:
        try:
            cm = best_cloud @ best_snap.cam.R.T + best_snap.cam.t
            span = cm[:, :2].max(axis=0) - cm[:, :2].min(axis=0)
            best.object_footprint_m = float(max(span))
        except Exception as exc:  # noqa: BLE001
            print(f"[grasp] object footprint estimate failed ({exc})")

    # offset height a little for side grasps
    best.grasp_xyz = (best.grasp_xyz[0], best.grasp_xyz[1], best.grasp_xyz[2] + 0.03)
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
    ctx.walkie.robot.head.get_angle
    cam = camera_pose(ctx.walkie)
    if cam is None:
        print("[grasp] look_at_object: no camera pose")
        return False
    tilt = _look_down_tilt(cam.t, xyz_map)
    try:
        # Limit tilt because graspnet is bad
        tilt = max(0.436332, tilt)
        ctx.walkie.robot.head.tilt(tilt)
    except Exception as exc:  # noqa: BLE001 — off-robot stub may lack robot.head
        print(f"[grasp] look_at_object: head tilt failed ({exc})")
        return False
    return True


def face_object(ctx: TaskContext, xyz_map: Vec3) -> bool:
    """Rotate the base so the robot heading points straight at the object.

    Centres the object in the camera's horizontal FOV so detection/GraspNet see
    it square-on — better masks, better grasps. One-shot rotate-in-place toward
    the map-frame point; best-effort, returns ``rotate_to``'s result.
    """
    p = ctx.current_pose()
    desired = math.atan2(xyz_map[1] - p["y"], xyz_map[0] - p["x"])
    return ctx.rotate_to(desired)


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
        return "MOVED" # Tests
        # return "MOVED" if res in ("SUCCEEDED", "CLOSE_ENOUGH") else "FAILED"

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


def align_arm_to_object(ctx: TaskContext, xyz_map: Vec3, *, arm: str = "left") -> bool:
    """Strafe the base sideways so *arm* lines up laterally with the object.

    Walkie's base is omnidirectional, so instead of rotating to face the object
    with the arm (which swings the shoulder on an arc and re-aims the base), we
    *translate* the base sideways until the object's lateral offset in the base
    frame matches the arm's own lateral mounting offset — putting the object
    directly in front of that arm and out of the centreline dead-zone. The goal
    is an absolute map-frame pose at the **current heading** (a pure strafe), so
    the map-frame grasp candidate stays valid (no re-plan needed).

    nav failure is deliberately ignored: a refused or clipped strafe usually just
    means the base is already as far over as the footprint/obstacles allow, which
    is good enough to grasp from. Best-effort; never raises.
    """
    side = (arm or "left").strip().lower()
    if side not in ("left", "right"):
        side = "left"
    frame = f"openarm_{side}_link3"
    arm_left = 0.0
    try:
        tf = ctx.walkie.robot.transform.lookup("base_footprint", frame, timeout=5.0)
        if tf and "position" in tf:
            arm_left = float(tf["position"]["y"])
    except Exception as exc:  # noqa: BLE001 — fall back to the centreline
        print(f"[grasp] align_arm_to_object: transform.lookup({frame}) raised ({exc}); "
              f"assuming arm on centreline")
    _, obj_left, _ = _world_to_base(ctx, xyz_map)
    strafe = obj_left - arm_left  # base must move this far +left to line the arm up
    p = ctx.current_pose()
    lx, ly = -math.sin(p["heading"]), math.cos(p["heading"])  # base +left axis in map frame
    tx, ty = p["x"] + strafe * lx, p["y"] + strafe * ly
    print(f"[grasp] align_arm_to_object[{side}]: obj_left={obj_left:+.2f}m arm_left={arm_left:+.2f}m "
          f"-> strafe {strafe:+.2f}m to ({tx:+.2f},{ty:+.2f})")
    try:
        res = ctx.walkie.nav.go_to(tx, ty, p["heading"], blocking=True)
        print(f"[grasp] align_arm_to_object[{side}]: nav -> {res} (failure ignored)")
    except Exception as exc:  # noqa: BLE001 — nav fail likely means already at the limit
        print(f"[grasp] align_arm_to_object[{side}]: nav raised ({exc}); ignored")
    return True


def creep_to_grasp_distance(
    ctx: TaskContext, xyz_map: Vec3, *, target_m: float = 0.50, max_advance_m: float = 0.35,
) -> bool:
    """Drive the base straight forward (along its heading) to close on the object.

    Nav's ``NavigateToObject`` standoff is unreliable on this robot — Nav2 often
    halts at the table/inflation boundary well short of the requested standoff, so
    the planned grasp ends up too far to reach accurately. This is a final, direct
    creep: a pure forward translation (heading held) that brings the object within
    *target_m* metres (planar). The base has already been faced/strafed at the
    object, so "forward" is "toward it". Pure translation keeps the map-frame grasp
    candidate valid (no re-plan). Capped at *max_advance_m* so a bad estimate can't
    drive the robot into the table. Best-effort; never raises.
    """
    dist = _xy_dist(ctx, xyz_map)
    if dist <= target_m:
        return True
    advance = min(dist - target_m, max_advance_m)
    p = ctx.current_pose()
    fx, fy = math.cos(p["heading"]), math.sin(p["heading"])  # base +forward in map
    tx, ty = p["x"] + advance * fx, p["y"] + advance * fy
    print(f"[grasp] creep_to_grasp_distance: {dist:.2f}m -> {target_m:.2f}m "
          f"(advance {advance:+.2f}m to {tx:+.2f},{ty:+.2f})")
    try:
        res = ctx.walkie.nav.go_to(tx, ty, p["heading"], blocking=True)
        print(f"[grasp] creep_to_grasp_distance: nav -> {res}")
    except Exception as exc:  # noqa: BLE001 — creep is best-effort
        print(f"[grasp] creep_to_grasp_distance: nav raised ({exc}); ignored")
    return True


def aim_forward_candidate(
    ctx: TaskContext, candidate: GraspCandidate, *, standoff_m: float = 0.10,
) -> GraspCandidate:
    """Re-point a grasp's wrist straight along the robot heading (map frame).

    GraspNet's returned wrist orientation is often awkward / IK-unsolvable on
    OpenArm. Instead of "positioning the arm at the object's full grasp pose", this
    points the gripper's approach axis (EE **+z**) along the robot's forward heading
    — taking the arm's forward to be the robot's. ``pick_object`` has already faced
    the object, so "forward" is "at the object". The base-frame wrist orientation is
    read from ``WALKIE_GRASP_POINT_RPY`` (default ``"0,1.5708,0"`` -> EE +z = base
    +x, horizontal forward) and rotated into the map frame by the current heading.

    The grasp *point* is kept; only the orientation, approach axis, and pre-grasp
    (re-backed-off *standoff_m* along the new -forward axis) change. Returns a new
    :class:`GraspCandidate` so the same pose drives both the arm and the held-object
    record (so the placer sets the object back down the way it was grasped).
    """
    raw = os.getenv("WALKIE_GRASP_POINT_RPY", "0,1.5708,0")
    rpy_base = [float(p.strip()) for p in raw.split(",")]
    theta = ctx.current_pose()["heading"]
    R_base = Rotation.from_euler("xyz", rpy_base).as_matrix()
    R_map = Rotation.from_euler("z", theta).as_matrix() @ R_base
    approach = R_map[:, 2]  # gripper points this way (map frame)
    grasp = np.asarray(candidate.grasp_xyz, dtype=float)
    pregrasp = grasp - approach * standoff_m
    print(f"[grasp] aim_forward_candidate: heading={math.degrees(theta):+.0f}deg "
          f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})")
    return replace(
        candidate,
        rotation=R_map,
        approach=approach,
        pregrasp_xyz=(float(pregrasp[0]), float(pregrasp[1]), float(pregrasp[2])),
    )


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

    Executes whatever orientation the *candidate* carries — callers wanting a
    forward-pointing wrist (instead of GraspNet's) re-point it first with
    :func:`aim_forward_candidate`.
    """
    motion_group, home_group, gripper_group = _arm_sides(arm)
    side = motion_group.split("_")[0]
    hand = getattr(ctx.walkie.arm, side)

    try:
        roll, pitch, yaw = Rotation.from_matrix(candidate.rotation).as_euler("xyz")
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: bad rotation matrix ({exc})")
        return False
    grasp_xyz, pregrasp_xyz = candidate.grasp_xyz, candidate.pregrasp_xyz
    print(f"[grasp] execute_grasp[{side}]: RPY=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) "
          f"grasp={grasp_xyz} pregrasp={pregrasp_xyz}")
    if viz:
        _draw_grasp_viz(ctx, candidate)

    succeeded = False
    collision_disabled = False
    try:
        # original_planner = ctx.walkie.robot.arm.get_param(name="planner_id")  # warm up the planner cache
        # ctx.walkie.robot.arm.set_param_result(name="planner_id", value="RRTstar")
        hand.gripper(1.0, blocking=True)  # open, ready to receive

        ctx.walkie.arm.toggle_gripper_collision(gripper_group, False)
        collision_disabled = True

        res = ctx.walkie.arm.go_to_pose(
            *pregrasp_xyz, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True,
        )
        print(f"[grasp] execute_grasp[{side}]: pregrasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: pregrasp move failed; aborting")
            return False

        res = ctx.walkie.arm.go_to_pose(
            *grasp_xyz, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True, cartesian_path=True
        )
        print(f"[grasp] execute_grasp[{side}]: grasp -> {res}")
        if res != "SUCCEEDED":
            print(f"[grasp] execute_grasp[{side}]: grasp move failed; aborting")
            return False

        hand.gripper(0.0, blocking=True)  # close on the object
        ctx.walkie.arm.go_to_home(group_name=motion_group, pose_name=home_pose, blocking=True)
        succeeded = True
        print(f"[grasp] execute_grasp[{side}]: success")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[grasp] execute_grasp[{side}]: hardware error ({exc})")
        return False
    finally:
        if tuck_on_abort and not succeeded:
            try:
                result = ctx.walkie.arm.go_to_home(group_name=home_group, pose_name=home_pose, blocking=True)
                print(f"[grasp] execute_grasp[{side}]: tuck-on-abort home -> {result}")
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: tuck-on-abort home failed ({exc})")
        # ctx.walkie.robot.arm.set_param_result(name="planner_id", value=original_planner)
        if collision_disabled:
            try:
                ctx.walkie.arm.toggle_gripper_collision(gripper_group, True)
            except Exception as exc:  # noqa: BLE001
                print(f"[grasp] execute_grasp[{side}]: re-enable gripper collision failed ({exc})")


def pick_object(
    ctx: TaskContext,
    prompts: list[str],
    *,
    arm: str = "auto",
    attempts: int = 10,
    pregrasp_standoff_m: float = 0.10,
    approach_preference: str = "none",
    approach_weight: float | None = None,
    optimal_standoff_m: float = 0.55,
    approach_trigger_m: float = 0.70,
    grasp_distance_m: float = 0.50,
    max_reach_xy_m: float = 0.75,
    min_grasp_z_m: float = 0.70,
    deadzone_half_m: float = 0.20,
    default_arm: str = "left",
    point_at_object: bool = True,
    track: bool = True,
    viz: bool = True,
) -> bool:
    """Full pick for the nearest object matching *prompts*: detect -> approach -> de-deadzone -> grasp.

    Sequences :func:`get_object_grasp_pos` (best-of-N planning) with the base/head
    repositioning the robot needs to actually reach the grasp. Every plan first
    faces the object (:func:`face_object` + :func:`look_at_object`) to centre it
    in view for a more accurate detection/grasp:

    1. Plan a grasp; bail if the object is below *min_grasp_z_m* (no remedy).
    2. If it's farther than *approach_trigger_m* (XY), drive to *optimal_standoff_m*
       facing it (head tracking), then re-plan from the new viewpoint.
    3. Pick the arm (``"auto"`` -> object's side, dead-centre -> *default_arm*).
    4. Rotate the base so the chosen arm points straight at the object
       (:func:`face_object_with_arm`, arm-forward = robot heading), recording the
       pre-rotate heading. This makes "forward" point at the object for the creep and
       wrist re-aim below; the map-frame candidate stays valid across the rotation.
    5. Creep the base straight forward (:func:`creep_to_grasp_distance`) so the
       object ends within *grasp_distance_m* — the approach often halts short of the
       table, leaving it too far to reach accurately. ``None`` disables the creep.
    6. With *point_at_object* (default), re-point the gripper straight forward along
       the robot heading (:func:`aim_forward_candidate`) instead of using GraspNet's
       wrist orientation (often IK-unsolvable on OpenArm) — the arm's forward is
       taken to be the robot's.
    7. Execute the grasp on the chosen arm, checking each arm move.
    8. Restore the pre-rotate heading (:func:`TaskContext.rotate_to`) so the base
       ends the pick in its original orientation, whether or not the grasp succeeded.

    Returns True only when the grasp executed cleanly. Degrades to False (never
    raises) at any failing step.

    Note: distinct from :func:`tasks.manipulation.pick_object` (which takes a
    pre-detected ``DetectedObject``) — import this one from ``tasks.skills``.
    """
    last_xyz: Vec3 | None = None

    def _grasp() -> GraspCandidate | None:
        # Always face the object before planning: rotating to centre it in the
        # camera's FOV (then tilting the head down at it) gives the detector and
        # GraspNet a square-on view, which improves accuracy. The first plan has
        # no position estimate yet, so it runs from the current view; every
        # re-plan faces the last known grasp point.
        nonlocal last_xyz
        if last_xyz is not None:
            face_object(ctx, last_xyz)
            look_at_object(ctx, last_xyz)
            time.sleep(0.5)  # let the base/head settle before the snapshot
        cand = get_object_grasp_pos(
            ctx, prompts, attempts=attempts, standoff_m=pregrasp_standoff_m,
            approach_preference=approach_preference, approach_weight=approach_weight,
        )
        if cand is not None:
            last_xyz = cand.grasp_xyz
        return cand
    
    home_res = ctx.walkie.arm.go_to_home(group_name="both_arms_lift", pose_name="standby", blocking=True)
    if home_res != "SUCCEEDED":  # staging move: warn but press on
        print(f"[grasp] stage home -> {home_res} (continuing)")

    cand = _grasp()
    if cand is None:
        print(f"[grasp] pick_object: no grasp for {prompts}")
        return False
    if cand.grasp_xyz[2] < min_grasp_z_m:
        print(f"[grasp] pick_object: object too low (z={cand.grasp_xyz[2]:.2f}m < "
              f"{min_grasp_z_m:.2f}m); cannot reach")
        return False

    base_lift_diff_m = ctx.walkie.robot.transform.lookup("base_footprint", "lift_link")["position"]["z"] - ctx.walkie.robot.lift.get(norm_pos=False) / 100.0
    optimum_lift_height = cand.grasp_xyz[2] + 0.12
    print(f"[grasp] pick_object: setting lift to {((optimum_lift_height - base_lift_diff_m) * 100):.2f}m for better reach")
    ctx.walkie.robot.lift.set(pos=(optimum_lift_height - base_lift_diff_m) * 100, norm_pos=False)
    
    # 2. Approach to the optimal standoff if too far, tracking with the head.
    status = approach_object(
        ctx, cand.grasp_xyz, standoff_m=optimal_standoff_m,
        trigger_m=approach_trigger_m, track=track,
    )
    if status == "FAILED":
        print("[grasp] pick_object: approach failed; aborting")
        return False
    
    cand = _grasp()  # grasp again from the new viewpoint
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

    # 4. Rotate the base so the chosen arm points straight at the object. The arm's
    #    forward is taken to be the robot's heading, so this is a rotate-to-face on the
    #    arm's shoulder (face_object_with_arm). After this the base's "forward" genuinely
    #    points at the object, which is exactly what the creep (step 5) and the wrist
    #    re-aim (step 6) both assume. The candidate is a map-frame pose, so it stays
    #    valid across the rotation. We record the pre-rotate heading and restore it once
    #    the grasp is done (success or not) so the base ends where it started.
    original_heading = ctx.current_pose()["heading"]
    face_object_with_arm(ctx, cand.grasp_xyz, arm=chosen)

    try:
        # 5. Creep the base straight forward to close the last gap — the approach often
        #    stops short (Nav2 halts at the table/inflation boundary), leaving the object
        #    too far to grasp accurately. A pure forward translation keeps the map-frame
        #    candidate valid (no re-plan). Skipped when already within grasp_distance_m.
        if grasp_distance_m is not None:
            creep_to_grasp_distance(ctx, cand.grasp_xyz, target_m=grasp_distance_m)

        # 6. Re-point the wrist straight forward at the object (instead of GraspNet's
        #    often-IK-unsolvable orientation), taking the arm's forward to be the robot's.
        #    Done after the creep so it uses the final heading; the new candidate drives
        #    both the arm and the held-object record (so the placer reuses the real grasp
        #    pose). Keeps GraspNet's orientation when disabled.
        if point_at_object:
            cand = aim_forward_candidate(ctx, cand, standoff_m=pregrasp_standoff_m)

        # 7. Final radial reach guard, then execute.
        reach = _xy_dist(ctx, cand.grasp_xyz)
        if reach > max_reach_xy_m:
            print(f"[grasp] pick_object: out of reach (xy={reach:.2f}m > {max_reach_xy_m:.2f}m); aborting")
            return False
        ok = execute_grasp(ctx, cand, arm=chosen, viz=viz)
        if ok:
            # Remember what we're holding (per arm) so tasks.skills.place can put it back
            # down at the same height above whatever surface it's placed on.
            from tasks.skills.held import record_held_object

            record_held_object(
                ctx,
                label=prompts[0] if prompts else "object",
                arm=chosen,
                grasp_xyz=cand.grasp_xyz,
                rotation=cand.rotation,
                width=cand.width,
                footprint_m=cand.object_footprint_m,
                support_surface_z=cand.support_surface_z,
                grasp_to_surface_offset=cand.grasp_to_surface_offset,
            )
        return ok
    finally:
        # 8. Rotate back to the heading we had before facing the object (the arm is
        #    home/tucked by now), so the base ends the pick in its original orientation.
        print(f"[grasp] pick_object: restoring heading to {math.degrees(original_heading):+.0f}deg")
        ctx.rotate_to(original_heading)
