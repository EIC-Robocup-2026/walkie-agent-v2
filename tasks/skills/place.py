"""Place the held object on a free spot of a horizontal surface.

The complement of :mod:`tasks.skills.grasp`. Picking remembers what it grabbed and
how high above its support surface it grabbed it (:mod:`tasks.skills.held`); placing
reads that back, scans for horizontal surfaces (:mod:`interfaces.perception.surfaces`),
picks a clear spot, reconstructs the original grasp height on the new surface, and
commands the arm to set the object down and let go.

    detect_surfaces(ctx)                      # what surfaces are around (for an agent)
    place_object(ctx)                         # auto: nearest reachable surface + free spot
    place_object(ctx, surface=s)              # agent chose the surface
    place_object(ctx, target_xy=(x, y))       # agent chose the spot

Design notes:
- Map frame is gravity-aligned (+Z up), so a horizontal surface is a constant-Z band.
- The stored grasp *rotation* is reused as the release wrist pose (the only pose a
  rigidly-held object can reproduce); the stored grasp *position* is NOT reused — the
  place point is always recomputed from a fresh scan, so it survives odometry drift.
- Like the grasp skill, every step degrades to ``False`` (never raises) so a task can
  score-degrade gracefully.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from interfaces.perception.geometry import voxel_downsample
from interfaces.perception.surfaces import (
    SurfacePlane,
    assign_objects_to_surfaces,
    detect_horizontal_surfaces,
    find_free_placement,
)
from tasks.base import TaskContext
from tasks.skills.grasp import (
    _arm_sides,
    _xy_dist,
    align_arm_to_object,
    approach_object,
    in_arm_deadzone,
    look_at_object,
)
from tasks.skills.held import HeldObject, clear_held_object, held_arms, recall_held_object
from tasks.skills.navigation import tilt_head

Vec3 = tuple[float, float, float]


# --- config helpers ---------------------------------------------------------
def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _surface_kwargs() -> dict:
    """Detection knobs from config (the WALKIE_SURFACE_* family)."""
    # Vertical-surface gate is ON by default (0.85): rejects walls and standing bodies
    # whose normals point sideways. Set the knob to "" to disable. Needs Open3D; degrades
    # to a no-op (one-time warning) without it.
    normal_raw = os.getenv("WALKIE_SURFACE_NORMAL_Z_MIN", "0.85").strip()
    return {
        "z_gap_m": _f("WALKIE_SURFACE_Z_GAP_M", "0.04"),
        "min_points_per_surface": _i("WALKIE_SURFACE_MIN_POINTS", "150"),
        "xy_cluster_eps": _f("WALKIE_SURFACE_XY_EPS_M", "0.05"),
        "xy_min_points": _i("WALKIE_SURFACE_XY_MIN_POINTS", "30"),
        "min_area_m2": _f("WALKIE_SURFACE_MIN_AREA_M2", "0.05"),
        "area_cell_m": _f("WALKIE_SURFACE_AREA_CELL_M", "0.05"),
        "normal_z_min": (float(normal_raw) if normal_raw else None),
        "z_bin_m": _f("WALKIE_SURFACE_Z_BIN_M", "0.01"),
    }


# --- scene cloud + surface scan ---------------------------------------------
def _full_scene_cloud(
    snap,
    *,
    voxel: float | None = None,
    max_depth: float | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    """Lift the whole frame to a map-frame cloud.

    Passes an all-ones mask with **explicit** ``voxel``/``max_points`` so the
    ``WALKIE_GRAPHS_*`` defaults baked into ``mask_to_points`` (voxel 0.02, cap
    2000) don't decimate a whole-scene lift. ``max_depth`` bounds the cloud (stereo
    error grows ~quadratically, so far points are noise anyway).
    """
    if snap is None or not getattr(snap, "has_geometry", False):
        return np.zeros((0, 3), dtype=np.float32)
    voxel = _f("WALKIE_PLACE_SCENE_VOXEL_M", "0.015") if voxel is None else voxel
    max_depth = _f("WALKIE_PLACE_MAX_DEPTH_M", "1.0") if max_depth is None else max_depth
    h, w = snap.depth.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)
    return snap.mask_to_points(
        mask,
        frame="map",
        voxel=voxel,
        max_points=max_points,
        max_depth=max_depth,
        erode_px=0,
    )


def _scan(
    ctx: TaskContext, snap=None, *, max_depth_m: float | None = None
) -> tuple[object, np.ndarray, list[SurfacePlane]]:
    """Snapshot -> full-scene cloud -> horizontal surfaces. Returns all three."""
    if snap is None:
        snap = ctx.snapshot()
    cloud = _full_scene_cloud(snap, max_depth=max_depth_m)
    surfaces = detect_horizontal_surfaces(cloud, **_surface_kwargs())
    return snap, cloud, surfaces


def _scan_multi_tilt(
    ctx: TaskContext,
    *,
    tilts: tuple[float, ...] | None = None,
    settle_sec: float | None = None,
    merge_voxel: float | None = None,
    max_depth_m: float | None = None,
) -> tuple[object, np.ndarray, list[SurfacePlane]]:
    """Two snapshots at different head tilts, fused, then horizontal-surface detection.

    Tilts the head to each of *tilts* (radians, +down), lifts each frame to a
    map-frame full-scene cloud, and fuses them (``vstack`` + ``voxel_downsample``)
    BEFORE detecting surfaces. Two viewpoints fill each other's self-occlusion, so the
    merged cloud gives ``detect_horizontal_surfaces`` denser, more-complete coverage —
    fewer surfaces missed, truer extents. The clouds are already in the (gravity-
    aligned) map frame, so a plain merge is correct — no re-framing needed (unlike the
    grasp path, which must hand GraspNet an optical-frame cloud).

    Falls back to a single :func:`_scan` when only one tilt yields geometry. Never raises.
    """
    tilts = tilts if tilts is not None else (
        _f("WALKIE_PLACE_TILT_A", "0.2"), _f("WALKIE_PLACE_TILT_B", "0.35"),
    )
    settle_sec = _f("WALKIE_PLACE_TILT_SETTLE_SEC", "0.4") if settle_sec is None else settle_sec
    merge_voxel = (
        _f("WALKIE_PLACE_SCENE_VOXEL_M", "0.015") if merge_voxel is None else merge_voxel
    )

    clouds: list[np.ndarray] = []
    last_snap = None
    for t in tilts:
        tilt_head(ctx, t, settle=settle_sec)
        snap = ctx.snapshot()
        if snap is None or not getattr(snap, "has_geometry", False):
            continue
        cloud = _full_scene_cloud(snap, max_depth=max_depth_m)
        if cloud.shape[0]:
            clouds.append(cloud)
            last_snap = snap

    if not clouds:
        print("[place] multi-tilt scan: no geometry; falling back to single scan")
        return _scan(ctx, None, max_depth_m=max_depth_m)
    if len(clouds) == 1:
        merged = clouds[0]
    else:
        merged = voxel_downsample(np.vstack(clouds), merge_voxel)
        print(f"[place] multi-tilt scan: fused {len(clouds)} views -> {merged.shape[0]} pts")
    surfaces = detect_horizontal_surfaces(merged, **_surface_kwargs())
    return last_snap, merged, surfaces


def _scan_auto(
    ctx: TaskContext, snap=None, *, max_depth_m: float | None = None
) -> tuple[object, np.ndarray, list[SurfacePlane]]:
    """Scan the scene, using the 2-tilt fused scan when enabled and no *snap* is given.

    Gated by ``WALKIE_PLACE_MULTI_TILT`` (default on). When a caller passes an explicit
    *snap* we honour it with a plain single :func:`_scan` (it already chose the frame).
    ``max_depth_m`` (``None`` -> the ``WALKIE_PLACE_MAX_DEPTH_M`` config default) bounds
    the lifted cloud so far background (walls/people) never reaches surface detection.
    """
    if snap is None and _b("WALKIE_PLACE_MULTI_TILT", "1"):
        return _scan_multi_tilt(ctx, max_depth_m=max_depth_m)
    return _scan(ctx, snap, max_depth_m=max_depth_m)


def detect_surfaces(
    ctx: TaskContext,
    *,
    snap=None,
    detect_objects: bool = False,
    object_prompts: list[str] | None = None,
    max_depth_m: float | None = None,
) -> list[SurfacePlane]:
    """Scan for horizontal surfaces around the robot (tables, shelves, the floor).

    Returns :class:`SurfacePlane`\\ s sorted highest-first. With *detect_objects*,
    also runs open-vocab detection (*object_prompts*) and prints which objects sit on
    each surface — the inventory an LLM placement agent reads (height + XY distance).
    Object detection is off by default to keep the place path fast. *max_depth_m* bounds
    the lifted cloud (``None`` -> the ``WALKIE_PLACE_MAX_DEPTH_M`` config default, 1 m) so
    background walls/people beyond the workspace are never mistaken for surfaces. Pass a
    larger *max_depth_m* to inventory surfaces from a distance (the 1 m default sees only
    what's right in front of the robot).
    """
    snap, _cloud, surfaces = _scan_auto(ctx, snap, max_depth_m=max_depth_m)
    if detect_objects and snap is not None and getattr(snap, "has_geometry", False):
        prompts = object_prompts or ["object"]
        objects: list[tuple[str, Vec3]] = []
        try:
            dets = ctx.walkieAI.image.detect(snap.img, prompts=prompts)
        except Exception as exc:  # noqa: BLE001
            print(f"[place] detect_surfaces: detection failed ({exc})")
            dets = []
        for det in dets:
            try:
                xyz = snap.bbox_world_point(det.bbox)
            except Exception:  # noqa: BLE001
                xyz = None
            if xyz is not None:
                objects.append((getattr(det, "class_name", None) or "object", xyz))
        assignment = assign_objects_to_surfaces(surfaces, objects)
        for s in surfaces:
            on = assignment.get(s.id, [])
            labels = ", ".join(o["label"] for o in on) or "(empty)"
            print(f"[place] surface {s.id}: z={s.z:.2f}m area={s.area:.2f}m^2 holds: {labels}")
    return surfaces


# --- base / arm geometry helpers --------------------------------------------
def _arm_mount_xy(ctx: TaskContext, arm: str) -> tuple[float, float] | None:
    """Map-frame (x, y) of *arm*'s shoulder link, for biasing the placement spot."""
    frame = f"openarm_{arm}_link3"
    try:
        tf = ctx.walkie.robot.transform.lookup("map", frame, timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[place] _arm_mount_xy: lookup({frame}) failed ({exc})")
        return None
    if tf and "position" in tf:
        return float(tf["position"]["x"]), float(tf["position"]["y"])
    return None


def _set_lift_for(ctx: TaskContext, ee_z: float) -> None:
    """Raise the torso lift so the arm reaches *ee_z* (mirrors pick_object).

    Targets ``lift_link`` ~0.1 m above *ee_z* so the end-effector sits within the
    arm's comfortable reach. *ee_z* should be the **hover** height (a bit above the
    surface), since the final descent onto the surface is done with the lift itself
    (see :func:`execute_place`).
    """
    try:
        base_lift_diff_m = (
            ctx.walkie.robot.transform.lookup("base_footprint", "lift_link")["position"]["z"]
            - ctx.walkie.robot.lift.get(norm_pos=False) / 100.0
        )
        optimum = ee_z + 0.1
        target_cm = (optimum - base_lift_diff_m) * 100
        print(f"[place] setting lift to {target_cm:.1f} (for ee_z={ee_z:.2f}m)")
        ctx.walkie.robot.lift.set(pos=target_cm, norm_pos=False)
    except Exception as exc:  # noqa: BLE001 — off-robot stub / missing lift
        print(f"[place] _set_lift_for: lift adjust failed ({exc})")


def _nudge_lift(ctx: TaskContext, delta_m: float) -> float | None:
    """Move the torso lift by *delta_m* metres (negative = down).

    The lift is a precise vertical prismatic joint, so with the arm joints held this
    translates the end-effector straight down/up by *delta_m*. Returns the previous
    lift position (cm) so the caller can restore it, or ``None`` when the lift can't
    be read/commanded (off-robot stub).
    """
    try:
        cur_cm = ctx.walkie.robot.lift.get(norm_pos=False)
        ctx.walkie.robot.lift.set(pos=cur_cm + delta_m * 100.0, norm_pos=False)
        return float(cur_cm)
    except Exception as exc:  # noqa: BLE001 — off-robot stub / missing lift
        print(f"[place] _nudge_lift: lift move failed ({exc})")
        return None


def _rotation_for(held: HeldObject, place_orientation: str) -> np.ndarray:
    """The release wrist rotation (3x3, map frame).

    ``"hold"`` reuses the stored grasp orientation — the only pose a rigidly-held
    object can reproduce. ``"topdown"`` builds an overhead pose from
    ``WALKIE_GRASP_RPY_TOPDOWN`` (reused from the manipulation config).
    """
    if place_orientation == "topdown":
        raw = os.getenv("WALKIE_GRASP_RPY_TOPDOWN", "-2.623,-0.033,-1.468")
        rpy = [float(p.strip()) for p in raw.split(",")]
        return Rotation.from_euler("xyz", rpy).as_matrix()
    return np.asarray(held.rotation, dtype=float)


def _choose_surface(
    ctx: TaskContext,
    surfaces: list[SurfacePlane],
    cloud: np.ndarray,
    *,
    footprint_m: float,
    clearance_m: float,
    min_place_z_m: float,
    arm_xy: tuple[float, float] | None,
) -> tuple[SurfacePlane, tuple[float, float]] | None:
    """Pick the nearest surface (within reach height) that has a free cell.

    Returns ``(surface, free_xy)`` or ``None``. Surfaces are tried nearest-first
    (planar distance from the robot), skipping any below ``min_place_z_m`` (the arm
    can't reach that low) — the same height guard the pick path uses.
    """
    p = ctx.current_pose()
    reachable = [s for s in surfaces if s.z >= min_place_z_m]
    reachable.sort(key=lambda s: s.distance_xy(p["x"], p["y"]))
    for s in reachable:
        xy = find_free_placement(
            s,
            cloud,
            footprint_m=footprint_m,
            clearance_m=clearance_m,
            surface_skin_m=_f("WALKIE_PLACE_SURFACE_SKIN_M", "0.02"),
            cell_m=_f("WALKIE_PLACE_CELL_M", "0.04"),
            edge_margin_m=_f("WALKIE_PLACE_EDGE_MARGIN_M", "0.05"),
            prefer="near" if arm_xy is not None else "center",
            prefer_xy=arm_xy,
        )
        if xy is not None:
            return s, xy
    return None


# --- arm release ------------------------------------------------------------
def execute_place(
    ctx: TaskContext,
    place_xyz: Vec3,
    rotation: np.ndarray,
    *,
    arm: str = "left",
    standoff_m: float = 0.10,
    lower_m: float = 0.08,
    settle_sec: float = 0.6,
    home_pose: str = "standby",
    tuck_on_abort: bool = True,
    viz: bool = True,
) -> bool:
    """Set the held object down at the map-frame *place_xyz* and release it.

    The arm does NOT descend to the surface itself — near full extension its
    cartesian IK is imprecise, so the object ends up dropped from too high. Instead
    the arm parks the object *lower_m* metres **above** the surface, then the precise
    torso **lift** lowers it straight down onto the surface before releasing:

        pre-place above -> arm descends to the hover (lower_m above the surface)
        -> lift lowers by lower_m (gentle, accurate) -> open gripper
        -> lift raises back (gripper lifts straight off) -> home.

    Gripper collision is disabled across the move (the held object would otherwise
    trip planning near the surface) and re-enabled in ``finally``. Each arm move's
    result string is checked; aborts on anything but ``"SUCCEEDED"``. Falls back to an
    arm-only descent when the lift can't be commanded (off-robot). Never raises.
    """
    motion_group, home_group, gripper_group = _arm_sides(arm)
    side = motion_group.split("_")[0]
    hand = getattr(ctx.walkie.arm, side)

    try:
        roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz")
    except Exception as exc:  # noqa: BLE001
        print(f"[place] execute_place[{side}]: bad rotation matrix ({exc})")
        return False

    px, py, pz = float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])
    hover = (px, py, pz + lower_m)  # park here; the lift does the final descent
    above = (px, py, pz + lower_m + standoff_m)
    print(f"[place] execute_place[{side}]: place=({px:+.3f},{py:+.3f},{pz:+.3f}) "
          f"hover=+{lower_m:.2f}m RPY=({roll:+.2f},{pitch:+.2f},{yaw:+.2f})")
    if viz and getattr(ctx, "viz", None) is not None:
        try:
            ctx.viz.clear("place", recursive=True)
            ctx.viz.axes("place/ee", (px, py, pz), rotation=np.asarray(rotation),
                         length=0.10, labels=True)
            ctx.viz.points("place/target", [[px, py, pz]], radii=[0.02],
                           colors=[(0, 200, 255)], labels=["place"])
        except Exception as exc:  # noqa: BLE001 — viz never load-bearing
            print(f"[place] viz failed ({exc})")

    succeeded = False
    collision_disabled = False
    prev_lift_cm: float | None = None
    try:
        ctx.walkie.arm.toggle_gripper_collision(gripper_group, False)
        collision_disabled = True

        res = ctx.walkie.arm.go_to_pose(
            *above, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True,
        )
        print(f"[place] execute_place[{side}]: pre-place -> {res}")
        if res != "SUCCEEDED":
            print(f"[place] execute_place[{side}]: pre-place failed; aborting")
            return False

        res = ctx.walkie.arm.go_to_pose(
            *hover, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True, cartesian_path=True,
        )
        print(f"[place] execute_place[{side}]: hover -> {res}")
        if res != "SUCCEEDED":
            print(f"[place] execute_place[{side}]: hover move failed; aborting")
            return False

        # Lower the object onto the surface with the precise lift, not the arm.
        prev_lift_cm = _nudge_lift(ctx, -lower_m)
        if prev_lift_cm is not None:
            time.sleep(settle_sec)
        else:
            # No lift control (off-robot) — fall back to an arm descent to the spot.
            res = ctx.walkie.arm.go_to_pose(
                px, py, pz, roll, pitch, yaw,
                group_name=home_group, frame_id="map", blocking=True, cartesian_path=True,
            )
            print(f"[place] execute_place[{side}]: no lift; arm descent -> {res}")

        hand.gripper(1.0, blocking=True)  # open: release the object

        # Raise the lift back so the open gripper lifts straight up off the object
        # before any arm motion (avoids dragging it or knocking neighbours).
        if prev_lift_cm is not None:
            try:
                ctx.walkie.robot.lift.set(pos=prev_lift_cm, norm_pos=False)
                time.sleep(settle_sec)
            except Exception as exc:  # noqa: BLE001
                print(f"[place] execute_place[{side}]: lift raise-back failed ({exc})")
            prev_lift_cm = None  # restored; don't restore again in finally

        res = ctx.walkie.arm.go_to_pose(
            *above, roll, pitch, yaw,
            group_name=home_group, frame_id="map", blocking=True, cartesian_path=True,
        )
        print(f"[place] execute_place[{side}]: retreat -> {res}")

        ctx.walkie.arm.go_to_home(group_name=motion_group, pose_name=home_pose, blocking=True)
        succeeded = True
        print(f"[place] execute_place[{side}]: success")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[place] execute_place[{side}]: hardware error ({exc})")
        return False
    finally:
        # If we lowered the lift but never raised it back (abort/error), restore it.
        if prev_lift_cm is not None:
            try:
                ctx.walkie.robot.lift.set(pos=prev_lift_cm, norm_pos=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[place] execute_place[{side}]: lift restore failed ({exc})")
        if collision_disabled:
            try:
                ctx.walkie.arm.toggle_gripper_collision(gripper_group, True)
            except Exception as exc:  # noqa: BLE001
                print(f"[place] execute_place[{side}]: re-enable gripper collision failed ({exc})")
        if tuck_on_abort and not succeeded:
            try:
                ctx.walkie.arm.go_to_home(group_name=home_group, pose_name="hands_up", blocking=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[place] execute_place[{side}]: tuck-on-abort home failed ({exc})")


# --- place-plan visualization -----------------------------------------------
def _draw_place_plan(
    ctx: TaskContext,
    cloud: np.ndarray,
    surfaces: list[SurfacePlane],
    chosen_surface: SurfacePlane,
    place_xyz: Vec3,
    rotation: np.ndarray,
    *,
    footprint_m: float | None = None,
) -> None:
    """Best-effort: show the place plan in the shared viewer BEFORE the arm moves.

    Logs, under the ``place_plan/`` namespace (a SIBLING of ``place/`` so
    :func:`execute_place`'s ``clear("place")`` can't wipe it mid-motion):

    - ``place_plan/scene``     — the full map-frame scene cloud (gray, decimated if
      huge): the "places to place" point cloud the surface scan ran on;
    - ``place_plan/surfaces``  — every detected horizontal surface as a flat AABB box;
    - ``place_plan/chosen``    — the chosen surface's AABB, highlighted (cyan);
    - ``place_plan/spot``      — the chosen free spot (the place point) as a marker,
      with the held object's footprint as a box around it;
    - ``place_plan/ee``        — the release wrist pose (the reused grasp orientation,
      i.e. the grasp position the object will be set down in) as an XYZ triad.

    So an operator can eyeball *where* the object will land, on *which* surface, and
    *how* the gripper is oriented before committing. Never raises — viz is never
    load-bearing (mirrors ``grasp._draw_grasp_viz``).
    """
    if getattr(ctx, "viz", None) is None:
        return
    try:
        ctx.viz.clear("place_plan", recursive=True)

        # 1. The scene cloud — the candidate "places to place". Bound the stream size
        #    the same way the scene-graph background viz does.
        if cloud is not None and len(cloud):
            pts = cloud
            if len(pts) > 50_000:
                pts = pts[:: len(pts) // 50_000 + 1]
            ctx.viz.points("place_plan/scene", pts, colors=[(150, 150, 150)], radii=0.004)

        # 2. Every detected surface as a thin AABB slab at its top height.
        for s in surfaces:
            cx = (s.aabb_min[0] + s.aabb_max[0]) / 2.0
            cy = (s.aabb_min[1] + s.aabb_max[1]) / 2.0
            half = [
                max((s.aabb_max[0] - s.aabb_min[0]) / 2.0, 1e-3),
                max((s.aabb_max[1] - s.aabb_min[1]) / 2.0, 1e-3),
                0.005,
            ]
            ctx.viz.box(["place_plan", "surfaces", str(s.id)], [cx, cy, s.z], half,
                        color=(90, 90, 110), label=f"surface {s.id} z={s.z:.2f}m")

        # 3. The chosen surface, highlighted (drawn separately so an explicit
        #    surface= that isn't in the scan's list still shows up).
        cs = chosen_surface
        ccx = (cs.aabb_min[0] + cs.aabb_max[0]) / 2.0
        ccy = (cs.aabb_min[1] + cs.aabb_max[1]) / 2.0
        chalf = [
            max((cs.aabb_max[0] - cs.aabb_min[0]) / 2.0, 1e-3),
            max((cs.aabb_max[1] - cs.aabb_min[1]) / 2.0, 1e-3),
            0.006,
        ]
        ctx.viz.box("place_plan/chosen", [ccx, ccy, cs.z], chalf,
                    color=(0, 200, 255), label=f"chosen z={cs.z:.2f}m")

        # 4. The chosen spot, the object's footprint there, and the release wrist pose.
        px, py, pz = float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])
        ctx.viz.points("place_plan/spot", [[px, py, pz]], radii=[0.025],
                       colors=[(255, 180, 0)], labels=["place"])
        if footprint_m:
            fp = float(footprint_m) / 2.0
            ctx.viz.box("place_plan/footprint", [px, py, pz + fp], [fp, fp, fp],
                        color=(255, 180, 0), label="object")
        ctx.viz.axes("place_plan/ee", (px, py, pz), rotation=np.asarray(rotation),
                     length=0.10, labels=True)
    except Exception as exc:  # noqa: BLE001 — viz is never load-bearing
        print(f"[place] place-plan viz failed ({exc})")


# --- full place -------------------------------------------------------------
def place_object(
    ctx: TaskContext,
    *,
    arm: str = "auto",
    surface: SurfacePlane | None = None,
    target_xy: tuple[float, float] | None = None,
    place_orientation: str = "hold",
    footprint_m: float | None = None,
    clearance_m: float | None = None,
    place_z_offset_m: float | None = None,
    lower_m: float | None = None,
    optimal_standoff_m: float = 0.55,
    approach_trigger_m: float = 0.60,
    max_reach_xy_m: float | None = None,
    min_place_z_m: float | None = None,
    max_depth_m: float | None = None,
    approach_max_depth_m: float | None = None,
    default_arm: str = "left",
    track: bool = True,
    viz: bool = True,
) -> bool:
    """Put the object the robot is holding down on a clear spot of a surface.

    Recalls the held object (per-arm), chooses a surface (explicit *surface*, else
    near *target_xy*, else the nearest reachable one with free space), finds an empty
    spot sized for the object, reconstructs the place height as
    ``surface.z + grasp_to_surface_offset + place_z_offset_m`` (so it lands as it was
    grasped), positions the base so the holding arm reaches, and releases.

    Args:
        ctx: Task context.
        arm: ``"auto"`` (the arm currently holding something), ``"left"`` or ``"right"``.
        surface: Place on this surface (skips surface selection).
        target_xy: Map-frame ``(x, y)`` to place at (skips free-space search).
        place_orientation: ``"hold"`` (reuse grasp pose) or ``"topdown"``.
        footprint_m: Object footprint for empty-space sizing; defaults to the value
            recorded at pick time, then the ``WALKIE_PLACE_FOOTPRINT_M`` config.
        clearance_m, place_z_offset_m, max_reach_xy_m, min_place_z_m: see config.
        max_depth_m: bound the *close* (post-approach) scene scan to this depth (None ->
            the ``WALKIE_PLACE_MAX_DEPTH_M`` config default, 1 m) so background walls/people
            can't be detected as the placement surface and the height stays accurate.
        approach_max_depth_m: depth bound for the *initial* surface-finding scan, run
            before the robot drives in (None -> ``WALKIE_PLACE_APPROACH_MAX_DEPTH_M``, 2 m).
            Wider than ``max_depth_m`` so a table at standoff range is still seen; the
            surface-normal gate still rejects walls/standing bodies at this range, and the
            nearest reachable surface is chosen, so far background isn't picked.
        optimal_standoff_m, approach_trigger_m: base approach to the surface.
        default_arm: fallback when the holding arm can't be inferred.

    Returns:
        True only when the object was released at a located spot; False (never
        raises) at any failing step, leaving the held-object record intact for retry.
    """
    clearance_m = _f("WALKIE_PLACE_CLEARANCE_M", "0.03") if clearance_m is None else clearance_m
    place_z_offset_m = (
        _f("WALKIE_PLACE_Z_OFFSET_M", "0.02") if place_z_offset_m is None else place_z_offset_m
    )
    max_reach_xy_m = (
        _f("WALKIE_PLACE_MAX_REACH_XY_M", "0.75") if max_reach_xy_m is None else max_reach_xy_m
    )
    min_place_z_m = (
        _f("WALKIE_PLACE_MIN_Z_M", "0.70") if min_place_z_m is None else min_place_z_m
    )
    approach_max_depth_m = (
        _f("WALKIE_PLACE_APPROACH_MAX_DEPTH_M", "2.0")
        if approach_max_depth_m is None
        else approach_max_depth_m
    )
    standoff_m = _f("WALKIE_PLACE_STANDOFF_M", "0.10")
    lower_m = _f("WALKIE_PLACE_LOWER_M", "0.08") if lower_m is None else lower_m
    settle_sec = _f("WALKIE_PLACE_SETTLE_SEC", "0.6")

    # 1. Which arm is holding something.
    chosen = (arm or "auto").strip().lower()
    if chosen == "auto":
        arms = held_arms(ctx)
        if not arms:
            print("[place] place_object: nothing is held; nothing to place")
            return False
        if len(arms) > 1:
            print(f"[place] place_object: both arms hold something ({arms}); "
                  f"specify arm= explicitly")
            return False
        chosen = arms[0]
    elif chosen not in ("left", "right"):
        print(f"[place] place_object: bad arm {arm!r}; using {default_arm}")
        chosen = default_arm

    held = recall_held_object(ctx, chosen)
    if held is None:
        print(f"[place] place_object: {chosen} arm is not holding anything")
        return False

    footprint_m = (
        footprint_m
        if footprint_m is not None
        else (held.footprint_m or _f("WALKIE_PLACE_FOOTPRINT_M", "0.12"))
    )
    arm_xy = _arm_mount_xy(ctx, chosen)

    # 2. Choose a surface + spot. Snapshot now, with the arm tucked from the pick, so
    #    the held object/gripper stay out of the table view. The 2-tilt fused scan
    #    (when enabled) gives surface detection denser, more-complete coverage. Use the
    #    wider approach depth here: the robot is still at standoff range, so a tight 1 m
    #    cut would clip the table and abort before we ever drive in (the normal gate still
    #    rejects walls/standing bodies, and the nearest reachable surface is chosen).
    snap, cloud, surfaces = _scan_auto(ctx, None, max_depth_m=approach_max_depth_m)
    if surface is not None:
        chosen_surface = surface
    elif not surfaces:
        print("[place] place_object: no horizontal surfaces detected")
        return False
    elif target_xy is not None:
        # Surface under/nearest the requested spot.
        tx, ty = target_xy
        containing = [s for s in surfaces if s.contains_xy(tx, ty) and s.z >= min_place_z_m]
        chosen_surface = (
            max(containing, key=lambda s: s.z)
            if containing
            else min(surfaces, key=lambda s: s.distance_xy(tx, ty))
        )
    else:
        picked = _choose_surface(
            ctx, surfaces, cloud,
            footprint_m=footprint_m, clearance_m=clearance_m,
            min_place_z_m=min_place_z_m, arm_xy=arm_xy,
        )
        if picked is None:
            print("[place] place_object: no reachable surface with free space")
            return False
        chosen_surface, free_xy = picked

    # 3. Spot on the chosen surface (skip search when target/auto already gave one).
    if surface is not None or target_xy is not None:
        if target_xy is not None and chosen_surface.contains_xy(*target_xy):
            free_xy = (float(target_xy[0]), float(target_xy[1]))
        else:
            free_xy = find_free_placement(
                chosen_surface, cloud,
                footprint_m=footprint_m, clearance_m=clearance_m,
                surface_skin_m=_f("WALKIE_PLACE_SURFACE_SKIN_M", "0.02"),
                cell_m=_f("WALKIE_PLACE_CELL_M", "0.04"),
                edge_margin_m=_f("WALKIE_PLACE_EDGE_MARGIN_M", "0.05"),
                prefer="near" if arm_xy is not None else "center",
                prefer_xy=arm_xy,
            )
            if free_xy is None:
                print("[place] place_object: chosen surface has no free space")
                return False

    # 4. Place height: reconstruct the grasp-to-surface offset on the new surface.
    offset = held.grasp_to_surface_offset
    if offset is None:
        offset = _f("WALKIE_PLACE_FALLBACK_OFFSET_M", "0.05")
        print(f"[place] place_object: no recorded offset; using fallback {offset:.2f}m")
    place_z = chosen_surface.z + offset + place_z_offset_m
    if place_z < min_place_z_m:
        print(f"[place] place_object: place_z={place_z:.2f}m below reach floor "
              f"{min_place_z_m:.2f}m; aborting")
        return False
    place_xyz: Vec3 = (free_xy[0], free_xy[1], place_z)
    print(f"[place] place_object: surface z={chosen_surface.z:.2f}m offset={offset:.2f}m "
          f"-> place=({place_xyz[0]:+.2f},{place_xyz[1]:+.2f},{place_z:.2f})")

    # 5. Raise the lift for the HOVER height (the lift does the final descent in
    #    execute_place), then drive to a standoff facing the spot.
    _set_lift_for(ctx, place_z + lower_m)
    status = approach_object(
        ctx, place_xyz, standoff_m=optimal_standoff_m,
        trigger_m=approach_trigger_m, track=track,
    )
    if status == "FAILED":
        print("[place] place_object: approach failed; aborting")
        return False
    if status == "MOVED":
        # Base moved -> the scan is stale. Re-scan and re-pick the same surface/spot.
        snap, cloud, surfaces = _scan_auto(ctx, None, max_depth_m=max_depth_m)
        if surfaces:
            chosen_surface = min(
                surfaces,
                key=lambda s: math.hypot(
                    s.centroid[0] - chosen_surface.centroid[0],
                    s.centroid[1] - chosen_surface.centroid[1],
                ),
            )
            arm_xy = _arm_mount_xy(ctx, chosen)
            re_xy = find_free_placement(
                chosen_surface, cloud,
                footprint_m=footprint_m, clearance_m=clearance_m,
                surface_skin_m=_f("WALKIE_PLACE_SURFACE_SKIN_M", "0.02"),
                cell_m=_f("WALKIE_PLACE_CELL_M", "0.04"),
                edge_margin_m=_f("WALKIE_PLACE_EDGE_MARGIN_M", "0.05"),
                prefer="near" if arm_xy is not None else "center",
                prefer_xy=arm_xy,
            )
            if re_xy is not None:
                place_z = chosen_surface.z + offset + place_z_offset_m
                place_xyz = (re_xy[0], re_xy[1], place_z)
                _set_lift_for(ctx, place_z + lower_m)
        look_at_object(ctx, place_xyz)

    # 6. Strafe the base so the holding arm lines up, then guard reach.
    align_arm_to_object(ctx, place_xyz, arm=chosen)
    # if in_arm_deadzone(ctx, place_xyz):
    #     print("[place] place_object: spot still in the arm dead-zone after aligning; aborting")
    #     return False
    reach = _xy_dist(ctx, place_xyz)
    if reach > max_reach_xy_m:
        print(f"[place] place_object: out of reach (xy={reach:.2f}m > {max_reach_xy_m:.2f}m); aborting")
        return False

    # 7. Release — park above the surface, then lower onto it with the lift. Draw the
    #    plan first (scene cloud, candidate surfaces, the spot + the release wrist pose)
    #    so an operator can eyeball the place before the arm commits.
    rotation = _rotation_for(held, place_orientation)
    if viz:
        _draw_place_plan(
            ctx, cloud, surfaces, chosen_surface, place_xyz, rotation,
            footprint_m=footprint_m,
        )
    ok = execute_place(
        ctx, place_xyz, rotation, arm=chosen,
        standoff_m=standoff_m, lower_m=lower_m, settle_sec=settle_sec, viz=viz,
    )
    if ok:
        clear_held_object(ctx, chosen)
        print(f"[place] place_object: placed {held.label!r}; {chosen} arm now empty")
    return ok
