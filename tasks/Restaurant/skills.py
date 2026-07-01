"""Reusable perception / interaction skills for the Restaurant task (rulebook 5.5).

Phase 0 (this slice) is REAL: scan the dining area, detect a calling/waving
customer from pose keypoints, lift them to a map point, and approach to a safe
stand-off facing them. Order-taking dialogue is real too. Manipulation
(pick/serve/tray) stays an honest stub for Phase 2.

Plain functions over a TaskContext, same style as tasks/HRI/skills.py. Everything
is best-effort: an AI-client / odometry failure logs and degrades, never raises.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass

from client import PersonPose
from tasks.base import TaskContext

from . import prompts

try:  # servo tilt limits (+ = down, - = up); mirror the SDK on a dev box w/o the robot
    from walkie_sdk.modules.head import HEAD_TILT_MAX, HEAD_TILT_MIN
except Exception:  # pragma: no cover - SDK import is hardware-side
    HEAD_TILT_MIN, HEAD_TILT_MAX = -math.pi / 4, math.pi / 3

BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _b(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _cxcywh_to_xyxy(bbox) -> BBox:
    """Pose-estimation bboxes are (cx, cy, w, h); the depth lift wants xyxy."""
    cx, cy, w, h = bbox
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _robot_pose(ctx: TaskContext) -> dict | None:
    """Real odometry fix, or None — never the zeros fallback (that would mis-aim)."""
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[restaurant.skills] odometry unavailable ({exc})")
        return None
    return pose or None


def _wrap(a: float) -> float:
    """Wrap an angle (rad) to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def _ray_point(pose: dict, bearing: float, dist: float) -> tuple[float, float]:
    """A map point *dist* metres ahead of *pose* along *bearing* (rad)."""
    return (pose["x"] + dist * math.cos(bearing), pose["y"] + dist * math.sin(bearing))


# ---------------------------------------------------------------------------
# Calling / waving detection (keypoints)
# ---------------------------------------------------------------------------
@dataclass
class Caller:
    """A customer detected calling/waving, anchored for approach.

    ``world_xy`` is the lifted map point when depth was valid at capture; the live
    bearing-first scan (``live_scan_for_caller``) leaves it ``None`` for a FAR caller
    whose depth is out of the ZED's reliable range — ``bearing`` is always set, and the
    live approach drives the bearing ray and refines the point once the customer is
    close. The old discrete sweep always lifts a point (it skips callers it can't), so
    its callers always carry a ``world_xy``.
    """

    world_xy: tuple[float, float] | None  # map-frame position, or None (far/no depth)
    bearing: float                 # map-frame heading from the robot toward them
    bbox_xyxy: BBox                # pixel box in the frame they were found
    confidence: float              # person-detection confidence


def is_calling(person: PersonPose, *, margin_frac: float | None = None,
               kp_conf: float | None = None) -> bool:
    """True when the person has a hand raised (a calling/waving signal).

    Keypoint heuristic: a wrist sits clearly ABOVE its shoulder. Image y grows
    downward, so "above" is ``wrist.y < shoulder.y``; the margin (a fraction of
    the person's bbox height) rejects an arm merely resting near shoulder level.
    Either arm counts. A future temporal pass can add wrist motion across frames
    for true "waving"; a single raised hand already scores "calling".
    """
    if margin_frac is None:
        margin_frac = _f("RESTAURANT_CALL_WRIST_MARGIN", "0.05")
    if kp_conf is None:
        kp_conf = _f("RESTAURANT_KP_CONF", "0.3")
    kp = {k.name: k for k in person.keypoints}
    h = float(person.bbox[3]) or 1.0  # bbox height in px
    margin = margin_frac * h

    def vis(name: str):
        k = kp.get(name)
        return k if (k is not None and k.confidence >= kp_conf) else None

    for side in ("left", "right"):
        wrist = vis(f"{side}_wrist")
        shoulder = vis(f"{side}_shoulder")
        if wrist is not None and shoulder is not None and wrist.y < shoulder.y - margin:
            return True
    return False


def _torso_bbox(person: PersonPose, *, kp_conf: float) -> BBox | None:
    """A small pixel patch on the person's upper chest, from the two shoulder keypoints.

    Lifting depth HERE (on the body) instead of from the whole person bbox avoids the
    median-of-bbox trap: a nearer table in the FOREGROUND of a far customer's box carries
    valid depth, so the bbox median lands on that near table and the robot drives to an
    empty nearby table instead of the customer.

    The patch is sized off the SHOULDER WIDTH, not the person bbox height: a waving
    customer's bbox is stretched tall by the raised arm, so a height-based patch reached
    down past the chest onto the lap/table/floor and the median drifted off the body —
    the point cloud had the person, but the reduced point didn't (the bug the team saw).
    Shoulder width is arm-independent, keeping the patch a tight sternum square. Returns
    xyxy, or None if both shoulders aren't confidently visible (caller then falls back to
    the full-bbox lift).
    """
    kp = {k.name: k for k in person.keypoints}

    def vis(name: str):
        k = kp.get(name)
        return k if (k is not None and k.confidence >= kp_conf) else None

    ls, rs = vis("left_shoulder"), vis("right_shoulder")
    if ls is None or rs is None:
        return None
    x1, x2 = sorted((float(ls.x), float(rs.x)))
    sw = (x2 - x1) or 1.0                  # shoulder width (px) — arm-independent scale
    y_sh = max(float(ls.y), float(rs.y))   # the lower shoulder
    inset = 0.20 * sw                      # pull the sides in off the arms / background gaps
    return (x1 + inset, y_sh + 0.10 * sw, x2 - inset, y_sh + 0.70 * sw)


def _person_world_xy(snap, person: PersonPose,
                     *, kp_conf: float | None = None) -> tuple[float, float] | None:
    """Map-frame (x, y) of a detected person — torso-keypoint lift, full-bbox fallback.

    Prefers a small chest patch (:func:`_torso_bbox`) so a far customer isn't placed on
    a nearer foreground table (see there); falls back to the full-bbox median lift when
    the shoulders aren't visible. None without geometry / no valid depth either way.
    """
    if not getattr(snap, "has_geometry", False):
        return None
    if kp_conf is None:
        kp_conf = _f("RESTAURANT_KP_CONF", "0.3")
    tb = _torso_bbox(person, kp_conf=kp_conf)
    if tb is not None:
        wxy = snap.bbox_world_xy(tb, shrink=1.0)  # already tight & on-body; don't shrink
        if wxy is not None:
            return wxy
    return snap.bbox_world_xy(_cxcywh_to_xyxy(person.bbox))


def _scan_offsets() -> list[float]:
    """Base-rotation offsets (deg) covering the scan arc, CENTER-OUT from the entry heading.

    Ordered by distance from centre — [0, -step, +step, -2·step, +2·step, …] — so the
    sweep checks STRAIGHT AHEAD first and only turns to the edges if nobody's there.
    With approach-on-first-sighting this means: when the next customer is in front (the
    common case, since we return to the bar facing the diners), the robot finds them
    without rotating the base at all — no swinging out to ±90° first ("turning idly
    before getting to work"). The set of angles is unchanged; only the visit order is.
    """
    arc = _f("RESTAURANT_SCAN_ARC_DEG", "180")
    step = max(5.0, _f("RESTAURANT_SCAN_STEP_DEG", "30"))
    half = arc / 2.0
    offsets = []
    off = -half
    while off <= half + 1e-6:
        offsets.append(off)
        off += step
    # Visit nearest-to-centre first (ties: negative before positive, stable).
    offsets.sort(key=lambda o: (abs(o), o))
    return offsets


def _dedup_callers(callers: list[Caller], radius_m: float) -> list[Caller]:
    """Collapse callers whose map points are within *radius_m* (same person, two views)."""
    kept: list[Caller] = []
    for c in callers:
        dup = next(
            (k for k in kept if math.hypot(k.world_xy[0] - c.world_xy[0],
                                           k.world_xy[1] - c.world_xy[1]) <= radius_m),
            None,
        )
        if dup is None:
            kept.append(c)
        elif c.confidence > dup.confidence:  # keep the more confident sighting
            kept[kept.index(dup)] = c
    return kept


# ---------------------------------------------------------------------------
# Bearing-first scan helpers (pure / unit-testable — no ctx / network)
# ---------------------------------------------------------------------------
def _torso_center_u(person: PersonPose, *, kp_conf: float) -> float:
    """Pixel COLUMN of the person's torso centre (RGB image coords).

    Prefers the two-shoulder chest patch (:func:`_torso_bbox`) so a raised waving arm
    doesn't drag the column sideways; falls back to the bbox centre ``cx`` (the pose
    bbox is ``(cx, cy, w, h)``)."""
    tb = _torso_bbox(person, kp_conf=kp_conf)
    if tb is not None:
        return (tb[0] + tb[2]) / 2.0
    return float(person.bbox[0])


def _bearing_from_pixel_u(u_rgb: float, snap, *,
                          robot_heading: float | None = None) -> float | None:
    """Absolute map-frame bearing toward a pixel column ``u_rgb``, from intrinsics.

    Pinhole: the angle off the optical axis is ``atan2(u - cx, fx)`` (positive to the
    RIGHT). This repo's heading sign (FaceTracker, tasks/skills/people.py): a column
    LEFT of centre (``u < cx``) needs a LARGER, CCW heading to face it — so
    ``bearing = heading - atan2(u - cx, fx)``. ``snap.intr`` is at DEPTH resolution while
    keypoint columns are in the RGB image, so ``u`` is scaled by ``intr.width / rgb_w``
    first (what ``CameraSnapshot.bbox_to_points`` does). Heading defaults to the
    snapshot's OWN capture-time ``robot_pose["heading"]`` (tie the bearing to the frame,
    not to where the base has since rotated). Returns ``None`` without intrinsics/heading.
    """
    intr = getattr(snap, "intr", None)
    if intr is None:
        return None
    if robot_heading is None:
        rp = getattr(snap, "robot_pose", None)
        if not rp or rp.get("heading") is None:
            return None
        robot_heading = float(rp["heading"])
    try:
        rgb_w = float(snap.img.size[0])  # PIL .size = (w, h)
    except Exception:  # noqa: BLE001 — synthetic snapshots: assume depth-res columns
        rgb_w = float(intr.width)
    u_depth = u_rgb * (intr.width / rgb_w) if rgb_w else u_rgb
    off = math.atan2(u_depth - intr.cx, intr.fx)  # + = right of optical axis
    return _wrap(robot_heading - off)


def _bearing_is_dup(bearing: float, kept_bearings, dedup_deg: float) -> bool:
    """True when *bearing* is within *dedup_deg* of any already-kept bearing."""
    tol = math.radians(dedup_deg)
    return any(abs(_wrap(bearing - b)) <= tol for b in kept_bearings)


def _caller_is_close(world_xy: tuple[float, float] | None, pose: dict,
                     reliable_m: float) -> bool:
    """CLOSE = a valid depth fix within the reliable band (drive straight to it).

    FAR (return False) = no ``world_xy`` (depth invalid / beyond reliable range) so the
    approach must drive the bearing ray and re-detect. Keys off depth VALIDITY, not a
    hard metre cut — a far customer whose depth reads as noise degrades correctly."""
    if world_xy is None:
        return False
    return math.hypot(world_xy[0] - pose["x"], world_xy[1] - pose["y"]) <= reliable_m


def _person_look_tilt() -> float:
    """Head tilt (rad) to hold whenever we look FOR PEOPLE. Default 0.0 = look straight
    forward (level): the key is to never leave the head pointing DOWN at a table/floor
    (which sees 0 persons). + = down, - = up; clamped to the servo range. Go negative
    via ``RESTAURANT_PERSON_LOOK_TILT_RAD`` (down to ``HEAD_TILT_MIN`` ≈ -45°) only if a
    low camera mount needs to look UP at people."""
    raw = _f("RESTAURANT_PERSON_LOOK_TILT_RAD", "0.0")
    return max(HEAD_TILT_MIN, min(HEAD_TILT_MAX, raw))


def _aim_head_for_people(ctx: TaskContext) -> None:
    """Disable auto-tilt and RAISE the head to the person-search tilt before a capture.

    Used everywhere we run pose estimation to find a person (sweep, re-acquire,
    appearance). Two traps it avoids: (1) ``set_auto_tilt(False)`` alone leaves the head
    wherever it was pointing — often tilted down at a table/floor from a prior action, so
    pose estimation sees 0 persons (the "saw 0 person-detection(s)" symptom); (2)
    ``ctx.rotate_to`` / ``go_to`` RE-ENABLE auto-tilt, so the head must be re-aimed right
    before EACH capture, not once. ``tilt()`` is fire-and-forget, so the callers settle
    briefly after. Best-effort — never breaks the detection on a head fault.
    """
    head = ctx.walkie.robot.head
    try:
        head.set_auto_tilt(False)
        head.tilt(_person_look_tilt())
    except Exception as exc:  # out-of-range / transport — never break the detection
        print(f"[restaurant.skills] person-look head-aim failed ({exc})")


def _aim_for_person_capture(ctx: TaskContext) -> None:
    """Raise the head and let it settle before a ONE-SHOT person capture (re-acquire /
    appearance). The sweep manages its own per-offset settle; these single snapshots
    don't, so without the short dwell ``tilt()`` (fire-and-forget) hasn't physically
    moved when the frame is grabbed and we'd detect against the old, down-pointed view."""
    _aim_head_for_people(ctx)
    settle = _f("RESTAURANT_PERSON_LOOK_SETTLE_SEC", "0.4")
    if settle > 0:
        time.sleep(settle)


def scan_for_callers(ctx: TaskContext) -> list[Caller]:
    """Sweep the base across the dining area and return every calling customer.

    The head only tilts (no pan), so the sweep is a series of small in-place base
    rotations; at each we snapshot, run pose estimation, keep people with a raised
    hand, and lift each to a map point against THAT snapshot's frozen geometry —
    so the world points are correct regardless of where the base was pointing.
    Requires an odometry fix (to turn map points into bearings and to return to
    the entry heading). Returns deduplicated callers; empty on no fix / none found.
    """
    pose = _robot_pose(ctx)
    _aim_head_for_people(ctx)  # raise the head up-front (re-aimed per offset below)
    if pose is None:
        print("[restaurant.skills] scan_for_callers: no odometry fix; skipping sweep")
        ctx.walkie.robot.head.set_auto_tilt(True)
        return []
    rx, ry = pose["x"], pose["y"]
    # Centre the sweep on the DINING area (the bar-anchor heading = the direction the robot
    # faced at the counter, toward the customers), NOT the current heading: after relaying
    # an order the robot is left facing the counter, but we must still sweep the diners to
    # find the next customer. Falls back to the current heading if there's no anchor yet.
    bar = ctx.data.get("bar_anchor")
    center = bar["heading"] if bar else pose["heading"]
    settle = _f("RESTAURANT_SCAN_SETTLE_SEC", "0.8")
    callers: list[Caller] = []
    seen_persons = seen_calling = 0  # diagnostics: why a sweep finds no callers
    for off in _scan_offsets():
        ctx.rotate_to(center + math.radians(off))
        _aim_head_for_people(ctx)  # rotate_to re-enables auto-tilt; re-raise before capturing
        if settle > 0:
            time.sleep(settle)  # let the base + head + depth settle before capturing
        snap = ctx.snapshot()
        if snap is None:
            continue
        try:
            persons = ctx.walkieAI.image.estimate_poses(snap.img)
        except Exception as exc:
            print(f"[restaurant.skills] pose estimation failed ({exc})")
            continue
        seen_persons += len(persons)
        for p in persons:
            if not is_calling(p):
                continue
            seen_calling += 1
            xyxy = _cxcywh_to_xyxy(p.bbox)
            world_xy = _person_world_xy(snap, p)
            if world_xy is None:
                continue  # can't approach what we can't place on the map
            bearing = math.atan2(world_xy[1] - ry, world_xy[0] - rx)
            callers.append(Caller(world_xy, bearing, xyxy, p.confidence or 0.0))
    ctx.rotate_to(center)  # leave the base back at the entry heading
    ctx.walkie.robot.head.set_auto_tilt(True)
    callers = _dedup_callers(callers, _f("RESTAURANT_CALLER_DEDUP_M", "0.6"))
    # The counts pin WHERE detection broke: 0 persons -> pose model saw nobody; persons but
    # 0 calling -> is_calling too strict (tune RESTAURANT_CALL_WRIST_MARGIN / _KP_CONF);
    # calling but 0 callers -> the depth lift failed (no/under-range depth on the body).
    print(f"[restaurant.skills] scan found {len(callers)} caller(s) "
          f"(saw {seen_persons} person-detection(s), {seen_calling} with a raised hand)")
    return callers


def find_first_caller(ctx: TaskContext,
                      blocked: list[tuple[float, float]] | None = None,
                      radius_m: float | None = None) -> Caller | None:
    """Approach-on-first-sighting: sweep the dining area and return the FIRST calling
    customer found, stopping the sweep the instant one appears.

    Unlike :func:`scan_for_callers` (which finishes the whole arc, dedups, and hands the
    batched loop every caller to choose from), this bails out at the first offset that
    yields a usable waver and leaves the base pointed at them — so the robot reacts
    immediately and the customer needn't keep waving through a full sweep + rotate-back.
    ``blocked`` map points (already-handled / given-up spots) are skipped in-sweep; returns
    None if the whole arc is clear of fresh wavers (base left back at the dining centre).
    """
    print("[restaurant.skills] Finding first caller")
    if blocked is None:
        blocked = []
    if radius_m is None:
        radius_m = _f("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    pose = _robot_pose(ctx)
    ctx.walkie.robot.head.set_auto_tilt(False)
    if pose is None:
        print("[restaurant.skills] find_first_caller: no odometry fix; skipping sweep")
        ctx.walkie.robot.head.set_auto_tilt(True)
        return None
    rx, ry = pose["x"], pose["y"]
    bar = ctx.data.get("bar_anchor")
    center = bar["heading"] if bar else pose["heading"]
    settle = _f("RESTAURANT_SCAN_SETTLE_SEC", "0.8")
    seen_persons = seen_calling = 0
    try:
        for off in _scan_offsets():
            ctx.rotate_to(center + math.radians(off))
            _aim_head_for_people(ctx)  # rotate_to re-enables auto-tilt; re-raise before capturing
            if settle > 0:
                time.sleep(settle)  # let the base + head + depth settle before capturing
            snap = ctx.snapshot()
            if snap is None:
                continue
            try:
                persons = ctx.walkieAI.image.estimate_poses(snap.img)
            except Exception as exc:
                print(f"[restaurant.skills] pose estimation failed ({exc})")
                continue
            seen_persons += len(persons)
            # Among the wavers visible at THIS offset, take the nearest usable one and go —
            # don't sweep on past a customer who's already waving at us.
            best: tuple[float, tuple[float, float], object] | None = None
            for p in persons:
                if not is_calling(p):
                    continue
                seen_calling += 1
                wxy = _person_world_xy(snap, p)
                if wxy is None:
                    continue  # waving, but no depth to place them on the map
                if any(math.hypot(wxy[0] - bx, wxy[1] - by) <= radius_m for bx, by in blocked):
                    continue  # already handled / given up on this spot
                d = math.hypot(wxy[0] - rx, wxy[1] - ry)
                if best is None or d < best[0]:
                    best = (d, wxy, p)
            if best is not None:
                d, wxy, p = best
                bearing = math.atan2(wxy[1] - ry, wxy[0] - rx)
                print(f"[restaurant.skills] find_first_caller: waving customer at offset "
                      f"{off:+.0f}deg, {d:.1f}m away → going straight to them")
                return Caller(wxy, bearing, _cxcywh_to_xyxy(p.bbox), p.confidence or 0.0)
        print(f"[restaurant.skills] find_first_caller: none waving across the sweep "
              f"(saw {seen_persons} person-detection(s), {seen_calling} with a raised hand)")
        ctx.rotate_to(center)  # nothing found → leave the base back at the dining centre
        return None
    finally:
        ctx.walkie.robot.head.set_auto_tilt(True)


def nearest_caller(ctx: TaskContext, callers: list[Caller]) -> Caller | None:
    """The calling customer closest to the robot (serve the nearest first — MVP policy)."""
    pose = _robot_pose(ctx)
    if pose is None or not callers:
        return callers[0] if callers else None
    rx, ry = pose["x"], pose["y"]
    return min(
        callers,
        key=lambda c: math.hypot(c.world_xy[0] - rx, c.world_xy[1] - ry),
    )


def exclude_handled(callers: list[Caller], handled_xys: list[tuple[float, float]],
                    radius_m: float | None = None) -> list[Caller]:
    """Drop callers within *radius_m* of any already-handled customer.

    Anti-double-serve: the serial loop re-scans the room after each order, so a
    customer who keeps waving (impatient, didn't notice the robot) would otherwise
    be re-selected and counted a second time — exiting the loop having served one
    distinct person when the rulebook needs >= 2 (the whole no-arm tier). We anchor
    "already handled" on position because the tables are well separated; telling
    apart two people at ONE table is left to ``order.appearance`` (captured + logged)
    as a future disambiguator — see design §5.1.
    """
    if radius_m is None:
        radius_m = _f("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    if not handled_xys:
        return callers
    out: list[Caller] = []
    for c in callers:
        if any(math.hypot(c.world_xy[0] - hx, c.world_xy[1] - hy) <= radius_m
               for hx, hy in handled_xys):
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Approach + facing (SLAM-backed: lift -> go_to a stand-off point)
# ---------------------------------------------------------------------------
def heading_to(ctx: TaskContext, world_xy: tuple[float, float]) -> float | None:
    """Map-frame heading from the robot toward *world_xy*; None without an odom fix."""
    pose = _robot_pose(ctx)
    if pose is None:
        return None
    return math.atan2(world_xy[1] - pose["y"], world_xy[0] - pose["x"])


def face_person(ctx: TaskContext, world_xy: tuple[float, float], *,
                min_turn_deg: float = 8.0) -> bool:
    """Rotate the base to face a map point (head is tilt-only). Best-effort.

    Skips the rotate when already within *min_turn_deg* of facing the point: the gaze
    re-centering in take_order calls this before every utterance, and without the guard
    each call fires a fresh Nav2 rotation even when the robot is already square on the
    customer — pure latency plus needless goal-checker oscillation.
    """
    heading = heading_to(ctx, world_xy)
    if heading is None:
        return False
    pose = _robot_pose(ctx)
    if pose is not None:
        delta = math.atan2(math.sin(heading - pose["heading"]),
                           math.cos(heading - pose["heading"]))
        if abs(math.degrees(delta)) < min_turn_deg:
            return True  # already facing — skip the rotate
    return ctx.rotate_to(heading)


def approach_to_standoff(ctx: TaskContext, world_xy: tuple[float, float], *,
                         standoff_m: float | None = None) -> bool:
    """Drive to a point *standoff_m* short of the customer, then face them.

    Stops at a conversational distance (the rulebook scores reaching the table,
    and any contact in this public venue triggers an e-stop, so we keep clear).
    Already inside the stand-off → just turn to face. Needs an odometry fix.

    NOTE: a conservative approach SPEED for the public space is a Nav2/robot-side
    parameter (RESTAURANT_APPROACH_SPEED is recorded for that tuning); ``go_to``
    here does not cap velocity itself.
    """
    if standoff_m is None:
        standoff_m = _f("RESTAURANT_STANDOFF_M", "0.8")
    pose = _robot_pose(ctx)
    if pose is None:
        print("[restaurant.skills] approach: no odometry fix; cannot navigate")
        return False
    rx, ry = pose["x"], pose["y"]
    cx, cy = world_xy
    dist = math.hypot(cx - rx, cy - ry)
    heading = math.atan2(cy - ry, cx - rx)  # face the customer
    if dist <= standoff_m:
        return ctx.rotate_to(heading)  # already close enough: just face them
    # Obstacle-aware approach. The naive ray stand-off (compute a point standoff_m short
    # of the customer, then NavigateToPose there) lands ON a chair/table when tables are
    # only ~1 m apart — the planner has no idea that point is occupied, so the robot parks
    # at the chair in front of the customer. NavigateToObject hands the customer position
    # + a stand-off to nav_commander, which finds a *reachable* free spot at that radius
    # facing them (heading omitted -> NavigateToObject; face_target = just face the person,
    # skip edge-fit).
    print(f"[restaurant.skills] approach (NavigateToObject) -> customer ({cx:.2f},{cy:.2f}), "
          f"{dist:.2f}m away, standoff {standoff_m:.2f}m")
    tol = float(os.getenv("WALKIE_NAV_GOAL_TOLERANCE_M", "0.0"))
    try:
        status = ctx.walkie.nav.go_to(
            x=cx, y=cy, blocking=True, standoff=standoff_m,
            align_method="face_target",
            # goal_tolerance=(tol if tol > 0 else None),
        )
    except Exception as exc:
        print(f"[restaurant.skills] approach: nav raised ({exc})")
        return False
    # NavigateToObject's status string is unreliable on this robot (cf. grasp.approach_object,
    # which hard-returns "MOVED"), so judge success by GEOMETRY: did we end up near the
    # customer? Accept ending within the stand-off plus a margin (we park in the aisle by
    # the table, not on the person).
    after = _robot_pose(ctx) or pose
    end_dist = math.hypot(cx - after["x"], cy - after["y"])
    reached = end_dist <= standoff_m + max(tol, 0.4)
    print(f"[restaurant.skills] approach -> status={status}, ended {end_dist:.2f}m from customer "
          f"({'reached' if reached else 'too far'})")
    face_person(ctx, world_xy)  # make sure we end facing them for the conversation
    return reached


def approach_customer(ctx: TaskContext, world_xy: tuple[float, float], *,
                      max_steps: int | None = None) -> bool:
    """Approach a waving customer in steps, re-detecting them as we close in.

    The depth camera is only accurate to ~``RESTAURANT_DEPTH_RELIABLE_M`` (≈ 4 m), so a
    customer sitting farther is lifted to a coarse, usually-SHORT point — one drive stops
    well shy of them. We instead close the gap over several steps: drive PART of the way
    (enough to bring the customer inside the reliable depth band), re-detect the SAME
    person from there (``find_person_near`` anchored to the running estimate — never the
    globally nearest caller, so we can't hop to another table), refine the point, and
    repeat. The final step parks at the conversational stand-off. Returns True once we end
    within reach of the customer, False if we lost them / nav refused throughout.

    Stall guard (RESTAURANT_APPROACH_MIN_PROGRESS_M / _MAX_STALLS): after each step we
    measure how far the base ACTUALLY translated. If it barely moved while the customer is
    still far, the path is blocked (chairs / a narrow aisle / no reachable free spot) and
    pressing on just burns steps — so we STOP and take the order from where we stand (return
    True) rather than abandoning a customer we simply can't roll right up to.
    """
    if max_steps is None:
        max_steps = int(os.getenv("RESTAURANT_APPROACH_MAX_STEPS", "3"))
    max_steps = max(1, max_steps)
    # Let the customer know they've been seen and we're on our way (rulebook-friendly
    # HRI; also tells the operator the detection fired). Best-effort — say() degrades
    # to print on audio failure and never blocks the approach.
    ctx.say(prompts.FOUND_CUSTOMER)
    standoff = _f("RESTAURANT_STANDOFF_M", "0.8")
    reliable = _f("RESTAURANT_DEPTH_RELIABLE_M", "3.5")
    min_progress = _f("RESTAURANT_APPROACH_MIN_PROGRESS_M", "0.2")
    max_stalls = max(1, int(os.getenv("RESTAURANT_APPROACH_MAX_STALLS", "1")))
    target = world_xy
    stalls = 0
    for step in range(max_steps):
        pose = _robot_pose(ctx)
        if pose is None:
            return False
        dist = math.hypot(target[0] - pose["x"], target[1] - pose["y"])
        if dist <= standoff + 0.4:  # close enough → final park & done
            return approach_to_standoff(ctx, target, standoff_m=standoff)
        # Close in PART of the way: stop at most `reliable` m from the (coarse) target so
        # the customer lands inside the accurate depth band for the next detection, and at
        # least halve the gap each step. Never closer than the final stand-off.
        intermediate = max(standoff, min(0.5 * dist, reliable))
        print(f"[restaurant.skills] approach_customer step {step + 1}/{max_steps}: "
              f"customer ~{dist:.1f}m away → stepping in to {intermediate:.1f}m")
        before = (pose["x"], pose["y"])
        approach_to_standoff(ctx, target, standoff_m=intermediate)
        # Did the base actually make headway this step? A near-stationary step means the
        # robot can't get any closer (blocked); honour the rulebook-friendly policy of
        # serving from where we can rather than giving the customer up.
        after = _robot_pose(ctx)
        moved = math.hypot(after["x"] - before[0], after["y"] - before[1]) if after else 0.0
        if moved < min_progress:
            stalls += 1
            print(f"[restaurant.skills] approach_customer: base moved only {moved:.2f}m this "
                  f"step (< {min_progress:.2f}m); stall {stalls}/{max_stalls}")
            if stalls >= max_stalls:
                print("[restaurant.skills] approach_customer: can't get closer (path blocked) "
                      "— stopping here and taking the order from this spot")
                face_person(ctx, target)
                return True
        else:
            stalls = 0  # made headway → reset the consecutive-stall streak
        # Re-detect the SAME customer from the closer, now-reliable viewpoint. Anchor to the
        # running estimate with a radius generous enough to absorb the far-lift error
        # (which biases short), tightening as we approach.
        radius = max(_f("RESTAURANT_REACQUIRE_RADIUS_M", "1.5"), 0.6 * dist)
        fresh = find_person_near(ctx, target, radius_m=radius, prefer_calling=True)
        if fresh is not None:
            target = fresh.world_xy
            print(f"[restaurant.skills] approach_customer: refined customer → "
                  f"({target[0]:.2f},{target[1]:.2f})")
        else:
            print("[restaurant.skills] approach_customer: customer not re-acquired; keeping estimate")
    # Ran out of steps — final stand-off attempt at the best estimate we have.
    return approach_to_standoff(ctx, target, standoff_m=standoff)


# ---------------------------------------------------------------------------
# Live continuous-rotation scan + live approach (RESTAURANT_LIVE_SCAN)
# ---------------------------------------------------------------------------
def _look_at_person(ctx: TaskContext, world_xy: tuple[float, float] | None,
                    *, settle: float = 0.0) -> None:
    """Aim the head at a person's map point — clamped so it NEVER points at the floor.

    Unlike grasp's ``look_at_object`` (which floors the tilt ~25° DOWN for GraspNet, the
    wrong way for a standing/seated person), this tilts toward an assumed torso height
    (``RESTAURANT_PERSON_TORSO_Z_M``) and caps the DOWNWARD tilt at
    ``RESTAURANT_PERSON_MAX_DOWN_TILT_RAD`` (and the servo's up limit), so a near customer
    can't drive the camera into the table. With no camera pose / no point it just holds the
    level person-look tilt. Best-effort; disables auto-tilt so the aim sticks."""
    head = ctx.walkie.robot.head
    tilt = _person_look_tilt()  # level fallback
    if world_xy is not None:
        try:
            from interfaces.devices.camera import camera_pose  # lazy: pulls cv2/numpy
            cam = camera_pose(ctx.walkie)
            if cam is not None:
                torso_z = _f("RESTAURANT_PERSON_TORSO_Z_M", "1.0")
                horiz = math.hypot(world_xy[0] - cam.t[0], world_xy[1] - cam.t[1]) or 1e-3
                raw = math.atan2(cam.t[2] - torso_z, horiz)  # + = down
                down_cap = _f("RESTAURANT_PERSON_MAX_DOWN_TILT_RAD", "0.2")
                tilt = max(HEAD_TILT_MIN, min(down_cap, raw))
        except Exception as exc:  # noqa: BLE001 — never break the approach on a head fault
            print(f"[restaurant.skills] _look_at_person: camera pose unavailable ({exc})")
    try:
        head.set_auto_tilt(False)
        head.tilt(tilt)
    except Exception as exc:  # noqa: BLE001
        print(f"[restaurant.skills] _look_at_person: head tilt failed ({exc})")
    if settle > 0:
        time.sleep(settle)


def _greeting_from_caption(ctx: TaskContext, caption: str | None) -> str:
    """LLM call-out from an appearance caption: name a feature + ask for the order.

    Falls back to the generic greeting when there's no caption or the model errors."""
    if not caption:
        return prompts.GREET_CUSTOMER
    try:
        from langchain.messages import HumanMessage, SystemMessage
        reply = ctx.model.invoke([
            SystemMessage(content=prompts.GREETING_INSTRUCTIONS),
            HumanMessage(content=caption),
        ])
        text = str(getattr(reply, "content", "") or "").strip()
        return text or prompts.GREET_CUSTOMER
    except Exception as exc:  # noqa: BLE001
        print(f"[restaurant.skills] greeting generation failed ({exc})")
        return prompts.GREET_CUSTOMER


def _best_caption(captions: list[str]) -> str | None:
    """Pick the most descriptive caption (longest non-empty) from the close-up rounds."""
    usable = [c for c in captions if c and c.strip()]
    return max(usable, key=len) if usable else None


def _detect_caller_near(ctx: TaskContext, target: tuple[float, float] | None,
                        bearing: float, *, radius_m: float):
    """One head-aimed snapshot: return (Caller|None, snap, bbox) for the SAME customer.

    Matches by map distance to *target* when a depth fix is available, else by absolute
    bearing (within twice the dedup bucket) — so a far, depth-less customer is still
    tracked by heading. Returns the snapshot + the matched person's xyxy box (for the
    appearance crop). Best-effort; (None, snap, None) when nobody matches."""
    _look_at_person(ctx, target, settle=_f("RESTAURANT_PERSON_LOOK_SETTLE_SEC", "0.4"))
    snap = ctx.snapshot()
    if snap is None:
        return None, None, None
    try:
        persons = ctx.walkieAI.image.estimate_poses(snap.img)
    except Exception as exc:  # noqa: BLE001
        print(f"[restaurant.skills] live re-detect pose estimation failed ({exc})")
        return None, snap, None
    rp = getattr(snap, "robot_pose", None)
    kp_conf = _f("RESTAURANT_KP_CONF", "0.3")
    bear_tol = math.radians(_f("RESTAURANT_HEADING_DEDUP_DEG", "8")) * 2.0
    best = None  # (score, Caller, box)
    for p in persons:
        xyxy = _cxcywh_to_xyxy(p.bbox)
        wxy = _person_world_xy(snap, p)
        if target is not None and wxy is not None:
            d = math.hypot(wxy[0] - target[0], wxy[1] - target[1])
            if d > radius_m:
                continue
            score = -d
            b = (math.atan2(wxy[1] - rp["y"], wxy[0] - rp["x"])
                 if rp else _bearing_from_pixel_u(_torso_center_u(p, kp_conf=kp_conf), snap))
        else:  # no depth on either side → match by bearing
            b = _bearing_from_pixel_u(_torso_center_u(p, kp_conf=kp_conf), snap)
            if b is None or abs(_wrap(b - bearing)) > bear_tol:
                continue
            score = -abs(_wrap(b - bearing))
        cand = Caller(wxy, b if b is not None else bearing, xyxy, p.confidence or 0.0)
        if best is None or score > best[0]:
            best = (score, cand, xyxy)
    if best is None:
        return None, snap, None
    return best[1], snap, best[2]


def _scan_sweep_gotostep(ctx: TaskContext, center: float, arc: float, process) -> None:
    """Rotate across *arc* in discrete in-place ``go_to`` steps; call ``process(snap)`` at
    each STATIONARY stop. CENTRE-OUT (straight ahead first; ties: negative before positive)
    so an approach-on-first sweep reacts to a customer dead ahead first. Stops early when
    ``process`` returns True. This is the reliable default — ``go_to`` always moves the base.
    """
    step = max(5.0, _f("RESTAURANT_SCAN_STEP_DEG", "30"))
    settle = _f("RESTAURANT_SCAN_SETTLE_SEC", "0.8")
    half = arc / 2.0
    offsets: list[float] = []
    o = -half
    while o <= half + 1e-6:
        offsets.append(o)
        o += step
    offsets.sort(key=lambda a: (abs(a), a))
    print(f"[restaurant.skills] live_scan: {arc:.0f}deg sweep in {len(offsets)} go_to steps")
    for off in offsets:
        ctx.rotate_to(center + math.radians(off))  # blocking in-place rotation (go_to)
        _aim_head_for_people(ctx)  # rotate_to re-enables auto-tilt; re-raise before capture
        if settle > 0:
            time.sleep(settle)
        if process(ctx.snapshot()):
            return


def _scan_sweep_cmdvel(ctx: TaskContext, center: float, arc: float, process) -> None:
    """Rotate across *arc* in ONE continuous turn at ``RESTAURANT_SCAN_RATE_DPS`` deg/s via
    ``nav.set_velocity``; call ``process(snap)`` ~every ``RESTAURANT_SCAN_DETECT_SEC`` DURING
    the spin.

    Uses the SDK's ``set_velocity`` (open-loop teleop drive) — it publishes the correct
    ``geometry_msgs/msg/TwistStamped`` to cmd_vel. A raw plain-Twist publish is malformed
    for this base (``cmd_vel_type`` is TwistStamped) and moves it 0 m. WARNING: bearings
    carry the snapshot's capture-latency skew at speed. Falls back to the go_to-step sweep
    if ``set_velocity`` isn't available. Guarantees a zero-velocity stop in ``finally``.
    """
    nav = getattr(ctx.walkie, "nav", None)
    if nav is None or not hasattr(nav, "set_velocity"):
        print("[restaurant.skills] live_scan: nav.set_velocity unavailable; using go_to steps")
        return _scan_sweep_gotostep(ctx, center, arc, process)

    rate = math.radians(max(1.0, _f("RESTAURANT_SCAN_RATE_DPS", "12")))
    hz = max(1.0, _f("RESTAURANT_SCAN_PUBLISH_HZ", "15"))
    dt = 1.0 / hz
    detect_sec = _f("RESTAURANT_SCAN_DETECT_SEC", "1.0")
    half = math.radians(arc) / 2.0

    # Pre-rotate (blocking) to the arc start, then sweep one continuous turn to the far end.
    ctx.rotate_to(center - half)
    _aim_head_for_people(ctx)
    target_sweep = math.radians(arc)
    deadline = time.monotonic() + target_sweep / rate + 3.0
    print(f"[restaurant.skills] live_scan: {arc:.0f}deg cmd_vel sweep @ {math.degrees(rate):.0f}deg/s "
          "(nav.set_velocity)")
    prev_h: float | None = None
    swept = 0.0
    last_detect = 0.0
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            cur = _robot_pose(ctx)
            if cur is None:
                print("[restaurant.skills] live_scan: lost odom mid-sweep; stopping")
                break
            if prev_h is not None:
                swept += abs(_wrap(cur["heading"] - prev_h))  # angle actually swept (wrap-safe)
            prev_h = cur["heading"]
            if swept >= target_sweep:
                break
            nav.set_velocity(0.0, 0.0, rate)  # yaw spin (TwistStamped via the SDK)
            if now - last_detect < detect_sec:
                time.sleep(dt)
                continue
            last_detect = now
            if process(ctx.snapshot()):  # base coasts through the round-trip / announce
                return
    finally:
        for _ in range(3):  # guaranteed stop
            try:
                nav.set_velocity(0.0, 0.0, 0.0)
            except Exception:  # noqa: BLE001
                break
        try:
            nav.stop()
        except Exception:  # noqa: BLE001
            pass


def live_scan_for_caller(ctx: TaskContext,
                         blocked: list[tuple[float, float]] | None = None,
                         radius_m: float | None = None,
                         *, arc_deg: float | None = None,
                         center: float | None = None) -> Caller | None:
    """Sweep the base across the dining area, pose-detecting waving customers.

    Two rotation modes (``RESTAURANT_SCAN_ROTATE_MODE``) share the SAME bearing-first
    detection: each waver's absolute BEARING is the torso-centre pixel + the snapshot's
    capture heading (depth-independent); callers dedup by ``RESTAURANT_HEADING_DEDUP_DEG``;
    ``world_xy`` is filled only when depth was valid (far caller = bearing-only, the live
    approach refines it); the running count is announced; ``RESTAURANT_APPROACH_FIRST``
    returns the first waver vs finishing the arc and picking the nearest.

    - ``gotostep`` (default): discrete in-place ``nav.go_to`` (``ctx.rotate_to``) steps of
      ``RESTAURANT_SCAN_STEP_DEG`` across ``RESTAURANT_SCAN_ARC_DEG``, CENTRE-OUT, detecting
      at each STATIONARY stop. Reliable (go_to always moves the base) and skew-free.
    - ``cmdvel``: ONE continuous in-place rotation at ``RESTAURANT_SCAN_RATE_DPS`` deg/s,
      publishing a Twist to ``cmd_vel`` (the creep channel), detecting ~every
      ``RESTAURANT_SCAN_DETECT_SEC`` DURING the turn. The original "spin while detecting"
      design — but on a base that IGNORES raw cmd_vel it just stands still (verify with
      ``uv run python -m manual_tests.test_base_creep --raw-only``), and bearings carry the
      capture-latency skew. Falls back to ``gotostep`` if the cmd_vel channel won't open.

    Returns None on no odom fix / none found.
    """
    if blocked is None:
        blocked = []
    if radius_m is None:
        radius_m = _f("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    pose = _robot_pose(ctx)
    if pose is None:
        print("[restaurant.skills] live_scan: no odometry fix; skipping sweep")
        return None

    # The sweep centre defaults to the dining heading (bar anchor) — correct when scanning
    # from the bar. A caller (e.g. lost-recovery mid-approach, where the robot has driven
    # into the room and turned toward the customer) can override it with the robot's CURRENT
    # heading so the re-scan looks where the customer actually was, not back at the bar.
    if center is None:
        bar = ctx.data.get("bar_anchor")
        center = bar["heading"] if bar else pose["heading"]
    arc = arc_deg if arc_deg is not None else _f("RESTAURANT_SCAN_ARC_DEG", "180")
    dedup_deg = _f("RESTAURANT_HEADING_DEDUP_DEG", "8")
    kp_conf = _f("RESTAURANT_KP_CONF", "0.3")
    approach_first = _b("RESTAURANT_APPROACH_FIRST", "1")

    callers: list[Caller] = []
    stats: dict = {"selected": None, "persons": 0, "calling": 0}

    def _process(snap) -> bool:
        """Detect wavers in *snap*, append fresh callers + announce; True = stop the sweep.

        Shared by both rotation modes — the only difference between them is HOW the base
        turns and WHEN this runs (at each stop vs ~1 Hz during a continuous spin)."""
        if snap is None:
            return False
        try:
            persons = ctx.walkieAI.image.estimate_poses(snap.img)
        except Exception as exc:  # noqa: BLE001
            print(f"[restaurant.skills] live_scan pose estimation failed ({exc})")
            return False
        stats["persons"] += len(persons)
        for p in persons:
            if not is_calling(p):
                continue
            stats["calling"] += 1
            bearing = _bearing_from_pixel_u(_torso_center_u(p, kp_conf=kp_conf), snap)
            if bearing is None:
                continue
            if _bearing_is_dup(bearing, [c.bearing for c in callers], dedup_deg):
                continue
            wxy = _person_world_xy(snap, p)
            if wxy is not None and any(
                    math.hypot(wxy[0] - bx, wxy[1] - by) <= radius_m for bx, by in blocked):
                continue  # already handled / given up on this spot
            caller = Caller(wxy, bearing, _cxcywh_to_xyxy(p.bbox), p.confidence or 0.0)
            callers.append(caller)
            n = len(callers)
            ctx.say(prompts.SEE_N_CUSTOMERS.format(n=n, plural="" if n == 1 else "s"))
            if approach_first:
                stats["selected"] = caller
                return True
        return False

    mode = os.getenv("RESTAURANT_SCAN_ROTATE_MODE", "gotostep").strip().lower()
    try:
        if mode == "cmdvel":
            _scan_sweep_cmdvel(ctx, center, arc, _process)
        else:
            _scan_sweep_gotostep(ctx, center, arc, _process)
    except Exception as exc:  # noqa: BLE001 — never let the scan raise mid-run
        print(f"[restaurant.skills] live_scan: error ({exc}); stopping")
    finally:
        ctx.walkie.robot.head.set_auto_tilt(True)
    print(f"[restaurant.skills] live_scan ({mode}): {len(callers)} caller(s) "
          f"(saw {stats['persons']} person-detection(s), {stats['calling']} with a raised hand)")

    selected = stats["selected"]
    if selected is not None:
        return selected
    if not callers:
        print("[restaurant.skills] live_scan: no callers across the sweep")
        ctx.rotate_to(center)  # leave the base back at the dining centre
        return None
    # Finish-sweep selection: nearest by map point (those with depth), else most central.
    rx, ry = pose["x"], pose["y"]
    placed = [c for c in callers if c.world_xy is not None]
    if placed:
        return min(placed, key=lambda c: math.hypot(c.world_xy[0] - rx, c.world_xy[1] - ry))
    return min(callers, key=lambda c: abs(_wrap(c.bearing - center)))


def _lost_recover(ctx: TaskContext, caller: Caller) -> Caller | None:
    """Re-find a customer the approach lost: a narrow live re-scan for any waver.

    Centres the re-scan on the robot's CURRENT heading (it has driven into the room while
    facing the customer), NOT the bar-anchored dining heading — the lost customer is in
    front of the robot now, not back toward the bar."""
    arc = _f("RESTAURANT_RESCAN_ARC_DEG", "60")
    cur = _robot_pose(ctx)
    center = cur["heading"] if cur else caller.bearing
    return live_scan_for_caller(ctx, [], _f("RESTAURANT_HANDLED_RADIUS_M", "0.6"),
                                arc_deg=arc, center=center)


def live_approach_caller(ctx: TaskContext, caller: Caller
                         ) -> tuple[bool, str | None, tuple[float, float] | None]:
    """Drive to a (possibly bearing-only) caller in one non-blocking approach.

    Models grasp's ``approach_object``: a single non-blocking ``go_to`` with a stand-off
    and ``align_method="face_target"`` (the base faces the customer), re-aiming the HEAD
    each tick (never a competing per-tick base rotate). On top of it:

    - **Close/far** by depth validity — a valid fix within ``RESTAURANT_DEPTH_RELIABLE_M``
      drives straight to the point; otherwise it drives a ``RESTAURANT_FAR_STEP_M`` step
      along the bearing ray and re-detects to refine (cancel + re-issue only when the
      refined point moved past ``RESTAURANT_REFINE_MIN_DELTA_M``).
    - **Velocity stall**: base under ``RESTAURANT_APPROACH_LOW_SPEED_MPS`` for
      ``RESTAURANT_APPROACH_LOW_SPEED_SEC`` → cancel; if the customer is still in view,
      treat the spot as reached (serve from here); else recover.
    - **Lost**: customer out of view for ``RESTAURANT_LOST_SEC`` → cancel and re-scan; if
      no waver is re-found, announce the loss, return to the bar, and report failure.
    - **Appearance**: up to ``RESTAURANT_CLOSE_APPEARANCE_ROUNDS`` clothing captions are
      captured up close (interleaved in the loop — no threads) and the best is returned so
      the order ask can be an appearance-aware call-out.

    Returns ``(reached, best_caption, final_world_xy)``. On success ``final_world_xy`` is
    a real refined point; on failure it is ``None``.
    """
    standoff = _f("RESTAURANT_STANDOFF_M", "0.8")
    reliable = _f("RESTAURANT_DEPTH_RELIABLE_M", "3.5")
    far_step = _f("RESTAURANT_FAR_STEP_M", str(reliable))
    refine_delta = _f("RESTAURANT_REFINE_MIN_DELTA_M", "0.4")
    low_mps = _f("RESTAURANT_APPROACH_LOW_SPEED_MPS", "0.08")
    low_sec = _f("RESTAURANT_APPROACH_LOW_SPEED_SEC", "5.0")
    lost_sec = _f("RESTAURANT_LOST_SEC", "5.0")
    detect_sec = _f("RESTAURANT_APPROACH_DETECT_SEC", "1.0")
    timeout = _f("RESTAURANT_APPROACH_TIMEOUT_SEC", "45")
    rounds = int(_f("RESTAURANT_CLOSE_APPEARANCE_ROUNDS", "3"))
    reacquire_r = _f("RESTAURANT_REACQUIRE_RADIUS_M", "1.5")
    goal_tol = _f("WALKIE_NAV_GOAL_TOLERANCE_M", "0.0")

    pose = _robot_pose(ctx)
    if pose is None:
        print("[restaurant.skills] live_approach: no odom fix; cannot navigate")
        return False, None, None

    ctx.say(prompts.FOUND_CUSTOMER)
    try:
        ctx.walkie.robot.head.set_auto_tilt(False)
    except Exception:  # noqa: BLE001
        pass

    target = caller.world_xy if caller.world_xy is not None else _ray_point(pose, caller.bearing, far_step)
    captions: list[str] = []

    def _issue(tx: float, ty: float) -> bool:
        try:
            ctx.walkie.nav.go_to(x=tx, y=ty, blocking=False, standoff=standoff,
                                 align_method="face_target")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[restaurant.skills] live_approach: nav raised ({exc})")
            return False

    def _cancel() -> None:
        try:
            ctx.walkie.nav.cancel()
        except Exception:  # noqa: BLE001
            pass

    _issue(*target)
    now0 = time.monotonic()
    last_seen = now0
    low_start: float | None = None
    last_detect = 0.0
    deadline = now0 + timeout
    reached = False

    while time.monotonic() < deadline:
        now = time.monotonic()
        # --- cheap velocity poll (stall timer) ---
        try:
            v = ctx.walkie.status.get_velocity()
        except Exception:  # noqa: BLE001
            v = None
        if v is not None:  # None = telemetry not up yet → don't trip the timer
            if abs(v.get("linear", 0.0)) < low_mps:
                low_start = low_start or now
            else:
                low_start = None
        try:
            navigating = ctx.walkie.nav.is_navigating
        except Exception:  # noqa: BLE001
            navigating = False

        # --- ~1 Hz re-detect: refine target, hold head, grab appearance ---
        if now - last_detect >= detect_sec:
            last_detect = now
            fresh, snap, box = _detect_caller_near(ctx, target, caller.bearing,
                                                   radius_m=max(reacquire_r, 0.6 * far_step))
            if fresh is not None:
                last_seen = now
                if fresh.world_xy is not None:
                    moved = math.hypot(target[0] - fresh.world_xy[0], target[1] - fresh.world_xy[1])
                    target = fresh.world_xy
                    if moved > refine_delta:  # refined enough to be worth re-issuing
                        _cancel()
                        _issue(*target)
                # appearance up close (interleaved; keep the best of N rounds)
                cur = _robot_pose(ctx)
                if (box is not None and snap is not None and fresh.world_xy is not None
                        and cur is not None and len(captions) < rounds
                        and _caller_is_close(fresh.world_xy, cur, reliable)):
                    cap = describe_customer(ctx, snap, box)
                    if cap:
                        captions.append(cap)

        # --- stall: low speed sustained ---
        if low_start is not None and (now - low_start) >= low_sec:
            print(f"[restaurant.skills] live_approach: base under {low_mps:.2f}m/s for "
                  f"{low_sec:.0f}s — cancelling and checking the customer is in view")
            _cancel()
            if (now - last_seen) <= lost_sec:
                print("[restaurant.skills] live_approach: customer in view — reachable, "
                      "serving from here")
                reached = True
                break
            recovered = _lost_recover(ctx, caller)
            if recovered is None:
                ctx.say(prompts.CUSTOMER_LOST)
                return_to_bar(ctx, face_counter=False)
                return False, None, None
            caller = recovered
            target = recovered.world_xy or _ray_point(_robot_pose(ctx) or pose, recovered.bearing, far_step)
            last_seen = time.monotonic()
            low_start = None
            _issue(*target)
            continue

        # --- lost: out of view too long ---
        if (now - last_seen) >= lost_sec:
            print(f"[restaurant.skills] live_approach: customer out of view for "
                  f"{lost_sec:.0f}s — cancelling and re-scanning")
            _cancel()
            recovered = _lost_recover(ctx, caller)
            if recovered is None:
                ctx.say(prompts.CUSTOMER_LOST)
                return_to_bar(ctx, face_counter=False)
                return False, None, None
            caller = recovered
            target = recovered.world_xy or _ray_point(_robot_pose(ctx) or pose, recovered.bearing, far_step)
            last_seen = time.monotonic()
            low_start = None
            _issue(*target)
            continue

        # --- arrived? (nav settled) ---
        if not navigating:
            after = _robot_pose(ctx) or pose
            end_d = math.hypot(target[0] - after["x"], target[1] - after["y"])
            if end_d <= standoff + max(goal_tol, 0.4):
                reached = True
                break
            # Nav stopped short (transient abort / unreachable standoff). Re-issue once
            # more toward the best target; the loop's stall/lost timers bound the retries.
            _issue(*target)

        time.sleep(0.1)

    if not reached:
        print("[restaurant.skills] live_approach: timed out before reaching the customer")
        _cancel()
        # Took too long but if the customer is in view, serve from here rather than abandon.
        if (time.monotonic() - last_seen) > lost_sec:
            ctx.say(prompts.CUSTOMER_LOST)
            return_to_bar(ctx, face_counter=False)
            return False, None, None

    # Final facing + a fresh fix so the recorded seat is accurate.
    face_person(ctx, target)
    fresh = find_person_near(ctx, target, radius_m=reacquire_r)
    final_xy = fresh.world_xy if fresh is not None else target
    best_caption = _best_caption(captions)
    if best_caption is None:  # never got a close caption → one best-effort capture now
        best_caption = capture_appearance(ctx, final_xy)
    try:
        ctx.walkie.robot.head.set_auto_tilt(True)
    except Exception:  # noqa: BLE001
        pass
    return True, best_caption, final_xy


# ---------------------------------------------------------------------------
# Interaction (order taking is real; relay speaks the order)
# ---------------------------------------------------------------------------
def describe_customer(ctx: TaskContext, snap, bbox_xyxy: BBox) -> str | None:
    """Caption a customer's appearance from their crop, to re-identify on return."""
    img = getattr(snap, "img", None)
    if img is None:
        return None
    x1, y1, x2, y2 = (int(v) for v in bbox_xyxy)
    m = 15  # px padding so clothing isn't clipped
    crop = img.crop((max(0, x1 - m), max(0, y1 - m),
                     min(img.width, x2 + m), min(img.height, y2 + m)))
    try:
        return ctx.walkieAI.image.caption(crop, prompt=prompts.CUSTOMER_APPEARANCE_PROMPT)
    except Exception as exc:
        print(f"[restaurant.skills] appearance caption failed ({exc})")
        return None


def capture_appearance(ctx: TaskContext, world_xy: tuple[float, float]) -> str | None:
    """Snapshot, find the person nearest *world_xy*, and caption their appearance.

    Detection + caption share one snapshot so the bbox lines up with the image.
    Best-effort; None if no person/geometry. Stored on the Order for re-ID/logging.
    """
    _aim_for_person_capture(ctx)  # raise the head before framing the person
    snap = ctx.snapshot()
    if snap is None or not getattr(snap, "has_geometry", False):
        return None
    try:
        persons = ctx.walkieAI.image.estimate_poses(snap.img)
    except Exception as exc:
        print(f"[restaurant.skills] appearance pose estimation failed ({exc})")
        return None
    best_box = None
    best_d = float("inf")
    for p in persons:
        xyxy = _cxcywh_to_xyxy(p.bbox)
        wxy = _person_world_xy(snap, p)
        if wxy is None:
            continue
        d = math.hypot(wxy[0] - world_xy[0], wxy[1] - world_xy[1])
        if d < best_d:
            best_box, best_d = xyxy, d
    if best_box is None:
        return None
    caption = describe_customer(ctx, snap, best_box)
    if caption:
        _remember_appearance(ctx, world_xy, caption, snap, best_box)
    return caption


def _cos_sim(a, b) -> float:
    """Cosine similarity of two vectors (norm-guarded)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-8)


def _remember_appearance(ctx: TaskContext, world_xy, caption: str, snap=None, bbox=None) -> None:
    """Embed + store the customer's appearance in the unified people store (attire-only).

    Best-effort: no-op on a ctx without a people-enabled world. The stable id is the
    rounded map point, so re-captures of the same seat update one record. Stores the
    CLIP-text embedding of the caption (for semantic re-ID) and, when a crop is given,
    the OSNet attire vector — the same store HRI uses for guests.
    """
    world = getattr(ctx, "world", None)
    if world is None:
        return
    try:
        cap_emb = ctx.walkieAI.image.embed_text(caption)
    except Exception:
        cap_emb = None
    app_emb = None
    if snap is not None and bbox is not None:
        img = getattr(snap, "img", None)
        if img is not None:
            try:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                app_emb = ctx.walkieAI.image.appearance(
                    img.crop((max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)))
                )
            except Exception:
                app_emb = None
    pid = f"customer-{round(float(world_xy[0]), 1)}-{round(float(world_xy[1]), 1)}"
    try:
        world.enroll_person(
            name="", drink="", face_embedding=[], person_id=pid,
            appearance_caption=caption, appearance_caption_embedding=cap_emb,
            app_embedding=app_emb,
            last_seen_pose=(float(world_xy[0]), float(world_xy[1]), 0.0),
            pose_label="seated",
        )
    except Exception as exc:  # noqa: BLE001 — re-ID memory must never break order-taking
        print(f"[restaurant.skills] appearance enroll skipped ({exc})")


def _match_appearance_candidate(ctx: TaskContext, snap, stored: str, cands):
    """Pick the in-range candidate whose attire caption best matches *stored*.

    Semantic CLIP-text similarity when an embed server is reachable (the unified re-ID
    signal — same embedding the people store uses), else the lexical word-overlap
    fallback. Returns the chosen candidate tuple, or None when nothing is a confident
    match (the caller then falls through to the nearest / calling pick)."""
    caps = [describe_customer(ctx, snap, c[2]) or "" for c in cands]
    try:
        embed = ctx.walkieAI.image.embed_text
        sv = embed(stored)
        if sv:
            min_sim = _f("RESTAURANT_APPEARANCE_MATCH_MIN", "0.6")
            best = None  # (sim, nearer_key, cand)
            for cap, c in zip(caps, cands):
                cv = embed(cap) if cap else None
                sim = _cos_sim(sv, cv) if cv else -1.0
                key = (sim, -c[0])
                if best is None or key > best[0]:
                    best = (key, c)
            if best is not None and best[0][0] >= min_sim:
                return best[1]
            return None
    except Exception:  # noqa: BLE001 — degrade to the lexical fallback below
        pass
    scored = [(_appearance_overlap(stored, cap), -c[0], c) for cap, c in zip(caps, cands)]
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return scored[0][2] if scored[0][0] > 0 else None


_APPEARANCE_STOPWORDS = {
    "a", "an", "the", "with", "and", "is", "are", "wearing", "wears", "person",
    "customer", "seated", "sitting", "this", "who", "has", "have", "their", "in",
    "on", "of", "short", "sentence", "appears", "to", "be", "they", "them", "looks",
    "that", "very", "or", "man", "woman", "guy", "lady",
}


def _appearance_overlap(stored: str, candidate: str) -> int:
    """Count of shared distinctive words between two appearance captions (stopwords
    removed). A cheap lexical re-ID tiebreaker — distinguishing 'red shirt, glasses'
    from 'blue hoodie' is enough to pick the right neighbour without a CLIP round-trip."""
    def words(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", (s or "").lower())
                if w not in _APPEARANCE_STOPWORDS and len(w) > 2}
    return len(words(stored) & words(candidate))


def find_person_near(ctx: TaskContext, world_xy: tuple[float, float], *,
                     radius_m: float | None = None,
                     prefer_calling: bool = False,
                     appearance: str | None = None) -> Caller | None:
    """Re-acquire a person near a remembered map point (anti-drift, design §5.1).

    One forward snapshot (head raised first): detect people, lift each, return the one
    closest to *world_xy* within *radius_m*. None if nobody qualifies. Used to re-find
    the customer (before serving) or the barman (at the bar) rather than trusting a
    stored coordinate after minutes in a moving room.

    Selection among in-range candidates: with *appearance* (the caption captured when
    the order was taken) AND more than one candidate, pick the best lexical caption
    match — re-identifying the SAME customer instead of a neighbour at an adjacent table
    (~1 m away). Else with *prefer_calling* a still-waving person wins. Otherwise the
    nearest. (The customer often lowers their hand once the robot is clearly coming, so
    calling is only a soft preference, not a filter.)
    """
    if radius_m is None:
        radius_m = _f("RESTAURANT_REACQUIRE_RADIUS_M", "1.5")
    pose = _robot_pose(ctx)
    _aim_for_person_capture(ctx)  # raise the head before re-acquiring (never look down)
    snap = ctx.snapshot()
    if snap is None or not getattr(snap, "has_geometry", False):
        return None
    try:
        persons = ctx.walkieAI.image.estimate_poses(snap.img)
    except Exception as exc:
        print(f"[restaurant.skills] re-acquire pose estimation failed ({exc})")
        return None
    cands: list[tuple[float, tuple[float, float], BBox, float, bool]] = []
    for p in persons:
        xyxy = _cxcywh_to_xyxy(p.bbox)
        wxy = _person_world_xy(snap, p)
        if wxy is None:
            continue
        d = math.hypot(wxy[0] - world_xy[0], wxy[1] - world_xy[1])
        if d <= radius_m:
            cands.append((d, wxy, xyxy, p.confidence or 0.0, is_calling(p)))
    if not cands:
        return None
    # Appearance re-ID first (only worth the per-candidate caption calls when there's an
    # actual ambiguity to resolve): pick the candidate whose attire best matches the
    # stored appearance — semantic CLIP-text similarity (unified re-ID), lexical fallback.
    # None on no confident match → fall through to the spatial/calling pick.
    if appearance and len(cands) > 1:
        chosen = _match_appearance_candidate(ctx, snap, appearance, cands)
        if chosen is not None:
            d, wxy, xyxy, conf, _calling = chosen
            bearing = math.atan2(wxy[1] - pose["y"], wxy[0] - pose["x"]) if pose else 0.0
            print(f"[restaurant.skills] re-acquire: appearance-matched customer at "
                  f"({wxy[0]:.2f},{wxy[1]:.2f})")
            return Caller(wxy, bearing, xyxy, conf)
    pool = [c for c in cands if c[4]] if prefer_calling else []
    pool = pool or cands
    d, wxy, xyxy, conf, _ = min(pool, key=lambda c: c[0])
    bearing = math.atan2(wxy[1] - pose["y"], wxy[0] - pose["x"]) if pose else 0.0
    return Caller(wxy, bearing, xyxy, conf)


_NEGATIVE_CUES = {"no", "nope", "nah", "not", "wrong", "incorrect", "isn't", "isnt"}


def _said_no(text: str) -> bool:
    """Loose 'the customer rejected the confirmation' check (biased to accept).

    Only an explicit negative word flips this true; an empty/garbled reply reads
    as acceptance, so venue noise never silently discards a correctly-heard order.
    """
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & _NEGATIVE_CUES)


def _capture_order(ctx: TaskContext, recenter, first_prompt: str) -> list[str]:
    """Ask -> listen -> LLM-parse, re-asking the SAME customer up to RESTAURANT_ORDER_RETRIES
    extra times on an empty/garbled parse (then []).

    Without this loop a single unclear reply returned [] and ServeCustomers gave up on the
    customer and went looking for a NEW one — the robot should ask again first (observed
    on-robot). A poor accent often makes STT mis-transcribe a real, spoken order, so we
    re-ask persistently (default 9 extra times = 10 listens) rather than abandon a customer
    who IS talking. The first prompt greets/asks; later tries nudge them to repeat (and,
    after the first re-ask, to slow down — see :data:`prompts.ASK_REPEAT_SLOW`).

    Bounding the worst case is COUNT-based, not duration-based (shortening the listen timeout
    would cut off slow/accented speakers — the opposite of the goal). The empty-vs-garbled
    answer is the discriminator: a non-empty-but-unparseable reply is the accent case and gets
    the full retry budget, but a string of EMPTY replies (true silence) means the seat is
    likely empty / the customer disengaged, so we bail after RESTAURANT_ORDER_MAX_SILENT
    consecutive empties instead of burning all 10 listen timeouts on dead air.
    """
    retries = int(os.getenv("RESTAURANT_ORDER_RETRIES", "9"))
    max_silent = int(os.getenv("RESTAURANT_ORDER_MAX_SILENT", "3"))
    silent = 0
    for attempt in range(retries + 1):
        recenter()
        # Attempt 0 greets/asks; the first miss nudges a repeat; later misses ask them to
        # slow down (the accent-garble case the persistent re-ask is here to recover).
        if attempt == 0:
            prompt = first_prompt
        elif attempt == 1:
            prompt = prompts.ASK_REPEAT
        else:
            prompt = prompts.ASK_REPEAT_SLOW
        answer = ctx.ask(prompt, retries=0)
        parsed = ctx.extract(prompts.Order, prompts.EXTRACT_ORDER_INSTRUCTIONS, answer or "")
        if parsed and parsed.items:
            return parsed.items
        if answer:
            silent = 0  # heard something (accent garble) — keep spending the full budget
        else:
            silent += 1
            if silent >= max_silent:
                break  # repeated dead air -> customer absent, stop wasting timeouts
    return []


def take_order(ctx: TaskContext, world_xy: tuple[float, float] | None = None, *,
               first_prompt: str | None = None) -> list[str]:
    """Greet the customer, capture and confirm their order. Real dialogue today.

    Gaze (rulebook-scored): if *world_xy* is given, the robot re-faces the customer before
    each utterance — MVP "look at the person" without a continuous-tracking thread (design
    §5.2). Capture re-asks the SAME customer on an unclear reply (see :func:`_capture_order`)
    instead of dropping them. Confirmation LISTENS: an explicit "no" re-takes the order (up
    to RESTAURANT_CONFIRM_RETRIES times — that step scores 2×160), while a silent/garbled
    reply counts as agreement so venue noise can't drop a good order. Returns [] only after
    the re-asks genuinely fail.

    *first_prompt* overrides the opening line (default :data:`prompts.GREET_CUSTOMER`).
    The live approach passes its appearance-aware call-out ("you in the red jacket — what
    would you like to order?") here so the order ask isn't a second, generic greeting.
    """
    def recenter():
        if world_xy is not None:
            face_person(ctx, world_xy)

    items = _capture_order(ctx, recenter, first_prompt or prompts.GREET_CUSTOMER)
    print(f"[restaurant.skills] captured items: {items}")
    if not items:
        return []  # un-parseable even after re-asking; the caller decides what to do next

    for _ in range(int(os.getenv("RESTAURANT_CONFIRM_RETRIES", "1")) + 1):
        recenter()
        reply = ctx.ask(prompts.CONFIRM_ORDER.format(items=", ".join(items)), retries=0)
        if not (reply and _said_no(reply)):
            break  # yes / silence / garble -> accept
        new_items = _capture_order(ctx, recenter, prompts.ASK_REPEAT)
        if new_items:
            items = new_items  # else keep the prior best-effort parse and proceed
    recenter()
    ctx.say(prompts.ORDER_TAKEN)
    return items


def return_to_bar(ctx: TaskContext, *, face_counter: bool = True) -> bool:
    """Drive to the bar anchor, turn to face the bar, and optionally find the barman.

    go_to gets us near the remembered anchor. We then TURN to face the counter/bar
    side — the anchor heading points at the DINERS (the robot started facing them),
    but the counter/kitchen is off to the side, and we must face it so the barman can
    reach the robot's tray. ``RESTAURANT_COUNTER_REL_DEG`` is that turn (0 = none); it
    runs regardless of ``RESTAURANT_BAR_REACQUIRE`` — facing the bar is needed even
    when we don't visually search for the barman. With ``RESTAURANT_BAR_REACQUIRE`` on
    we additionally re-acquire the barman from the live camera and face them precisely
    (design §5.1), degrading to just facing the bar if none is seen.

    ``face_counter=False`` parks at the anchor WITHOUT the counter turn: the caller is
    about to do something that faces the diners anyway (a scan for the next customer),
    so turning to the counter here would only be whipped back a moment later — the idle
    back-and-forth spin the robot showed between serving and looking for the next caller.
    Pass True (default) only when the NEXT action at the bar is a load/pick (it needs to
    face the counter).
    """
    bar = ctx.data.get("bar_anchor")
    if not bar:
        return False
    ok = ctx.goto(bar["x"], bar["y"], bar["heading"])
    # Turn to face the counter/bar side. +90 = counter on the left, -90 = on the right,
    # 180 = behind the diner-facing start; flip the sign if it turns the wrong way.
    rel = math.radians(_f("RESTAURANT_COUNTER_REL_DEG", "0"))
    if face_counter and rel:
        ctx.rotate_to(bar["heading"] + rel)
    # Optionally also re-acquire the barman visually and face them precisely.
    if not face_counter or not _b("RESTAURANT_BAR_REACQUIRE", "1"):
        return ok  # parked at the bar; skip the vision barman search
    barman = find_person_near(ctx, (bar["x"], bar["y"]),
                              radius_m=_f("RESTAURANT_BARMAN_RADIUS_M", "2.5"))
    if barman is not None:
        face_person(ctx, barman.world_xy)
    return ok


def return_to_customer(ctx: TaskContext, world_xy: tuple[float, float], *,
                       appearance: str | None = None) -> tuple[float, float] | None:
    """Return to a customer: go near the stored point, then re-acquire them fresh.

    Returns the customer's refreshed map point (for serving) or None if they could not
    be re-found. Approaches the fresh detection to a stand-off; updates nothing in place
    (caller stores the returned point on the Order).

    Re-acquisition uses a TIGHTER radius than the approach re-acquire
    (``RESTAURANT_SERVE_REACQUIRE_RADIUS_M``, default 0.7 m < the ~1 m table spacing): we
    already drove back to a good point, so anyone beyond that is a neighbour, not our
    customer. *appearance* (captured at order time) breaks ties when two people fall
    inside the radius — re-identifying the right diner rather than serving the neighbour.
    """
    approach_to_standoff(ctx, world_xy)  # get into viewing range of the table
    radius = _f("RESTAURANT_SERVE_REACQUIRE_RADIUS_M", "0.7")
    fresh = find_person_near(ctx, world_xy, radius_m=radius, appearance=appearance)
    target = fresh.world_xy if fresh is not None else world_xy
    approach_to_standoff(ctx, target)
    return target if fresh is not None else None


def relay_to_barman(ctx: TaskContext, items: list[str]) -> bool:
    """Speak the order to the barman at the Kitchen-bar (real). Returns whether spoken."""
    if not items:
        return False
    ctx.say(prompts.GREET_BARMAN)
    ctx.say(prompts.RELAY_TO_BARMAN.format(items=", ".join(items)))
    return True


# ---------------------------------------------------------------------------
# Manipulation
# ---------------------------------------------------------------------------
# Pick + place now run through the shared grasp/place pipeline (tasks.skills.
# pick_object / place_object — GraspNet planning + depth-lifted surface placement),
# orchestrated per-item by subtasks._pick_and_serve. The old hand-rolled reach math
# (_map_to_base / _in_reach / pick_item / serve_item / collect_items / serve_order)
# was retired with it. The only gate left here: untested arm motion in a public
# venue is dangerous (contact = e-stop), so the serve loop only commands the arm
# when RESTAURANT_ARM_CALIBRATED=1 — otherwise it rehearses nav/HRI and skips pick.

def _arm_calibrated() -> bool:
    return os.getenv("RESTAURANT_ARM_CALIBRATED", "0").lower() in ("1", "true", "yes")


def transport_with_tray(ctx: TaskContext, items: list[str]) -> bool:
    """Optional bonus (rulebook extra reward 2×200): carry an order on a tray.

    PHASE 3 FAIL-SAFE STUB. A tray is the only way to carry a multi-item order in
    one trip (one gripper holds one object), so it both unlocks the bonus and
    enables true batched delivery. But it is bimanual and twice the calibration
    guesswork of a single grasp, so it is left as a logged, no-move scaffold:
    place each item on the tray, grasp the tray with both arms, carry, deliver.
    Returns False until implemented + calibrated. See docs/RESTAURANT_DESIGN.md.
    """
    print(f"[restaurant.skills] TODO transport_with_tray({items}) — bimanual tray "
          "carry not implemented (Phase 3 bonus; needs on-robot calibration)")
    return False
