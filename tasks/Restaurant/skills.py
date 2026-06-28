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


# ---------------------------------------------------------------------------
# Calling / waving detection (keypoints)
# ---------------------------------------------------------------------------
@dataclass
class Caller:
    """A customer detected calling/waving, anchored in the map for approach."""

    world_xy: tuple[float, float]  # map-frame position (lifted from the person bbox)
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
    """Base-rotation offsets (deg) covering the scan arc, centered on entry heading."""
    arc = _f("RESTAURANT_SCAN_ARC_DEG", "180")
    step = max(5.0, _f("RESTAURANT_SCAN_STEP_DEG", "30"))
    half = arc / 2.0
    offsets = []
    off = -half
    while off <= half + 1e-6:
        offsets.append(off)
        off += step
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
    ctx.walkie.robot.head.set_auto_tilt(False)
    if pose is None:
        print("[restaurant.skills] scan_for_callers: no odometry fix; skipping sweep")
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
    for off in _scan_offsets():
        ctx.rotate_to(center + math.radians(off))
        if settle > 0:
            time.sleep(settle)  # let the base + depth settle before capturing
        snap = ctx.snapshot()
        if snap is None:
            continue
        try:
            persons = ctx.walkieAI.image.estimate_poses(snap.img)
        except Exception as exc:
            print(f"[restaurant.skills] pose estimation failed ({exc})")
            continue
        for p in persons:
            if not is_calling(p):
                continue
            xyxy = _cxcywh_to_xyxy(p.bbox)
            world_xy = _person_world_xy(snap, p)
            if world_xy is None:
                continue  # can't approach what we can't place on the map
            bearing = math.atan2(world_xy[1] - ry, world_xy[0] - rx)
            callers.append(Caller(world_xy, bearing, xyxy, p.confidence or 0.0))
    ctx.rotate_to(center)  # leave the base back at the entry heading
    ctx.walkie.robot.head.set_auto_tilt(True)
    callers = _dedup_callers(callers, _f("RESTAURANT_CALLER_DEDUP_M", "0.6"))
    print(f"[restaurant.skills] scan found {len(callers)} caller(s)")
    return callers


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
            align_method="face_target", goal_tolerance=(tol if tol > 0 else None),
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
                      max_steps: int = 5) -> bool:
    """Approach a waving customer in steps, re-detecting them as we close in.

    The depth camera is only accurate to ~``RESTAURANT_DEPTH_RELIABLE_M`` (≈ 4 m), so a
    customer sitting farther is lifted to a coarse, usually-SHORT point — one drive stops
    well shy of them. We instead close the gap over several steps: drive PART of the way
    (enough to bring the customer inside the reliable depth band), re-detect the SAME
    person from there (``find_person_near`` anchored to the running estimate — never the
    globally nearest caller, so we can't hop to another table), refine the point, and
    repeat. The final step parks at the conversational stand-off. Returns True once we end
    within reach of the customer, False if we lost them / nav refused throughout.
    """
    standoff = _f("RESTAURANT_STANDOFF_M", "0.8")
    reliable = _f("RESTAURANT_DEPTH_RELIABLE_M", "3.5")
    target = world_xy
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
        approach_to_standoff(ctx, target, standoff_m=intermediate)
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
    return describe_customer(ctx, snap, best_box) if best_box is not None else None


def find_person_near(ctx: TaskContext, world_xy: tuple[float, float], *,
                     radius_m: float | None = None,
                     prefer_calling: bool = False) -> Caller | None:
    """Re-acquire a person near a remembered map point (anti-drift, design §5.1).

    One forward snapshot: detect people, lift each, return the one closest to
    *world_xy* within *radius_m*. None if nobody qualifies. Used to re-find the
    customer (before serving) or the barman (at the bar) rather than trusting a
    stored coordinate after minutes in a moving room.

    With *prefer_calling* a still-waving person within range wins over a closer
    non-waving one — disambiguates the customer from a neighbour at an adjacent table
    (only ~1 m away here). Falls back to the nearest person if nobody is waving, since
    the customer often lowers their hand once the robot is clearly coming to them.
    """
    if radius_m is None:
        radius_m = _f("RESTAURANT_REACQUIRE_RADIUS_M", "1.5")
    pose = _robot_pose(ctx)
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
    on-robot). The first prompt greets/asks; later tries nudge them to repeat.
    """
    retries = int(os.getenv("RESTAURANT_ORDER_RETRIES", "2"))
    prompt = first_prompt
    for _ in range(retries + 1):
        recenter()
        answer = ctx.ask(prompt, retries=0)
        parsed = ctx.extract(prompts.Order, prompts.EXTRACT_ORDER_INSTRUCTIONS, answer or "")
        if parsed and parsed.items:
            return parsed.items
        prompt = prompts.ASK_REPEAT
    return []


def take_order(ctx: TaskContext, world_xy: tuple[float, float] | None = None) -> list[str]:
    """Greet the customer, capture and confirm their order. Real dialogue today.

    Gaze (rulebook-scored): if *world_xy* is given, the robot re-faces the customer before
    each utterance — MVP "look at the person" without a continuous-tracking thread (design
    §5.2). Capture re-asks the SAME customer on an unclear reply (see :func:`_capture_order`)
    instead of dropping them. Confirmation LISTENS: an explicit "no" re-takes the order (up
    to RESTAURANT_CONFIRM_RETRIES times — that step scores 2×160), while a silent/garbled
    reply counts as agreement so venue noise can't drop a good order. Returns [] only after
    the re-asks genuinely fail.
    """
    def recenter():
        if world_xy is not None:
            face_person(ctx, world_xy)

    items = _capture_order(ctx, recenter, prompts.GREET_CUSTOMER)
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


def return_to_bar(ctx: TaskContext) -> bool:
    """Drive to the bar anchor; optionally re-acquire the barman and face them.

    go_to gets us near the remembered anchor. With RESTAURANT_BAR_REACQUIRE on
    (default), the truth is the live camera, so we then turn toward the counter and
    look for the barman to face them for the relay (design §5.1), degrading to just
    reaching the anchor if no barman is seen. With it OFF, the robot simply parks at
    the fixed bar pose and stops — no rotate, no person search (the robot just comes
    to its station to fetch items).
    """
    bar = ctx.data.get("bar_anchor")
    if not bar:
        return False
    ok = ctx.goto(bar["x"], bar["y"], bar["heading"])
    if not _b("RESTAURANT_BAR_REACQUIRE", "1"):
        return ok  # just park at the fixed bar point — no rotate / barman search
    # Turn to face the order station before relaying. The anchor heading points at the
    # DINERS (the robot started facing them), but the counter/kitchen is off to the side —
    # RESTAURANT_COUNTER_REL_DEG is how far to rotate from the anchor heading to face it
    # (0 = straight ahead; +90 = counter on the left, -90 = on the right; flip the sign if
    # it turns the wrong way). Then refine onto the barman if one is visible there.
    rel = math.radians(_f("RESTAURANT_COUNTER_REL_DEG", "0"))
    if rel:
        ctx.rotate_to(bar["heading"] + rel)
    barman = find_person_near(ctx, (bar["x"], bar["y"]),
                              radius_m=_f("RESTAURANT_BARMAN_RADIUS_M", "2.5"))
    if barman is not None:
        face_person(ctx, barman.world_xy)
    return ok


def return_to_customer(ctx: TaskContext, world_xy: tuple[float, float]) -> tuple[float, float] | None:
    """Return to a customer: go near the stored point, then re-acquire them fresh.

    Returns the customer's refreshed map point (for serving) or None if they
    could not be re-found. Approaches the fresh detection to a stand-off; updates
    nothing in place (caller stores the returned point on the Order).
    """
    approach_to_standoff(ctx, world_xy)  # get into viewing range of the table
    fresh = find_person_near(ctx, world_xy)
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
