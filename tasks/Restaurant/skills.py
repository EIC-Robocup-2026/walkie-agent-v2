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
import time
from dataclasses import dataclass

from client.pose_estimation import PersonPose
from tasks.base import TaskContext

from . import prompts

BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _f(name: str, default: str) -> float:
    return float(os.getenv(name, default))


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


def _scan_offsets() -> list[float]:
    """Base-rotation offsets (deg) covering the scan arc, centered on entry heading."""
    arc = _f("RESTAURANT_SCAN_ARC_DEG", "120")
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
    if pose is None:
        print("[restaurant.skills] scan_for_callers: no odometry fix; skipping sweep")
        return []
    rx, ry, center = pose["x"], pose["y"], pose["heading"]
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
            persons = ctx.walkieAI.pose_estimation.estimate(snap.img)
        except Exception as exc:
            print(f"[restaurant.skills] pose estimation failed ({exc})")
            continue
        for p in persons:
            if not is_calling(p):
                continue
            xyxy = _cxcywh_to_xyxy(p.bbox)
            world_xy = snap.bbox_world_xy(xyxy) if getattr(snap, "has_geometry", False) else None
            if world_xy is None:
                continue  # can't approach what we can't place on the map
            bearing = math.atan2(world_xy[1] - ry, world_xy[0] - rx)
            callers.append(Caller(world_xy, bearing, xyxy, p.confidence or 0.0))
    ctx.rotate_to(center)  # leave the base back at the entry heading
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


# ---------------------------------------------------------------------------
# Approach + facing (SLAM-backed: lift -> go_to a stand-off point)
# ---------------------------------------------------------------------------
def heading_to(ctx: TaskContext, world_xy: tuple[float, float]) -> float | None:
    """Map-frame heading from the robot toward *world_xy*; None without an odom fix."""
    pose = _robot_pose(ctx)
    if pose is None:
        return None
    return math.atan2(world_xy[1] - pose["y"], world_xy[0] - pose["x"])


def face_person(ctx: TaskContext, world_xy: tuple[float, float]) -> bool:
    """Rotate the base to face a map point (head is tilt-only). Best-effort."""
    heading = heading_to(ctx, world_xy)
    return ctx.rotate_to(heading) if heading is not None else False


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
    dx, dy = cx - rx, cy - ry
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)  # face the customer at the end
    if dist <= standoff_m:
        return ctx.rotate_to(heading)  # already close enough: just face them
    ux, uy = dx / dist, dy / dist
    tx, ty = cx - ux * standoff_m, cy - uy * standoff_m
    print(f"[restaurant.skills] approach -> ({tx:.2f},{ty:.2f}) facing {math.degrees(heading):.0f}deg "
          f"(customer {dist:.2f}m, standoff {standoff_m:.2f}m)")
    return ctx.goto(tx, ty, heading)


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
        return ctx.walkieAI.image_caption.caption(crop, prompt=prompts.CUSTOMER_APPEARANCE_PROMPT)
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
        persons = ctx.walkieAI.pose_estimation.estimate(snap.img)
    except Exception as exc:
        print(f"[restaurant.skills] appearance pose estimation failed ({exc})")
        return None
    best_box = None
    best_d = float("inf")
    for p in persons:
        xyxy = _cxcywh_to_xyxy(p.bbox)
        wxy = snap.bbox_world_xy(xyxy)
        if wxy is None:
            continue
        d = math.hypot(wxy[0] - world_xy[0], wxy[1] - world_xy[1])
        if d < best_d:
            best_box, best_d = xyxy, d
    return describe_customer(ctx, snap, best_box) if best_box is not None else None


def find_person_near(ctx: TaskContext, world_xy: tuple[float, float], *,
                     radius_m: float | None = None) -> Caller | None:
    """Re-acquire a person near a remembered map point (anti-drift, design §5.1).

    One forward snapshot: detect people, lift each, return the one closest to
    *world_xy* within *radius_m*. None if nobody qualifies. Used to re-find the
    customer (before serving) or the barman (at the bar) rather than trusting a
    stored coordinate after minutes in a moving room.
    """
    if radius_m is None:
        radius_m = _f("RESTAURANT_REACQUIRE_RADIUS_M", "1.5")
    pose = _robot_pose(ctx)
    snap = ctx.snapshot()
    if snap is None or not getattr(snap, "has_geometry", False):
        return None
    try:
        persons = ctx.walkieAI.pose_estimation.estimate(snap.img)
    except Exception as exc:
        print(f"[restaurant.skills] re-acquire pose estimation failed ({exc})")
        return None
    best: Caller | None = None
    best_d = radius_m
    for p in persons:
        xyxy = _cxcywh_to_xyxy(p.bbox)
        wxy = snap.bbox_world_xy(xyxy)
        if wxy is None:
            continue
        d = math.hypot(wxy[0] - world_xy[0], wxy[1] - world_xy[1])
        if d <= best_d:
            bearing = (math.atan2(wxy[1] - pose["y"], wxy[0] - pose["x"])
                       if pose else 0.0)
            best, best_d = Caller(wxy, bearing, xyxy, p.confidence or 0.0), d
    return best


def take_order(ctx: TaskContext, world_xy: tuple[float, float] | None = None) -> list[str]:
    """Greet the customer, capture and confirm their order. Real dialogue today.

    Gaze (rulebook-scored): if *world_xy* is given, the robot re-faces the
    customer before each utterance — MVP "look at the person" without a full
    continuous-tracking thread (design doc §5.2). The order is asked in one
    question (no penalised confirmation questions); re-asking on a misheard reply
    is allowed and not penalised.
    """
    def recenter():
        if world_xy is not None:
            face_person(ctx, world_xy)

    recenter()
    answer = ctx.ask(prompts.GREET_CUSTOMER)
    if not answer:
        recenter()
        answer = ctx.ask(prompts.ASK_REPEAT, retries=0)
    parsed = ctx.extract(prompts.Order, prompts.EXTRACT_ORDER_INSTRUCTIONS, answer or "")
    items = parsed.items if parsed else []
    if items:
        recenter()
        ctx.say(prompts.CONFIRM_ORDER.format(items=", ".join(items)))
        ctx.say(prompts.ORDER_TAKEN)
    return items


def return_to_bar(ctx: TaskContext) -> bool:
    """Drive to the bar anchor, then re-acquire the barman and face them.

    go_to gets us near the remembered anchor; the truth is the live camera, so we
    look for the person there and face them for the relay (design §5.1). Degrades
    to just reaching the anchor if no barman is seen.
    """
    bar = ctx.data.get("bar_anchor")
    if not bar:
        return False
    ok = ctx.goto(bar["x"], bar["y"], bar["heading"])
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


# --- Manipulation primitives (EXTENSION POINTS — Phase 2, not implemented) --
def collect_items(ctx: TaskContext, items: list[str]) -> bool:
    """Pick the ordered items from the kitchen-bar (optionally onto a tray).

    STUB: needs grasping on go_to_pose/control_gripper. Returns False so the
    caller score-degrades.
    """
    ctx.say(prompts.PICK_NOT_AVAILABLE)
    print(f"[restaurant.skills] TODO collect_items({items}) — manipulation not implemented")
    return False


def serve_order(ctx: TaskContext, items: list[str]) -> bool:
    """Place the items on the customer's table (or hand off the tray).

    STUB: depends on collect_items. Returns False until implemented.
    """
    print(f"[restaurant.skills] TODO serve_order({items}) — not implemented")
    return False
