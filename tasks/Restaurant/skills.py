"""Reusable perception / interaction / manipulation skills for Restaurant (rulebook 5.5).

Gesture (waving/calling) detection and customer approach are real heuristics;
order-taking dialogue is real (ask + STT + LLM extract); item collection and
serving reuse the shared grasp planner in tasks/manipulation.py (the planner is
the stub, the arm motion is real). The unattached-tray optional goal is left to
the subtask as a gated stub.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from tasks.base import TaskContext
from tasks.manipulation import perceive_surface, pick_object, place_at_pose
from tasks.skills import cxcywh_to_xyxy, is_calling_gesture

from . import prompts

BBox = tuple[float, float, float, float]


@dataclass
class CallingCustomer:
    """A customer detected calling/waving, with where to drive and how to face them."""

    bbox_xyxy: BBox           # person box in the snapshot (xyxy pixels)
    heading: float            # map-frame heading to turn toward them
    world_xy: tuple[float, float] | None  # map-frame table position when lifted
    crop: object | None = None  # PIL crop of the person, for the identify fallback


def detect_calling_customer(ctx: TaskContext) -> CallingCustomer | None:
    """Find a customer waving/calling and return where to go + how to face them.

    Real heuristic: pose-estimate everyone, keep those with a raised wrist, pick
    the most central one, lift their box to a map-frame table position via the
    depth snapshot, and compute a heading toward them. Returns None when nobody
    is clearly calling. Never raises.
    """
    snap = ctx.snapshot()
    if snap is None:
        return None
    img = snap.img
    try:
        persons = ctx.walkieAI.image.estimate_poses(img)
    except Exception as exc:
        print(f"[restaurant.skills] pose estimation failed ({exc})")
        return None
    conf = float(os.getenv("RESTAURANT_WAVE_CONF", "0.3"))
    callers = [p for p in persons if is_calling_gesture(p, conf)]
    if not callers:
        return None
    img_w = img.width
    target = min(callers, key=lambda p: abs(p.bbox[0] - img_w / 2))  # most central
    xyxy = cxcywh_to_xyxy(target.bbox)

    world_xy = None
    if getattr(snap, "has_geometry", False):
        try:
            pt = snap.bbox_world_point(xyxy)
            world_xy = tuple(pt[:2]) if pt is not None else None
        except Exception:
            world_xy = None

    pose = ctx.current_pose()
    if world_xy is not None:
        heading = math.atan2(world_xy[1] - pose["y"], world_xy[0] - pose["x"])
    else:
        # Fallback: pixel x -> heading offset from the robot's current facing.
        hfov = math.radians(float(os.getenv("RESTAURANT_CAMERA_HFOV_DEG", "110")))
        offset = (target.bbox[0] / img_w - 0.5) * hfov
        heading = pose["heading"] - offset

    crop = None
    try:
        m = 20  # px padding so the person isn't clipped
        x1, y1, x2, y2 = xyxy
        crop = img.crop((
            max(0, int(x1 - m)), max(0, int(y1 - m)),
            min(img.width, int(x2 + m)), min(img.height, int(y2 + m)),
        ))
    except Exception:
        crop = None
    return CallingCustomer(bbox_xyxy=xyxy, heading=heading, world_xy=world_xy, crop=crop)


def navigate_to_customer(ctx: TaskContext, cust: CallingCustomer) -> bool:
    """Drive to the calling customer's table (online nav in an unmapped venue).

    With a lifted map-frame position, drive to a stop-short goal on the
    robot->customer line and face them; otherwise just rotate to face. Returns
    whether the navigation reported success (False triggers the identify
    fallback so partial points can still be claimed).
    """
    if cust.world_xy is None:
        ctx.rotate_to(cust.heading)
        print("[restaurant.skills] no 3D fix on the customer; rotated to face only")
        return False
    stop = float(os.getenv("RESTAURANT_APPROACH_DISTANCE_M", "0.8"))
    pose = ctx.current_pose()
    wx, wy = cust.world_xy
    dx, dy = wx - pose["x"], wy - pose["y"]
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    if dist <= stop:
        return ctx.goto(pose["x"], pose["y"], heading)  # already close: just face
    frac = (dist - stop) / dist
    return ctx.goto(pose["x"] + dx * frac, pose["y"] + dy * frac, heading)


def identify_customer(ctx: TaskContext, cust: CallingCustomer) -> None:
    """Partial-credit fallback: clearly identify the detected customer.

    The rulebook awards partial points when a detected customer isn't reached if
    the robot clearly identifies the person. Face them and announce a visual
    description (captioned from their crop). Best-effort, never raises.
    """
    if cust.heading is not None:
        ctx.rotate_to(cust.heading)
    desc = None
    if cust.crop is not None:
        try:
            desc = ctx.walkieAI.image.caption(cust.crop, prompt=prompts.IDENTIFY_CAPTION_PROMPT)
        except Exception as exc:
            print(f"[restaurant.skills] identify caption failed ({exc})")
    ctx.say(prompts.IDENTIFY_CUSTOMER.format(desc=desc or "the person who is calling"))


def take_order(ctx: TaskContext) -> list[str]:
    """Greet the customer, capture and confirm their order. Real dialogue today.

    The caller faces the customer first (eye contact is scored). The rulebook
    scores 'understand and confirm the order'.
    """
    answer = ctx.ask(prompts.GREET_CUSTOMER)
    if not answer:
        answer = ctx.ask(prompts.ASK_REPEAT, retries=0)
    parsed = ctx.extract(prompts.Order, prompts.EXTRACT_ORDER_INSTRUCTIONS, answer or "")
    items = parsed.items if parsed else []
    if items:
        ctx.say(prompts.CONFIRM_ORDER.format(items=", ".join(items)))
        ctx.say(prompts.ORDER_TAKEN)
    return items


def _bar_classes(item: str) -> list[str]:
    extra = [c.strip() for c in os.getenv(
        "RESTAURANT_BAR_CLASSES", "can,bottle,cup,snack,fruit,box,bag"
    ).split(",") if c.strip()]
    # The ordered item first so the open-vocab detector is biased toward it.
    return [item] + [c for c in extra if c.lower() != item.lower()]


def pick_bar_item(ctx: TaskContext, item: str) -> bool:
    """Perceive the kitchen-bar and pick the object matching *item*. Real arm motion.

    One object per call (single-arm carry). Returns False (and announces) when the
    item can't be found or the grasp fails, so the caller score-degrades.
    """
    objects = perceive_surface(ctx, _bar_classes(item))
    if not objects:
        ctx.say(prompts.ITEM_NOT_FOUND.format(item=item))
        return False
    matches = [o for o in objects if item.lower() in (o.class_name or "").lower()]
    target = max(matches or objects, key=lambda o: o.confidence)
    print(f"[restaurant.skills] picking '{item}' as {target.class_name} ({target.confidence:.2f})")
    return pick_object(ctx, target)


def serve_item(ctx: TaskContext) -> bool:
    """Place the held item on the customer's table (RESTAURANT_SERVE_POSE).

    Falls back to a drop-in-front when no serve pose is configured. Returns
    whether a located placement was done.
    """
    return place_at_pose(ctx, os.getenv("RESTAURANT_SERVE_POSE", ""))
