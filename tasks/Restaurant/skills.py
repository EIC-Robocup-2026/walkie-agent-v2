"""Reusable perception / interaction skills for the Restaurant task (rulebook 5.5).

PLACEHOLDER. Gesture (waving/calling) detection, online navigation to a table,
and item manipulation are the hard parts and are honest stubs. Order-taking
dialogue is real (ask + STT + LLM extract).
"""

from __future__ import annotations

import math
import os

from tasks.base import TaskContext

from . import prompts


def detect_calling_customer(ctx: TaskContext) -> float | None:
    """Find a customer calling/waving and return a map-frame heading toward them.

    STUB heuristic: detect people with pose estimation and return the heading to
    the nearest one as a placeholder for real wave/gesture recognition. Returns
    None when nobody is detected. TODO: classify an actual waving gesture from the
    pose keypoints (arm raised + motion) rather than "any person".
    """
    img = ctx.capture()
    if img is None:
        return None
    try:
        persons = ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[restaurant.skills] pose estimation failed ({exc})")
        return None
    if not persons:
        return None
    # TODO: real gesture classification. For now, aim at the most central person
    # (closest to the image centre x) as a stand-in for "the one calling".
    hfov = math.radians(float(os.getenv("RESTAURANT_CAMERA_HFOV_DEG", "110")))
    img_w = img.width
    target = min(persons, key=lambda p: abs(p.bbox[0] - img_w / 2))
    # Pixel x -> heading offset from the robot's current facing.
    offset = (target.bbox[0] / img_w - 0.5) * hfov
    pose = ctx.current_pose()
    print(f"[restaurant.skills] TODO gesture detection stub — aiming at central person")
    return pose["heading"] - offset  # +x to the right -> turn right (negative)


def navigate_to_customer(ctx: TaskContext, heading: float) -> bool:
    """Drive to the calling customer's table (online nav in an unknown venue).

    STUB: rotates toward the customer and reports partial progress. Real version
    needs online mapping + approach planning (no prior map is allowed). Returns
    False until implemented so the caller can claim partial points by clearly
    identifying the person instead (rulebook remark).
    """
    ctx.rotate_to(heading)
    print("[restaurant.skills] TODO navigate_to_customer — online nav not implemented")
    return False


def take_order(ctx: TaskContext) -> list[str]:
    """Greet the customer, capture and confirm their order. Real dialogue today.

    The rulebook scores 'understand and confirm the order' and penalises not
    making eye contact — a real version should face the customer while talking.
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


# --- Manipulation primitives (EXTENSION POINTS — not implemented) -----------
def collect_items(ctx: TaskContext, items: list[str]) -> bool:
    """Pick the ordered items from the kitchen-bar (optionally onto a tray).

    STUB: needs grasping. Returns False so the caller score-degrades.
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
