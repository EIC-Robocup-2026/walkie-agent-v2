"""Pick-and-Place-specific skills (rulebook 5.2).

A thin facade over the shared manipulation layer (tasks/manipulation.py): the
grasp planner + real pick/place + perception-lift live there and are reused by
the other manipulation tasks (Restaurant, ...). What stays here is the
PickAndPlace-specific glue: which classes to detect, the LLM destination sort,
the destination-furniture place logic, and the *communicate-perception* layer.

Arm-gating (mirrors Restaurant's RESTAURANT_ARM_CALIBRATED): the arm is being
brought up as a separate skill, so PickAndPlace gates every grasp/place behind
``PNP_ARM_CALIBRATED`` (default off). With the gate off the flow still runs
end-to-end and scores its non-arm budget — navigate, *recognize each object*,
and *indicate the correct placement* (rulebook scoresheet + remark 16, which
lets the robot communicate perception by pointing / announcing / visualizing one
object at a time, no grasp required). Flip the gate to 1 once the arm skill lands
and the manipulation budget unlocks with no flow rewrite.
"""

from __future__ import annotations

import math
import os

from tasks.base import TaskContext
from tasks.manipulation import (
    DetectedObject,
    perceive_surface as _perceive_surface,
    pick_object as _pick_object,
    place_at_pose as _place_at_pose,
    plan_grasp,
    release_in_front,
)

from . import prompts

# Re-exported so tasks/PickAndPlace/subtasks.py keeps importing from .skills.
__all__ = [
    "DetectedObject",
    "arm_enabled",
    "announce_object",
    "indicate_placement",
    "perceive_and_indicate_shelf",
    "perceive_surface",
    "pick_object",
    "place_at",
    "place_object",
    "plan_grasp",
    "sort_object",
]


# --- gating -----------------------------------------------------------------
def arm_enabled() -> bool:
    """Master arm gate. Off (default) -> indicate-only; on -> real grasp/place.

    The arm is a separate skill under development; until it lands every pick/place
    is gated so the non-arm pipeline (nav + perception + communicating placement)
    can be validated and scored on its own. Mirrors RESTAURANT_ARM_CALIBRATED.
    """
    return os.getenv("PNP_ARM_CALIBRATED", "0").lower() in ("1", "true", "yes")


def _point_at_objects() -> bool:
    """PNP_POINT_AT_OBJECTS: rotate the base to face each object while indicating.

    Off by default — naming the object already satisfies remark 16, and a base
    rotation per object is slow against the 7-minute clock. Turn on for a demo
    where an unambiguous physical "point" is worth the time.
    """
    return os.getenv("PNP_POINT_AT_OBJECTS", "0").lower() in ("1", "true", "yes")


def _classes(env: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env, default).split(",") if c.strip()]


# --- perception -------------------------------------------------------------
def perceive_surface(ctx: TaskContext, classes: list[str] | None = None) -> list[DetectedObject]:
    """Detect dining-table / extra-surface objects (PNP_TABLE_CLASSES by default)."""
    resolved = classes or _classes(
        "PNP_TABLE_CLASSES", "cup,mug,plate,fork,knife,spoon,bottle,box,can,bowl"
    )
    return _perceive_surface(ctx, resolved)


def _face(ctx: TaskContext, world_xy: tuple[float, float]) -> None:
    """Rotate the base to face *world_xy* — a non-arm "point" at one object."""
    pose = ctx.current_pose()
    heading = math.atan2(world_xy[1] - pose["y"], world_xy[0] - pose["x"])
    ctx.rotate_to(heading)


def announce_object(ctx: TaskContext, obj: DetectedObject) -> None:
    """Speak one recognized object — the scoresheet 'correctly recognize an object'.

    Optionally rotates the base to face it first (PNP_POINT_AT_OBJECTS) so the
    indication is unambiguous (rulebook remark 16, communicating perception).
    """
    if _point_at_objects() and obj.world_xy is not None:
        _face(ctx, obj.world_xy)
    ctx.say(prompts.RECOGNIZE_OBJECT.format(obj=obj.class_name))
    ctx.score("recognize_object")  # 'correctly recognize an object' (claimed)


def indicate_placement(
    ctx: TaskContext, obj: DetectedObject, destination: str, group: str | None = None
) -> None:
    """Announce the placement we intend for *obj* — 'indicate the correct placement'.

    Always runs (arm gate independent): this is how the robot communicates its
    perception/plan to the referee per rulebook remark 16, and it scores whether
    or not the arm later actually moves the object.
    """
    if _point_at_objects() and obj.world_xy is not None:
        _face(ctx, obj.world_xy)
    where = destination if not group else f"{destination}, grouped with the {group}"
    ctx.say(prompts.INDICATE_PLACEMENT.format(obj=obj.class_name, where=where))


def perceive_and_indicate_shelf(ctx: TaskContext) -> list[DetectedObject]:
    """Perceive the cabinet shelves and announce the groups present.

    Scores the scoresheet's 'perceive objects on a shelf and indicate the correct
    placement' line: the robot looks at the cabinet, names what is already grouped
    there, and states it will match new items to those groups. Best-effort — an
    empty perception just announces that the shelves look empty.
    """
    classes = _classes("PNP_CABINET_CLASSES", "box,can,bottle,snack,cup,bowl,carton,jar")
    shelf = _perceive_surface(ctx, classes)
    groups = sorted({o.class_name for o in shelf})
    ctx.say(prompts.SHELF_PERCEIVE.format(groups=", ".join(groups) if groups else "an empty shelf"))
    ctx.score("shelf_indicate")  # 'perceive on a shelf + indicate placement' (claimed)
    return shelf


# --- sorting ----------------------------------------------------------------
def sort_object(ctx: TaskContext, obj: DetectedObject) -> prompts.ObjectSort | None:
    """LLM decision: dishwasher vs trash vs cabinet (+ group) for one object.

    Real glue — the categorisation is pure reasoning over the class name and the
    run's designated trash category, so it works today. Returns None on
    extraction failure (caller should default to the cabinet).
    """
    trash_category = os.getenv("PNP_TRASH_CATEGORY", "").strip() or "(announced at setup)"
    text = (
        f"Object class: {obj.class_name}. "
        f"Designated trash category for this run: {trash_category}."
    )
    return ctx.extract(prompts.ObjectSort, prompts.SORT_OBJECT_INSTRUCTIONS, text)


# --- manipulation (gated) ---------------------------------------------------
def pick_object(ctx: TaskContext, obj: DetectedObject) -> bool:
    """Grasp *obj* for transport — gated by PNP_ARM_CALIBRATED.

    Gate off (default): no arm motion. Announces that the arm is not enabled and
    returns False so the caller's place step also stays indicate-only. Gate on:
    delegates to the shared real grasp pipeline (tasks/manipulation.pick_object).
    """
    if not arm_enabled():
        ctx.say(prompts.PICK_GATED.format(obj=obj.class_name))
        print(f"[pnp.skills] pick_object({obj.class_name}) — arm gated (PNP_ARM_CALIBRATED=0)")
        return False
    return _pick_object(ctx, obj)


def place_at(ctx: TaskContext, pose6: str) -> bool:
    """Release the held object at a fixed arm-frame pose — gated by PNP_ARM_CALIBRATED."""
    if not arm_enabled():
        print("[pnp.skills] place_at — arm gated; nothing to release")
        return False
    return _place_at_pose(ctx, pose6)


def place_object(ctx: TaskContext, destination: str, *, group: str | None = None) -> bool:
    """Navigate to *destination* furniture and release the held object there.

    Gated by PNP_ARM_CALIBRATED: with the arm off there is nothing held to place,
    so this no-ops (the placement was already *indicated* by indicate_placement).
    With the arm on, navigation to the furniture waypoint is real (config poses);
    the place pose is the stub (config PNP_PLACE_POSE_<DEST>), falling back to a
    drop-in-front when none is configured. Returns whether a located placement was done.
    """
    if not arm_enabled():
        print(f"[pnp.skills] place_object({destination}) — arm gated; placement indicated only")
        return False
    pose_key = {
        "dishwasher": "PNP_DISHWASHER_POSE",
        "trash": "PNP_TRASH_BIN_POSE",
        "cabinet": "PNP_CABINET_POSE",
    }.get(destination)
    if pose_key and os.getenv(pose_key):
        from .subtasks import _pose  # local import: shared pose parser

        x, y, h = _pose(pose_key)
        ctx.goto(x, y, h)
    place_pose = (
        os.getenv(
            {
                "dishwasher": "PNP_PLACE_POSE_DISHWASHER",
                "trash": "PNP_PLACE_POSE_TRASH",
                "cabinet": "PNP_PLACE_POSE_CABINET",
            }.get(destination, ""),
            "",
        ).strip()
        or os.getenv("PNP_PLACE_POSE_DEFAULT", "").strip()
    )
    if not place_pose:
        print(f"[pnp.skills] place_object({destination}) — no place pose; dropping in front")
        return release_in_front(ctx)
    ok = _place_at_pose(ctx, place_pose)
    ctx.say(prompts.PLACED.format(destination=destination))
    return ok
