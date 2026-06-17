"""Pick-and-Place-specific skills (rulebook 5.2).

A thin facade over the shared manipulation layer (tasks/manipulation.py): the
grasp planner + real pick/place + perception-lift live there and are reused by
the other manipulation tasks (Restaurant, ...). What stays here is the
PickAndPlace-specific glue: which classes to detect, the LLM destination sort,
and the destination-furniture place logic.
"""

from __future__ import annotations

import os

from tasks.base import TaskContext
from tasks.manipulation import (
    DetectedObject,
    perceive_surface as _perceive_surface,
    pick_object,
    place_at_pose,
    plan_grasp,
    release_in_front,
)

from . import prompts

# Re-exported so tasks/PickAndPlace/subtasks.py keeps importing from .skills.
__all__ = [
    "DetectedObject",
    "perceive_surface",
    "pick_object",
    "place_at",
    "place_object",
    "plan_grasp",
    "sort_object",
]

# ServeBreakfast drops items at fixed table slots — same as a generic placement.
place_at = place_at_pose


def _classes(env: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env, default).split(",") if c.strip()]


def perceive_surface(ctx: TaskContext, classes: list[str] | None = None) -> list[DetectedObject]:
    """Detect dining-table / extra-surface objects (PNP_TABLE_CLASSES by default)."""
    resolved = classes or _classes(
        "PNP_TABLE_CLASSES", "cup,mug,plate,fork,knife,spoon,bottle,box,can,bowl"
    )
    return _perceive_surface(ctx, resolved)


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


def place_object(ctx: TaskContext, destination: str, *, group: str | None = None) -> bool:
    """Navigate to *destination* furniture and release the held object there.

    Navigation to the furniture waypoint is real (config poses); the place pose
    is the stub (config PNP_PLACE_POSE_<DEST>). Falls back to a drop-in-front when
    no place pose is configured. Returns whether a located placement was done.
    """
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
    ok = place_at_pose(ctx, place_pose)
    ctx.say(prompts.PLACED.format(destination=destination))
    return ok
