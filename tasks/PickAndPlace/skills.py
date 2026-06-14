"""Reusable perception / manipulation skills for Pick and Place (rulebook 5.2).

PLACEHOLDER. Plain functions over a TaskContext, mirroring tasks/HRI/skills.py.
Perception lifts and the arm primitives are stubs — fill them in as the SDK
manipulation API lands. Anything that turns out generic enough (e.g. a real
pick/place) should graduate to tasks/base.py so the other manipulation tasks
(Laundry, Restaurant) can lift it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tasks.base import TaskContext

from . import prompts

BBox = tuple[float, float, float, float]


@dataclass
class DetectedObject:
    """One object perceived on a surface, with its map-frame position when lifted."""

    bbox_xyxy: BBox
    class_name: str
    confidence: float
    world_xy: tuple[float, float] | None = None


def _classes(env: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env, default).split(",") if c.strip()]


def perceive_surface(ctx: TaskContext, classes: list[str] | None = None) -> list[DetectedObject]:
    """Detect objects on the surface in front of the robot (open-vocab).

    Returns DetectedObjects with bboxes; world_xy is lifted against the snapshot
    geometry when available. Empty list on any capture/detection failure (the
    task degrades, never raises). The rulebook requires the robot to *communicate
    its perception* to the referee — the caller should announce the result.
    """
    snap = ctx.snapshot()
    if snap is None:
        return []
    prompts_list = classes or _classes(
        "PNP_TABLE_CLASSES", "cup,mug,plate,fork,knife,spoon,bottle,box,can"
    )
    try:
        detections = ctx.walkieAI.object_detection.detect(snap.img, prompts=prompts_list)
    except Exception as exc:
        print(f"[pnp.skills] detection failed ({exc})")
        return []
    out: list[DetectedObject] = []
    for det in detections:
        world_xy = None
        if getattr(snap, "has_geometry", False):
            try:
                world_xy = snap.bbox_world_xy(det.bbox)
            except Exception:
                world_xy = None
        out.append(
            DetectedObject(
                bbox_xyxy=tuple(det.bbox),
                class_name=det.class_name or "object",
                confidence=det.confidence or 0.0,
                world_xy=world_xy,
            )
        )
    return out


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


# --- Manipulation primitives (EXTENSION POINTS — not implemented) -----------
def pick_object(ctx: TaskContext, obj: DetectedObject) -> bool:
    """Grasp *obj* and lift it for transport.

    STUB: needs a real arm/grasp planner that does not exist yet. Announces the
    limitation and reports failure so the caller can score-degrade gracefully
    (the rulebook allows partial scoring). Implement against walkie.arm.
    """
    ctx.say(prompts.PICK_NOT_AVAILABLE.format(obj=obj.class_name))
    print(f"[pnp.skills] TODO pick_object({obj.class_name}) — manipulation not implemented")
    return False


def place_object(ctx: TaskContext, destination: str, *, group: str | None = None) -> bool:
    """Place the held object at *destination* ('dishwasher'|'trash'|'cabinet').

    STUB: navigation to the destination furniture is real-ish (config poses);
    the actual place motion is not implemented. Returns False until then.
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
    print(f"[pnp.skills] TODO place_object({destination}, group={group}) — not implemented")
    return False
