"""Reusable perception / manipulation skills for Doing Laundry (rulebook 5.4).

PLACEHOLDER. Detection of clothing is real-ish (open-vocab detector); the
deformable-manipulation primitives (pick a T-shirt, fold it, stack it) are honest
stubs — they need a bimanual grasp + fold planner that does not exist yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tasks.base import TaskContext

BBox = tuple[float, float, float, float]


@dataclass
class Garment:
    """One detected piece of clothing with its map-frame position when lifted."""

    bbox_xyxy: BBox
    class_name: str
    confidence: float
    world_xy: tuple[float, float] | None = None


def perceive_clothes(ctx: TaskContext) -> list[Garment]:
    """Detect clothing in front of the robot (basket / table / machine drum).

    The rulebook says laundry is exclusively T-shirts; the detector prompt list
    is kept broad so a bunched-up garment still fires. Empty list on any failure.
    """
    snap = ctx.snapshot()
    if snap is None:
        return []
    classes = [
        c.strip()
        for c in os.getenv("LAUNDRY_CLOTH_CLASSES", "t-shirt,shirt,clothing,cloth,towel").split(",")
        if c.strip()
    ]
    try:
        detections = ctx.walkieAI.object_detection.detect(snap.img, prompts=classes)
    except Exception as exc:
        print(f"[laundry.skills] detection failed ({exc})")
        return []
    out: list[Garment] = []
    for det in detections:
        world_xy = None
        if getattr(snap, "has_geometry", False):
            try:
                world_xy = snap.bbox_world_xy(det.bbox)
            except Exception:
                world_xy = None
        out.append(
            Garment(
                bbox_xyxy=tuple(det.bbox),
                class_name=det.class_name or "clothing",
                confidence=det.confidence or 0.0,
                world_xy=world_xy,
            )
        )
    return out


# --- Manipulation primitives (EXTENSION POINTS — not implemented) -----------
def pick_garment(ctx: TaskContext, garment: Garment) -> bool:
    """Grasp one garment (one at a time — picking multiple is penalised).

    STUB: needs a deformable grasp. Returns False so the caller score-degrades.
    """
    from . import prompts

    ctx.say(prompts.PICK_NOT_AVAILABLE)
    print(f"[laundry.skills] TODO pick_garment({garment.class_name}) — not implemented")
    return False


def fold_garment(ctx: TaskContext, garment: Garment) -> bool:
    """Fold one garment neatly on the folding table (scored on neatness).

    STUB: needs a bimanual fold sequence. Returns False until implemented.
    """
    from . import prompts

    ctx.say(prompts.FOLD_NOT_AVAILABLE)
    print(f"[laundry.skills] TODO fold_garment({garment.class_name}) — not implemented")
    return False


def stack_garment(ctx: TaskContext) -> bool:
    """Stack the just-folded garment onto the neat pile (extra points per item).

    STUB: depends on fold_garment. Returns False until implemented.
    """
    print("[laundry.skills] TODO stack_garment — not implemented")
    return False
