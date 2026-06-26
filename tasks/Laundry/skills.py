"""Reusable perception / manipulation skills for Doing Laundry (rulebook 5.4).

Detection of clothing is real (open-vocab detector). ``pick_garment`` now routes
to the shared grasp system (``tasks.skills.pick_object``) behind the
``LAUNDRY_ARM_CALIBRATED`` gate, mirroring PickAndPlace / Restaurant — a single
garment is a graspable object, so it reuses the same GraspNet pipeline. The
deformable steps (``fold_garment`` / ``stack_garment``) are still honest stubs:
they need a bimanual fold planner that does not exist yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from tasks.base import TaskContext
from tasks.skills import pick_object as _pick_object

BBox = tuple[float, float, float, float]


def arm_enabled() -> bool:
    """Master arm gate. Off (default) -> announce-only; on -> real grasp.

    The arm is brought up as a separate skill; until it lands every grasp is
    gated so the nav + perception flow can be validated on its own. Mirrors
    PNP_ARM_CALIBRATED / RESTAURANT_ARM_CALIBRATED.
    """
    return os.getenv("LAUNDRY_ARM_CALIBRATED", "0").lower() in ("1", "true", "yes")


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
        detections = ctx.walkieAI.image.detect(snap.img, prompts=classes)
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


# --- Manipulation primitives ------------------------------------------------
def pick_garment(ctx: TaskContext, garment: Garment) -> bool:
    """Grasp one garment (one at a time — picking multiple is penalised).

    Gated by LAUNDRY_ARM_CALIBRATED. Gate off (default): announce-only, returns
    False so the caller score-degrades. Gate on: delegates to the shared grasp
    pipeline (``tasks.skills.pick_object``), re-acquiring the garment by its class
    name and running GraspNet — the same path PickAndPlace / Restaurant use.
    """
    if not arm_enabled():
        from . import prompts

        ctx.say(prompts.PICK_NOT_AVAILABLE)
        print(f"[laundry.skills] pick_garment({garment.class_name}) — arm gated "
              "(LAUNDRY_ARM_CALIBRATED=0)")
        return False
    return _pick_object(ctx, prompts=[garment.class_name])


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
