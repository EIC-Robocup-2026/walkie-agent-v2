"""Doing Laundry subtasks + the build_laundry_task factory (rulebook 5.4).

PLACEHOLDER scaffold mirroring tasks/HRI/subtasks.py. Flow per the rulebook:
navigate to the laundry area, (optionally) open the washing machine and remove
clothes, transport clothes to the folding table (optionally via the basket),
then fold (and stack) at least one T-shirt.

Folding and clothing pick-up are deformable-manipulation extension points — they
are honest stubs that announce the limitation and degrade (partial scoring is
allowed), exactly like tasks/HRI/FollowHostAndDropBag.

Blackboard layout (ctx.data):
    clothes: list[Garment]   # what RetrieveLaundry moved onto the table
"""

from __future__ import annotations

import os

from tasks.base import StepResult, SubTask, Task, TaskContext
from walkie_world.map.locations import resolve_pose

from . import prompts
from .skills import Garment, fold_garment, perceive_clothes, pick_garment, stack_garment

# PnP-style: map each Laundry nav waypoint to its canonical name in the shared
# LocationBook (the map editor's output), with the *_POSE env var as fallback.
_LOCATION_NAME = {
    "LAUNDRY_AREA_POSE": "laundry_area",
    "LAUNDRY_BASKET_POSE": "laundry_basket",
    "LAUNDRY_TABLE_POSE": "folding_table",
    "LAUNDRY_WASHER_POSE": "washing_machine",
}


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    """Map-frame waypoint: shared LocationBook (by name) -> *_POSE env var -> default."""
    return resolve_pose(_LOCATION_NAME.get(env_key), env_fallback=env_key, default=default)


def _optional(env_key: str) -> bool:
    return os.getenv(env_key, "0").lower() in ("1", "true", "yes")


class GoToLaundryArea(SubTask):
    """Enter the arena and navigate to the laundry area when the door opens."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.START_ANNOUNCE)
        x, y, h = _pose("LAUNDRY_AREA_POSE")
        if ctx.goto(x, y, h):
            ctx.score("navigate_laundry_area")  # the ONLY non-arm line on the sheet
            return StepResult.DONE
        return StepResult.RETRY


class OpenWashingMachine(SubTask):
    """Optional goal: open the washing-machine door and remove clothes.

    Gated by LAUNDRY_ENABLE_WASHER. STUB: no autonomous appliance/deformable
    manipulation yet — the rulebook lets the robot ask for help, so it does.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not _optional("LAUNDRY_ENABLE_WASHER"):
            return StepResult.DONE
        x, y, h = _pose("LAUNDRY_WASHER_POSE")
        ctx.goto(x, y, h)
        ctx.say(prompts.ASK_OPEN_WASHER)
        print("[laundry] TODO OpenWashingMachine — door + drum retrieval not implemented")
        return StepResult.DONE


class RetrieveLaundry(SubTask):
    """Move clothes from the basket (or machine) to the folding table.

    Optionally carries the basket itself (LAUNDRY_USE_BASKET) for extra points.
    TODO: pick-up + transport are stubs; perception is real.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.RETRIEVE_ANNOUNCE)
        x, y, h = _pose("LAUNDRY_BASKET_POSE")
        ctx.goto(x, y, h)
        clothes = perceive_clothes(ctx)
        ctx.data["clothes"] = clothes
        print(f"[laundry] perceived {len(clothes)} garment(s) to move")
        # TODO: if LAUNDRY_USE_BASKET, carry the basket to the folding surface;
        # otherwise pick garments one at a time and place them on the table.
        x, y, h = _pose("LAUNDRY_TABLE_POSE")
        ctx.goto(x, y, h)
        return StepResult.DONE


class FoldLaundry(SubTask):
    """Fold (and stack) the clothes on the table. At least one is the main goal.

    TODO: fold_garment / stack_garment are deformable-manip stubs.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.FOLD_ANNOUNCE)
        # Re-perceive on the table rather than trusting the pre-transport list.
        clothes: list[Garment] = perceive_clothes(ctx) or ctx.data.get("clothes", [])
        max_items = int(os.getenv("LAUNDRY_MAX_FOLD", "1"))
        # The arm lines are guarded on stub success (mirrors PnP's arm pass): the
        # pick/fold/stack stubs return False today, so nothing below fires until the
        # deformable-manip skill lands — at which point the tally is already correct.
        folded = 0
        for garment in clothes[:max_items]:
            if not pick_garment(ctx, garment):  # STUB
                continue
            ctx.score("pick_up_clothing")  # arm: grasped one garment (claimed)
            if not fold_garment(ctx, garment):  # STUB
                continue
            ctx.score("fold_clothing" if folded == 0 else "fold_additional")  # 1st vs additional
            folded += 1
            if stack_garment(ctx):  # STUB
                ctx.score("stack_folded")  # arm: stacked neatly (claimed)
        ctx.say(prompts.DONE_ANNOUNCE)
        return StepResult.DONE


def build_laundry_task(ctx: TaskContext) -> Task:
    """Construct the Doing Laundry task. Pure: touches no hardware at build time."""
    return Task(
        "Laundry",
        [
            GoToLaundryArea(),
            OpenWashingMachine(),  # optional, gated
            RetrieveLaundry(),
            FoldLaundry(),
        ],
        ctx,
    )
