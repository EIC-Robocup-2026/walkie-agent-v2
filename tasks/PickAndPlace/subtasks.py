"""Pick and Place subtasks + the build_pick_and_place_task factory (rulebook 5.2).

PLACEHOLDER scaffold mirroring tasks/HRI/subtasks.py. The flow follows the
rulebook procedure: enter the kitchen, tidy the dining table (tableware/cutlery
-> dishwasher, trash -> bin, other -> cabinet grouped), serve breakfast, tidy
the extra surface. Optional goals (operate the dishwasher, pour) are gated stubs.

Each step does the real glue it can today (navigation to config poses, LLM
sorting, spoken perception announcements) and leaves the manipulation as honest
TODO stubs that degrade rather than crash — partial scoring is allowed, so a
failed non-critical step logs and the task moves on (see tasks/base.py).

Blackboard layout (ctx.data):
    table_objects: list[DetectedObject]   # from PerceiveDiningTable
    sorted:        {destination: [DetectedObject, ...]}  # after TidyDiningTable
"""

from __future__ import annotations

import os

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import DetectedObject, perceive_surface, pick_object, place_object, sort_object


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    """Parse a map-frame waypoint 'x,y,heading_rad' from the environment."""
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _optional(env_key: str) -> bool:
    return os.getenv(env_key, "0").lower() in ("1", "true", "yes")


class GoToKitchen(SubTask):
    """Enter the arena and navigate to the kitchen when the door opens."""

    critical = True  # nothing else works if we never reach the kitchen

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("PNP_KITCHEN_POSE")
        return StepResult.DONE if ctx.goto(x, y, h) else StepResult.RETRY


class PerceiveDiningTable(SubTask):
    """Detect + recognise the objects on the dining table, announce them.

    The rulebook scores 'correctly recognize an object' and requires the robot to
    communicate its perception to the referee, so this step is mostly real today.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("PNP_DINING_TABLE_POSE")
        ctx.goto(x, y, h)
        objects = perceive_surface(ctx)
        ctx.data["table_objects"] = objects
        ctx.say(prompts.PERCEPTION_ANNOUNCE.format(count=len(objects)))
        for o in objects:
            print(f"[pnp] perceived {o.class_name} ({o.confidence:.2f}) @ {o.world_xy}")
        return StepResult.DONE


class TidyDiningTable(SubTask):
    """Sort each table object and place it (dishwasher / trash / cabinet).

    TODO: the pick + place motions are stubs (skills.pick_object/place_object);
    the sorting decision is already real.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        objects: list[DetectedObject] = ctx.data.get("table_objects", [])
        sorted_map: dict[str, list[DetectedObject]] = ctx.data.setdefault("sorted", {})
        for o in objects:
            decision = sort_object(ctx, o)
            dest = (decision.destination if decision else None) or "cabinet"
            group = decision.cabinet_group if decision else None
            sorted_map.setdefault(dest, []).append(o)
            if pick_object(ctx, o):  # STUB -> currently False
                place_object(ctx, dest, group=group)  # STUB
        return StepResult.DONE  # partial scoring: never block the rest of the task


class ServeBreakfast(SubTask):
    """Fetch bowl+spoon (surface) and milk+cereal (cabinet), arrange on the table.

    TODO: all fetch/place motions are stubs. Breakfast layout rule: spoon next to
    bowl, cereal next to milk, clear space around them (rulebook 5.2 remark 11/12).
    """

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say("Now I will set up breakfast.")
        # TODO: pick bowl + spoon from PNP_BREAKFAST_SURFACE_POSE, milk + cereal
        # from PNP_CABINET_POSE, and place them in a standard meal arrangement on
        # the cleared dining table. Optionally pour (PourBreakfast).
        print("[pnp] TODO ServeBreakfast — fetch/arrange not implemented")
        ctx.say(prompts.BREAKFAST_DONE)
        return StepResult.DONE


class TidyExtraSurface(SubTask):
    """Move the two common objects on the extra surface into the cabinet, grouped."""

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("PNP_EXTRA_SURFACE_POSE")
        ctx.goto(x, y, h)
        objects = perceive_surface(ctx)
        ctx.say(prompts.PERCEPTION_ANNOUNCE.format(count=len(objects)))
        for o in objects:
            decision = sort_object(ctx, o)
            group = decision.cabinet_group if decision else None
            if pick_object(ctx, o):  # STUB
                place_object(ctx, "cabinet", group=group)  # STUB
        return StepResult.DONE


class OperateDishwasher(SubTask):
    """Optional goal: open/close the dishwasher door, pull/push the rack, tab slot.

    Gated by PNP_ENABLE_DISHWASHER. STUB: no autonomous appliance manipulation
    yet — the rulebook lets the robot ask the referee for help, so it does.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not _optional("PNP_ENABLE_DISHWASHER"):
            return StepResult.DONE
        ctx.say(prompts.ASK_OPEN_DISHWASHER)
        print("[pnp] TODO OperateDishwasher — appliance manipulation not implemented")
        return StepResult.DONE


class PourBreakfast(SubTask):
    """Optional goal: pour milk + cereal into the bowl. Gated by PNP_ENABLE_POUR."""

    def run(self, ctx: TaskContext) -> StepResult:
        if not _optional("PNP_ENABLE_POUR"):
            return StepResult.DONE
        ctx.say(prompts.ASK_OPEN_MILK)
        print("[pnp] TODO PourBreakfast — pouring not implemented")
        return StepResult.DONE


def build_pick_and_place_task(ctx: TaskContext) -> Task:
    """Construct the Pick and Place task. Pure: touches no hardware at build time."""
    return Task(
        "PickAndPlace",
        [
            GoToKitchen(),
            PerceiveDiningTable(),
            TidyDiningTable(),
            OperateDishwasher(),  # optional, gated
            ServeBreakfast(),
            PourBreakfast(),      # optional, gated
            TidyExtraSurface(),
        ],
        ctx,
    )
