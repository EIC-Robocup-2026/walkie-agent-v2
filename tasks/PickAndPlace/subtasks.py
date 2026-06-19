"""Pick and Place subtasks + the build_pick_and_place_task factory (rulebook 5.2).

Mirrors tasks/HRI/subtasks.py. The flow follows the rulebook procedure: enter
the kitchen, tidy the dining table (tableware/cutlery -> dishwasher, trash ->
bin, other -> cabinet grouped), serve breakfast, tidy the extra surface.

Manipulation is real: each pick/place commands the arm and gripper (see
tasks/PickAndPlace/skills.py). The deliberately-stubbed part is grasp *planning*
(plan_grasp/plan_place return the hand pose from config + the object's 3D point)
and the appliance-manipulation optional goals (open/close dishwasher, pour),
which ask the referee for help. Every non-critical step degrades rather than
crashes — partial scoring is allowed, so a failed step logs and the task moves
on (see tasks/base.py).

Blackboard layout (ctx.data):
    table_objects:    list[DetectedObject]  # from PerceiveDiningTable
    sorted:           {destination: [DetectedObject, ...]}  # after TidyDiningTable
    dishwasher_items: int  # count placed inside (gates CloseDishwasher)
"""

from __future__ import annotations

import os

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import (
    DetectedObject,
    perceive_surface,
    pick_object,
    place_at,
    place_object,
    sort_object,
)


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    """Parse a map-frame waypoint 'x,y,heading_rad' from the environment."""
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _optional(env_key: str) -> bool:
    return os.getenv(env_key, "0").lower() in ("1", "true", "yes")


def _list(env_key: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env_key, default).split(",") if c.strip()]


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

    The robot can only carry an object at a time, so each iteration returns to
    the table, picks one object, then drives to its destination and releases.
    Object positions are stored map-frame (`world_xyz`), so returning to the
    table pose keeps each grasp reachable. Sorting and motion are real;
    `place_object` degrades to a drop-in-front until destination poses are set.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        objects: list[DetectedObject] = ctx.data.get("table_objects", [])
        sorted_map: dict[str, list[DetectedObject]] = ctx.data.setdefault("sorted", {})
        table = _pose("PNP_DINING_TABLE_POSE")
        for o in objects:
            decision = sort_object(ctx, o)
            dest = (decision.destination if decision else None) or "cabinet"
            group = decision.cabinet_group if decision else None
            sorted_map.setdefault(dest, []).append(o)
            ctx.goto(*table)  # back to the table so the object is in reach
            if pick_object(ctx, o) and place_object(ctx, dest, group=group):
                if dest == "dishwasher":
                    # Drives CloseDishwasher (closing only scores after >=1 item in).
                    ctx.data["dishwasher_items"] = ctx.data.get("dishwasher_items", 0) + 1
        return StepResult.DONE  # partial scoring: never block the rest of the task


def _fetch_and_place(
    ctx: TaskContext, source_pose_key: str, classes: list[str], obj_name: str, slot_env: str
) -> bool:
    """Fetch the first item matching *classes* from a surface, place it at *slot_env*.

    Drives to the source furniture, perceives it, picks the best-matching object,
    drives to the dining table, and releases at the configured breakfast slot
    pose. Best-effort: returns False (and announces) on any miss, never raises.
    """
    ctx.say(prompts.BREAKFAST_FETCH.format(obj=obj_name))
    x, y, h = _pose(source_pose_key)
    ctx.goto(x, y, h)
    objects = perceive_surface(ctx, classes=classes)
    target = max(objects, key=lambda o: o.confidence) if objects else None
    if target is None or not pick_object(ctx, target):
        ctx.say(prompts.BREAKFAST_NOT_FOUND.format(obj=obj_name))
        return False
    tx, ty, th = _pose("PNP_DINING_TABLE_POSE")
    ctx.goto(tx, ty, th)
    return place_at(ctx, os.getenv(slot_env, ""))


class ServeBreakfast(SubTask):
    """Fetch bowl+spoon (surface) and milk+cereal (cabinet), arrange on the table.

    Breakfast layout rule (rulebook 5.2 remark 11/12): the spoon next to the
    bowl, the cereal next to the milk, with clear space around them — encoded as
    four per-slot arm poses (PNP_BREAKFAST_*_POSE). Every fetch/place is
    best-effort so a single miss never blocks the rest of breakfast.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.BREAKFAST_START)
        surface_classes = _list("PNP_BREAKFAST_SURFACE_CLASSES", "bowl,spoon")
        cabinet_classes = _list("PNP_BREAKFAST_CABINET_CLASSES", "milk,cereal,box,carton")
        # (item label, source furniture pose, detector classes, slot pose env)
        items = [
            ("bowl", "PNP_BREAKFAST_SURFACE_POSE", surface_classes, "PNP_BREAKFAST_BOWL_POSE"),
            ("spoon", "PNP_BREAKFAST_SURFACE_POSE", surface_classes, "PNP_BREAKFAST_SPOON_POSE"),
            ("milk", "PNP_CABINET_POSE", cabinet_classes, "PNP_BREAKFAST_MILK_POSE"),
            ("cereal", "PNP_CABINET_POSE", cabinet_classes, "PNP_BREAKFAST_CEREAL_POSE"),
        ]
        for name, src, classes, slot in items:
            _fetch_and_place(ctx, src, classes, name, slot)
        ctx.say(prompts.BREAKFAST_DONE)
        return StepResult.DONE


class TidyExtraSurface(SubTask):
    """Move the common objects on the extra surface into the cabinet, grouped."""

    def run(self, ctx: TaskContext) -> StepResult:
        surface = _pose("PNP_EXTRA_SURFACE_POSE")
        ctx.goto(*surface)
        objects = perceive_surface(ctx)
        ctx.say(prompts.PERCEPTION_ANNOUNCE.format(count=len(objects)))
        for o in objects:
            decision = sort_object(ctx, o)
            group = decision.cabinet_group if decision else None
            ctx.goto(*surface)  # back to the surface so the object is in reach
            if pick_object(ctx, o):
                place_object(ctx, "cabinet", group=group)
        return StepResult.DONE


class OpenDishwasher(SubTask):
    """Optional goal: get the dishwasher open *before* loading tableware.

    Gated by PNP_ENABLE_DISHWASHER. STUB: no autonomous appliance manipulation
    yet — the rulebook lets the robot ask the referee for help, so it does.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not _optional("PNP_ENABLE_DISHWASHER"):
            return StepResult.DONE
        ctx.say(prompts.ASK_OPEN_DISHWASHER)
        print("[pnp] OpenDishwasher — autonomous door not implemented; asked referee")
        return StepResult.DONE


class CloseDishwasher(SubTask):
    """Optional goal: close the dishwasher *after* at least one item is loaded.

    Closing only scores once >=1 item has been placed inside (rulebook 5.2
    remark 8), so this no-ops unless TidyDiningTable loaded something. STUB: asks
    the referee to close the door.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if not _optional("PNP_ENABLE_DISHWASHER"):
            return StepResult.DONE
        if ctx.data.get("dishwasher_items", 0) <= 0:
            print("[pnp] CloseDishwasher — nothing loaded; skipping close")
            return StepResult.DONE
        ctx.say(prompts.ASK_CLOSE_DISHWASHER)
        print("[pnp] CloseDishwasher — autonomous door not implemented; asked referee")
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
            OpenDishwasher(),     # optional, gated — open before loading tableware
            TidyDiningTable(),
            CloseDishwasher(),    # optional, gated — close after >=1 item loaded
            ServeBreakfast(),
            PourBreakfast(),      # optional, gated
            TidyExtraSurface(),
        ],
        ctx,
    )
