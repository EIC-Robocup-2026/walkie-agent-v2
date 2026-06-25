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
from tasks.skills.locations import get_location_book, resolve_pose

from . import prompts
from .skills import (
    DetectedObject,
    announce_object,
    arm_enabled,
    indicate_placement,
    perceive_and_indicate_shelf,
    perceive_surface,
    pick_object,
    place_at,
    place_object,
    sort_object,
)

# Map each PnP nav waypoint to its canonical name in the shared LocationBook (the
# map editor's output). NOTE: the arm-frame place poses (PNP_PLACE_POSE_* and the
# PNP_BREAKFAST_{BOWL,SPOON,MILK,CEREAL}_POSE arm slots) are NOT here — they go
# through place_at(), not _pose(), and stay env-only.
_LOCATION_NAME = {
    "PNP_KITCHEN_POSE": "kitchen",
    "PNP_DINING_TABLE_POSE": "dining_table",
    "PNP_DISHWASHER_POSE": "dishwasher",
    "PNP_CABINET_POSE": "cabinet",
    "PNP_TRASH_BIN_POSE": "trash_bin",
    "PNP_BREAKFAST_SURFACE_POSE": "breakfast_surface",
    "PNP_EXTRA_SURFACE_POSE": "extra_surface",
}


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    """Map-frame waypoint: shared LocationBook (by name) -> *_POSE env var -> default."""
    return resolve_pose(_LOCATION_NAME.get(env_key), env_fallback=env_key, default=default)


def _has_pose(env_key: str) -> bool:
    """True if a waypoint is configured for *env_key* — in the map or the env var.

    Lets callers (skills.place_object) decide whether to drive to a furniture
    waypoint before placing, now that the pose can come from the map and not just
    the env var.
    """
    name = _LOCATION_NAME.get(env_key)
    return bool(name and get_location_book().has(name)) or bool(os.getenv(env_key))


def _optional(env_key: str) -> bool:
    return os.getenv(env_key, "0").lower() in ("1", "true", "yes")


def _list(env_key: str, default: str) -> list[str]:
    return [c.strip() for c in os.getenv(env_key, default).split(",") if c.strip()]


def _indicate_shelf_once(ctx: TaskContext) -> None:
    """Drive to the cabinet and indicate its shelf groups — at most once per run.

    Scores the 'perceive objects on a shelf and indicate the correct placement'
    line (2x30). Guarded by a blackboard flag so TidyDiningTable and
    TidyExtraSurface (both cabinet-bound) don't double-perceive the shelves.
    """
    if ctx.data.get("shelf_indicated"):
        return
    ctx.goto(*_pose("PNP_CABINET_POSE"))
    perceive_and_indicate_shelf(ctx)
    ctx.data["shelf_indicated"] = True


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
        if ctx.goto(x, y, h):
            ctx.score("navigate_table")  # reached the dining table (claimed)
        objects = perceive_surface(ctx)
        ctx.data["table_objects"] = objects
        ctx.say(prompts.PERCEPTION_ANNOUNCE.format(count=len(objects)))
        for o in objects:
            announce_object(ctx, o)  # 'correctly recognize an object' (scoresheet 5.2)
            print(f"[pnp] perceived {o.class_name} ({o.confidence:.2f}) @ {o.world_xy}")
        return StepResult.DONE


class TidyDiningTable(SubTask):
    """Sort each table object, indicate its placement, then (arm-gated) place it.

    Two passes so the score budget is decoupled from the arm. **Pass 1 always
    runs**: sort each object and *communicate* its intended placement to the
    referee (recognize + indicate-placement, rulebook remark 16), plus one shelf
    perception for cabinet-bound items — this is the entire non-arm budget and it
    banks whether or not the arm later moves anything. **Pass 2 is gated** on
    `PNP_ARM_CALIBRATED`: the robot returns to the table, picks each object, and
    places it at its destination (map-frame `world_xyz` keeps each grasp
    reachable; `place_object` degrades to a drop-in-front until poses are set).
    """

    def run(self, ctx: TaskContext) -> StepResult:
        objects: list[DetectedObject] = ctx.data.get("table_objects", [])
        sorted_map: dict[str, list[DetectedObject]] = ctx.data.setdefault("sorted", {})
        table = _pose("PNP_DINING_TABLE_POSE")

        # Pass 1 (always — the non-arm scoring budget): sort each object and
        # communicate its intended placement to the referee (rulebook remark 16 +
        # scoresheet 'indicate the correct placement'). Runs even with the arm gated.
        plans: list[tuple[DetectedObject, str, str | None]] = []
        for o in objects:
            decision = sort_object(ctx, o)
            dest = (decision.destination if decision else None) or "cabinet"
            group = decision.cabinet_group if decision else None
            sorted_map.setdefault(dest, []).append(o)
            indicate_placement(ctx, o, dest, group)
            plans.append((o, dest, group))
        # Shelf perception + indication for cabinet-bound items (scoresheet 2x30).
        if any(dest == "cabinet" for _, dest, _ in plans):
            _indicate_shelf_once(ctx)

        # Pass 2 (arm-gated): physically pick + place each object. Skipped wholesale
        # until the arm skill lands (PNP_ARM_CALIBRATED=1) — the placement was
        # already *indicated* above, so the non-arm score is banked regardless.
        if not arm_enabled():
            return StepResult.DONE
        for o, dest, group in plans:
            ctx.goto(*table)  # back to the table so the object is in reach
            if not pick_object(ctx, o):
                continue
            ctx.score("pick_transport")    # arm: picked an object for transport
            ctx.score("first_pick_bonus")  # one-time (clamped to 1)
            if place_object(ctx, dest, group=group):
                ctx.score("place_designated")  # arm: placed at its destination
                if dest == "dishwasher":
                    ctx.score("place_dishwasher")
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
    if target is None:
        ctx.say(prompts.BREAKFAST_NOT_FOUND.format(obj=obj_name))
        return False
    announce_object(ctx, target)  # 'recognize' scores even when the arm is gated
    if not arm_enabled():
        print(f"[pnp] breakfast: arm gated; indicated {obj_name} only (no fetch)")
        return False
    if not pick_object(ctx, target):
        ctx.say(prompts.BREAKFAST_NOT_FOUND.format(obj=obj_name))
        return False
    ctx.score("pick_transport")
    ctx.score("first_pick_bonus")
    tx, ty, th = _pose("PNP_DINING_TABLE_POSE")
    ctx.goto(tx, ty, th)
    if place_at(ctx, os.getenv(slot_env, "")):
        ctx.score("place_designated")
        return True
    return False


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

        # Pass 1 (always): recognize + indicate the cabinet placement for each object.
        plans: list[tuple[DetectedObject, str | None]] = []
        for o in objects:
            announce_object(ctx, o)
            decision = sort_object(ctx, o)
            group = decision.cabinet_group if decision else None
            indicate_placement(ctx, o, "cabinet", group)
            plans.append((o, group))
        if plans:
            _indicate_shelf_once(ctx)

        # Pass 2 (arm-gated): move each common object into the cabinet, grouped.
        if not arm_enabled():
            return StepResult.DONE
        for o, group in plans:
            ctx.goto(*surface)  # back to the surface so the object is in reach
            if not pick_object(ctx, o):
                continue
            ctx.score("pick_transport")
            ctx.score("first_pick_bonus")
            if place_object(ctx, "cabinet", group=group):
                ctx.score("place_designated")
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


class NavTour(SubTask):
    """Visit each configured waypoint and announce arrival — a localization /
    waypoint sanity check with no perception or manipulation.

    The first thing to validate on a freshly-mapped arena: that every PNP_*_POSE
    is set and reachable before any perception/arm bring-up depends on it.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        tour = [
            ("kitchen", "PNP_KITCHEN_POSE"),
            ("dining table", "PNP_DINING_TABLE_POSE"),
            ("dishwasher", "PNP_DISHWASHER_POSE"),
            ("cabinet", "PNP_CABINET_POSE"),
            ("trash bin", "PNP_TRASH_BIN_POSE"),
            ("breakfast surface", "PNP_BREAKFAST_SURFACE_POSE"),
            ("extra surface", "PNP_EXTRA_SURFACE_POSE"),
        ]
        for label, key in tour:
            x, y, h = _pose(key)
            reached = ctx.goto(x, y, h)
            line = prompts.NAV_TOUR_ARRIVE if reached else prompts.NAV_TOUR_FAIL
            ctx.say(line.format(place=label))
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


# --- Isolated slices for step-by-step on-robot bring-up (selected by PNP_SLICE).
# Order = rough bring-up order: prove the waypoints, then perception, then the
# perception+sort+indicate scoring path, then the whole flow. Every slice runs
# arm-gated unless PNP_ARM_CALIBRATED=1 (see tasks/PickAndPlace/skills.py).
def build_nav_slice(ctx: TaskContext) -> Task:
    """Waypoint tour only — validate localization + every PNP_*_POSE."""
    return Task("PickAndPlace:nav", [NavTour()], ctx)


def build_perceive_slice(ctx: TaskContext) -> Task:
    """Drive to the table, perceive + announce each recognized object. No sort/arm."""
    return Task("PickAndPlace:perceive", [PerceiveDiningTable()], ctx)


def build_sort_slice(ctx: TaskContext) -> Task:
    """Perceive, then sort + indicate the correct placement for each object.

    The full non-arm scoring path (recognize + indicate placement + shelf
    indication) without the whole-task length. With the arm gated this is the
    primary score-earning rehearsal.
    """
    return Task("PickAndPlace:sort", [PerceiveDiningTable(), TidyDiningTable()], ctx)


def build_breakfast_slice(ctx: TaskContext) -> Task:
    """Recognize the breakfast items at their sources + announce the layout plan."""
    return Task("PickAndPlace:breakfast", [ServeBreakfast()], ctx)
