"""Restaurant subtasks + the build_restaurant_task factory (rulebook 5.5).

Mirrors tasks/HRI/subtasks.py. Customers call dynamically and any number of
times, so the serving loop cannot be pre-listed as fixed steps the way HRI lists
two guests — ServeCustomers loops until it has served the target number (or runs
out of callers). Each iteration follows the rulebook: detect a waving customer ->
navigate to their table (or clearly identify them) -> take + confirm the order ->
relay it to the barman -> collect the items from the kitchen-bar and serve them
one at a time.

Gesture detection, customer approach, order dialogue and item manipulation are
real (the grasp planner behind collect/serve is a stub — see tasks/manipulation.py
— but the arm motion is real). The unattached-tray optional goal is a gated stub.
Restart handling (rulebook 5.5.1) is operator-driven and out of scope.

Blackboard layout (ctx.data):
    served: list[list[str]]   # the orders delivered, for logging
"""

from __future__ import annotations

import os

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import (
    detect_calling_customer,
    identify_customer,
    navigate_to_customer,
    pick_bar_item,
    serve_item,
    take_order,
)


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


class GoToStart(SubTask):
    """Position next to the kitchen-bar, facing the dining area (rulebook setup)."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("RESTAURANT_KITCHEN_BAR_POSE")
        return StepResult.DONE if ctx.goto(x, y, h) else StepResult.RETRY


class ServeCustomers(SubTask):
    """Serve up to RESTAURANT_TARGET_CUSTOMERS callers, one full cycle each.

    One cycle = detect a waving customer -> approach (or identify) -> take +
    confirm order -> relay to the barman -> for each item: pick at the bar,
    navigate to the customer, serve, return to the bar. Detection, approach and
    dialogue are real; the grasp planner behind pick/serve is a stub but the arm
    motion is real, so each leg degrades gracefully (partial scoring).
    """

    def run(self, ctx: TaskContext) -> StepResult:
        target = int(os.getenv("RESTAURANT_TARGET_CUSTOMERS", "2"))
        served: list[list[str]] = ctx.data.setdefault("served", [])
        attempts = 0
        max_attempts = target + int(os.getenv("RESTAURANT_EXTRA_ATTEMPTS", "3"))
        bar = _pose("RESTAURANT_KITCHEN_BAR_POSE")
        if os.getenv("RESTAURANT_USE_TRAY", "0").lower() in ("1", "true", "yes"):
            # Optional goal (2x200): place items on an unattached tray, carry the
            # tray, unload at the table. Not implemented — fall back to per-item.
            print("[restaurant] RESTAURANT_USE_TRAY set, but tray transport is not "
                  "implemented; serving items individually instead")

        while len(served) < target and attempts < max_attempts:
            attempts += 1
            customer = detect_calling_customer(ctx)
            if customer is None:
                ctx.say(prompts.NO_CUSTOMER)
                continue

            # Approach the table; if the drive fails, clearly identify the person
            # for partial points and still take the order from where we are.
            if not navigate_to_customer(ctx, customer):
                identify_customer(ctx, customer)
            else:
                ctx.rotate_to(customer.heading)  # face them for eye contact

            items = take_order(ctx)
            if not items:
                continue

            # Relay the whole order to the barman at the kitchen-bar (once).
            ctx.goto(*bar)
            ctx.say(prompts.RELAY_TO_BARMAN.format(items=", ".join(items)))

            # Collect + serve one item per trip (single-arm carry).
            for item in items:
                ctx.goto(*bar)
                ctx.say(prompts.COLLECTING_ITEM.format(item=item))
                if not pick_bar_item(ctx, item):
                    continue
                navigate_to_customer(ctx, customer)
                if serve_item(ctx):
                    ctx.say(prompts.SERVE_ITEM.format(item=item))

            served.append(items)  # logged even if a leg degraded (order was taken)
            ctx.goto(*bar)  # back to the bar for the next caller

        ctx.say(prompts.ALL_DONE)
        print(f"[restaurant] orders handled: {served}")
        return StepResult.DONE


def build_restaurant_task(ctx: TaskContext) -> Task:
    """Construct the Restaurant task. Pure: touches no hardware at build time."""
    return Task(
        "Restaurant",
        [
            GoToStart(),
            ServeCustomers(),
        ],
        ctx,
    )
