"""Restaurant subtasks + the build_restaurant_task factory (rulebook 5.5).

PLACEHOLDER scaffold mirroring tasks/HRI/subtasks.py. Customers call dynamically
and any number of times, so the serving loop cannot be pre-listed as fixed steps
the way HRI lists two guests — ServeCustomers loops until it has served the
target number (or runs out of callers). Each iteration follows the rulebook:
detect a calling customer -> navigate to their table -> take + confirm the order
-> relay it to the barman -> collect the items from the kitchen-bar -> deliver.

Gesture detection, online navigation and manipulation are honest stubs that
degrade (partial scoring is allowed). Restart handling (rulebook 5.5.1) is an
operator-driven concern and is out of scope for this scaffold.

Blackboard layout (ctx.data):
    served: list[list[str]]   # the orders delivered, for logging
"""

from __future__ import annotations

import os

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import (
    collect_items,
    detect_calling_customer,
    navigate_to_customer,
    serve_order,
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

    One cycle = detect -> navigate -> order -> relay to barman -> collect ->
    deliver, then back to the kitchen-bar. The manipulation/nav legs are stubs;
    detection + order dialogue are real-ish.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        target = int(os.getenv("RESTAURANT_TARGET_CUSTOMERS", "2"))
        served: list[list[str]] = ctx.data.setdefault("served", [])
        attempts = 0
        max_attempts = target + int(os.getenv("RESTAURANT_EXTRA_ATTEMPTS", "3"))

        while len(served) < target and attempts < max_attempts:
            attempts += 1
            heading = detect_calling_customer(ctx)
            if heading is None:
                ctx.say(prompts.NO_CUSTOMER)
                continue

            navigate_to_customer(ctx, heading)  # STUB -> partial: still take order
            items = take_order(ctx)
            if not items:
                continue

            # Relay to the barman at the kitchen-bar.
            x, y, h = _pose("RESTAURANT_KITCHEN_BAR_POSE")
            ctx.goto(x, y, h)
            ctx.say(prompts.RELAY_TO_BARMAN.format(items=", ".join(items)))

            # Collect + deliver (both STUBs today).
            if collect_items(ctx, items):
                navigate_to_customer(ctx, heading)
                if serve_order(ctx, items):
                    ctx.say(prompts.SERVE_ANNOUNCE.format(items=", ".join(items)))

            served.append(items)  # logged even if delivery degraded (order was taken)
            ctx.goto(x, y, h)  # back to the bar for the next caller

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
