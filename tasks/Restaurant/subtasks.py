"""Restaurant subtasks + task factories (rulebook 5.5).

Phase 0 (this slice) is REAL: GoToStart -> ScanForCaller -> ApproachCustomer
(scan the dining area, detect a waving customer from pose keypoints, lift them to
a map point, drive to a stand-off facing them). The downstream serve flow
(take order -> relay to barman -> pick -> serve) runs too, with order-taking and
relay real and manipulation as Phase-2 stubs that degrade.

Two factories:
- ``build_phase0_slice``  — GoToStart -> ScanAndApproach, for on-robot bring-up of
  just the detection + approach skills (this box can't dry-run reactive loops).
- ``build_restaurant_task`` — the full MVP serial loop (one customer at a time;
  the LLM interleave scheduler is a later phase, rulebook bonus only).

Blackboard layout (ctx.data):
    bar_anchor: {"x","y","heading"}   # the Kitchen-bar pose, captured at GoToStart
    orders:     {id: Order}           # every order seen this run
    target:     Caller                # the caller ApproachCustomer is heading to
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum, auto

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .skills import (
    approach_to_standoff,
    capture_appearance,
    collect_items,
    nearest_caller,
    relay_to_barman,
    return_to_bar,
    return_to_customer,
    scan_for_callers,
    serve_order,
    take_order,
)


class OrderStatus(Enum):
    DETECTED = auto()
    APPROACHED = auto()
    ORDERED = auto()
    RELAYED = auto()
    PICKED = auto()
    SERVED = auto()
    FAILED = auto()


@dataclass
class Order:
    """One customer's order through the serve pipeline (see design doc §6.1)."""

    id: int
    world_xy: tuple[float, float] | None
    bearing: float | None
    items: list[str] = field(default_factory=list)
    appearance: str | None = None  # caption, to re-identify the customer on return
    status: OrderStatus = OrderStatus.DETECTED


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _int(env_key: str, default: str) -> int:
    return int(os.getenv(env_key, default))


# ---------------------------------------------------------------------------
# Phase 0 states
# ---------------------------------------------------------------------------
class GoToStart(SubTask):
    """Go to the Kitchen-bar start pose and remember it as the bar anchor.

    The bar anchor is where relayed orders are placed and picked; later phases
    re-acquire the bar/barman visually on return rather than trusting this point
    blindly (design doc §5.1).
    """

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("RESTAURANT_KITCHEN_BAR_POSE")
        ok = ctx.goto(x, y, h)
        # Capture the real pose we ended at as the bar anchor (falls back to config).
        pose = ctx.current_pose()
        ctx.data["bar_anchor"] = pose if pose.get("x") or pose.get("y") else {"x": x, "y": y, "heading": h}
        return StepResult.DONE if ok else StepResult.RETRY


class ScanAndApproach(SubTask):
    """Phase 0 core: sweep for a waving customer, then approach to a stand-off.

    Re-sweeps on an empty scan (callers come and go); aborts to the next step
    only after exhausting retries. Stores the chosen caller on ctx.data["target"].
    """

    max_retries = 2

    def run(self, ctx: TaskContext) -> StepResult:
        callers = scan_for_callers(ctx)
        target = nearest_caller(ctx, callers)
        if target is None:
            ctx.say(prompts.NO_CUSTOMER)
            return StepResult.RETRY
        ctx.data["target"] = target
        if approach_to_standoff(ctx, target.world_xy):
            return StepResult.DONE
        # Reached no nav goal (no odom fix / nav refused) — re-sweep and retry.
        return StepResult.RETRY


# ---------------------------------------------------------------------------
# Full MVP serial loop (Phase 0 real; order/relay real; pick/serve = Phase 2 stubs)
# ---------------------------------------------------------------------------
class ServeCustomers(SubTask):
    """Serve up to RESTAURANT_TARGET_CUSTOMERS callers, one full cycle each.

    One cycle = scan -> approach -> take order (gaze) -> relay at the bar
    (re-acquire barman) -> pick -> return to customer (re-acquire) -> serve.
    Detection/approach/order/relay are real; pick/serve degrade (Phase 2).
    Serial by design — the interleave scheduler is a later phase (bonus only).
    """

    def run(self, ctx: TaskContext) -> StepResult:
        target = _int("RESTAURANT_TARGET_CUSTOMERS", "2")
        max_attempts = target + _int("RESTAURANT_EXTRA_ATTEMPTS", "3")
        orders: dict[int, Order] = ctx.data.setdefault("orders", {})
        served = 0
        attempts = 0

        while served < target and attempts < max_attempts:
            attempts += 1

            # 1. Detect + approach (Phase 0).
            caller = nearest_caller(ctx, scan_for_callers(ctx))
            if caller is None:
                ctx.say(prompts.NO_CUSTOMER)
                continue
            order = Order(id=len(orders) + 1, world_xy=caller.world_xy, bearing=caller.bearing)
            orders[order.id] = order
            if not approach_to_standoff(ctx, caller.world_xy):
                order.status = OrderStatus.FAILED
                continue
            order.status = OrderStatus.APPROACHED
            order.appearance = capture_appearance(ctx, caller.world_xy)  # for re-ID/logging

            # 2. Take + confirm the order (real), re-facing the customer (gaze).
            items = take_order(ctx, world_xy=order.world_xy)
            if not items:
                order.status = OrderStatus.FAILED
                continue
            order.items = items
            order.status = OrderStatus.ORDERED

            # 3. Relay at the bar — go_to the anchor, then re-acquire the barman.
            return_to_bar(ctx)
            if relay_to_barman(ctx, items):
                order.status = OrderStatus.RELAYED

            # 4. Pick + serve (Phase 2 stubs — degrade, order still counts as handled).
            if collect_items(ctx, items):
                order.status = OrderStatus.PICKED
                # Re-acquire the customer visually (don't trust the stale point, §5.1).
                fresh = return_to_customer(ctx, order.world_xy) if order.world_xy else None
                if fresh is not None:
                    order.world_xy = fresh
                    if serve_order(ctx, items):
                        order.status = OrderStatus.SERVED
                else:
                    ctx.say(prompts.SERVE_NO_CUSTOMER)

            served += 1
            return_to_bar(ctx)  # back to the bar for the next caller

        ctx.say(prompts.ALL_DONE)
        print("[restaurant] orders: " + ", ".join(
            f"#{o.id}={o.status.name}({o.items})" for o in orders.values()))
        return StepResult.DONE


class ServeCustomersBatched(SubTask):
    """Phase 3 (opt-in): take several orders in one sweep, then deliver each.

    The rulebook explicitly allows taking/placing several orders before delivery.
    Batching the order-TAKING (one scan, approach the nearest few, take all their
    orders) trims walking and fits more customers into the 15-min limit. Delivery
    is still per-order (one gripper can't carry a multi-item order without a tray —
    see skills.transport_with_tray). Pure scheduling logic; pick/serve degrade as
    in Phase 2. Selected by RESTAURANT_BATCH=1.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        target = _int("RESTAURANT_TARGET_CUSTOMERS", "2")
        batch_size = max(1, _int("RESTAURANT_BATCH_SIZE", "2"))
        orders: dict[int, Order] = ctx.data.setdefault("orders", {})

        # Phase A — gather a batch of orders (nearest callers first).
        callers = scan_for_callers(ctx)
        if not callers:
            ctx.say(prompts.NO_CUSTOMER)
            return StepResult.DONE
        p = ctx.current_pose()
        callers.sort(key=lambda c: math.hypot(c.world_xy[0] - p["x"], c.world_xy[1] - p["y"]))
        taken: list[Order] = []
        for caller in callers[:min(batch_size, target)]:
            order = Order(id=len(orders) + 1, world_xy=caller.world_xy, bearing=caller.bearing)
            orders[order.id] = order
            if not approach_to_standoff(ctx, caller.world_xy):
                order.status = OrderStatus.FAILED
                continue
            order.status = OrderStatus.APPROACHED
            order.appearance = capture_appearance(ctx, caller.world_xy)
            items = take_order(ctx, world_xy=order.world_xy)
            if not items:
                order.status = OrderStatus.FAILED
                continue
            order.items, order.status = items, OrderStatus.ORDERED
            taken.append(order)

        if not taken:
            ctx.say(prompts.ALL_DONE)
            return StepResult.DONE

        # Phase B — deliver each (per-order bar trip; tray would allow one trip).
        for order in taken:
            return_to_bar(ctx)
            if relay_to_barman(ctx, order.items):
                order.status = OrderStatus.RELAYED
            if collect_items(ctx, order.items):
                order.status = OrderStatus.PICKED
                fresh = return_to_customer(ctx, order.world_xy) if order.world_xy else None
                if fresh is not None:
                    order.world_xy = fresh
                    if serve_order(ctx, order.items):
                        order.status = OrderStatus.SERVED
                else:
                    ctx.say(prompts.SERVE_NO_CUSTOMER)

        ctx.say(prompts.ALL_DONE)
        print("[restaurant] batched orders: " + ", ".join(
            f"#{o.id}={o.status.name}({o.items})" for o in orders.values()))
        return StepResult.DONE


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_phase0_slice(ctx: TaskContext) -> Task:
    """Phase 0 only: GoToStart -> ScanAndApproach. For on-robot bring-up."""
    return Task("Restaurant-Phase0", [GoToStart(), ScanAndApproach()], ctx)


def build_restaurant_task(ctx: TaskContext) -> Task:
    """Full task. Serial loop by default; batched order-taking when RESTAURANT_BATCH=1.

    Pure: touches no hardware at build time.
    """
    batched = os.getenv("RESTAURANT_BATCH", "0").lower() in ("1", "true", "yes")
    serve = ServeCustomersBatched() if batched else ServeCustomers()
    return Task("Restaurant", [GoToStart(), serve], ctx)
