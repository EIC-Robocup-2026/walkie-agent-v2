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
import time
from dataclasses import dataclass, field
from enum import Enum, auto

from tasks.base import StepResult, SubTask, Task, TaskContext
from tasks.skills import (
    detect_surfaces,
    get_object_grasp_pos,
    held_arms,
    pick_object,
    place_object,
    recall_held_object,
)
from tasks.skills.geometry import parse_pose

from . import prompts
from .skills import (
    _arm_calibrated,
    approach_customer,
    capture_appearance,
    exclude_handled,
    find_first_caller,
    nearest_caller,
    relay_to_barman,
    return_to_bar,
    return_to_customer,
    scan_for_callers,
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
    """Bar pose from the env var ONLY — never the shared map (rulebook 5.5).

    Restaurant is deliberately decoupled from the pre-map LocationBook: the arena
    isn't surveyed in advance, so the bar is anchored on the robot's pose at
    GoToStart, not a stored waypoint. An explicit ``RESTAURANT_KITCHEN_BAR_POSE``
    stays available as a manual bring-up override (drive to a fixed pose); it never
    falls through to a ``kitchen_bar`` map waypoint, even if one exists for GPSR.
    """
    raw = os.getenv(env_key)
    return parse_pose(raw if raw and raw.strip() else default)


def _int(env_key: str, default: str) -> int:
    return int(os.getenv(env_key, default))


def _f(env_key: str, default: str) -> float:
    return float(os.getenv(env_key, default))


def _b(env_key: str, default: str) -> bool:
    return os.getenv(env_key, default).strip().lower() in ("1", "true", "yes", "on")


def _next_caller(ctx: TaskContext, blocked: list[tuple[float, float]], radius: float):
    """Pick the next customer to serve.

    Default (``RESTAURANT_APPROACH_FIRST=1``): stop at the FIRST waver seen during the
    sweep and head straight over (``find_first_caller`` — exclusion of handled/given-up
    spots happens in-sweep). Set 0 for the original behaviour: finish the whole arc, drop
    handled spots, then take the nearest. (Batched order-taking always does the full sweep.)
    """
    if _b("RESTAURANT_APPROACH_FIRST", "1"):
        return find_first_caller(ctx, blocked, radius)
    return nearest_caller(ctx, exclude_handled(scan_for_callers(ctx), blocked, radius))


def _tray_mode() -> bool:
    """TRAY mode (default): the barman loads items onto the robot's tray and the
    customer takes them off — no arm grasp/place. 0 = the gripper pick/place path."""
    return os.getenv("RESTAURANT_TRAY_MODE", "1").strip().lower() in ("1", "true", "yes", "on")


def _await_handoff(ctx: TaskContext, ask_prompt: str, wait_key: str) -> None:
    """Wait for a human to load/unload the tray: ask for a spoken go-ahead, and on
    silence fall back to a fixed dwell so a quiet human never stalls the run."""
    reply = ctx.ask(ask_prompt, retries=1)
    if not reply:
        dwell = _f(wait_key, "5.0")
        if dwell > 0:
            time.sleep(dwell)


def _odom_fix(ctx: TaskContext) -> dict | None:
    """A genuine odometry fix, or None on failure (never the zeros fallback)."""
    try:
        return ctx.walkie.status.get_position() or None
    except Exception:
        return None


def _pick_and_serve(ctx: TaskContext, order: Order) -> None:
    """Phase 2: per-item pick at the bar -> carry to the customer -> place on their table.

    One gripper carries one object, so delivery is a per-item round trip (pick ->
    re-acquire the customer -> place -> back to the bar), NOT pick-all-then-serve.
    Per-item partial credit (pick + serve are scored per object): a failed item
    doesn't forfeit the others. Uses the grasp/place skills (tasks.skills.pick_object
    / place_object): pick records what it grabbed + how high it sat above its support
    surface, place reconstructs that height on the customer's table so it lands upright.
    Gated off until RESTAURANT_ARM_CALIBRATED=1 so nav/HRI can be rehearsed without arm
    motion. Shared by the serial and batched loops so they can't drift apart.
    """
    if not order.items:
        return
    if not _arm_calibrated():
        print("[restaurant] pick/serve: arm UNCALIBRATED — set RESTAURANT_ARM_CALIBRATED=1 "
              "to enable the grasp/place pipeline; skipping manipulation")
        ctx.say(prompts.PICK_NOT_AVAILABLE)
        return

    served: list[str] = []
    for item in order.items:
        # 1. Pick one item — we arrive (and return between items) at the bar anchor.
        if not pick_object(
            ctx, prompts=[item], arm="auto",
            pregrasp_standoff_m=0.2, approach_preference="side", approach_weight=4.0,
        ):
            print(f"[restaurant] could not pick {item!r}; trying the next item")
            continue
        order.status = OrderStatus.PICKED
        ctx.score("pickup_items")      # arm: picked an item from the bar
        ctx.score("first_pick_bonus")  # one-time (clamped to 1)

        # 2. Re-acquire the customer visually rather than trusting the stale point (§5.1).
        fresh = return_to_customer(ctx, order.world_xy) if order.world_xy else None
        if fresh is None:
            # Still holding the item and lost the customer — give up the rest of the order.
            ctx.say(prompts.SERVE_NO_CUSTOMER)
            return_to_bar(ctx)
            break
        order.world_xy = fresh
        ctx.score("return_table")      # returned to the customer table with the order

        # 3. Place it on a clear spot of the table in front of the customer. place_object
        #    auto-picks the nearest reachable surface; pass surface=/target_xy= to steer it.
        if place_object(ctx):
            served.append(item)
            ctx.score("serve_order")       # arm: served the item to the customer
            ctx.score("first_place_bonus")  # one-time (clamped to 1)
            ctx.say(prompts.SERVE_ANNOUNCE.format(items=item))
        else:
            print(f"[restaurant] reached the customer but could not place {item!r}")

        # 4. Back to the bar for the next item.
        return_to_bar(ctx)

    if served:
        order.status = OrderStatus.SERVED
    print(f"[restaurant] order #{order.id}: served {served} of {order.items}")


def _serve_with_tray(ctx: TaskContext, order: Order) -> None:
    """Tray mode: the robot holds its installed tray; the barman loads the items onto
    it and the customer takes them off — no arm grasp/place.

    The whole order rides on one tray, so this is a SINGLE round trip (load at the
    bar -> carry -> the customer unloads), not the per-item trips the one-gripper path
    needs. Scores the same lines as :func:`_pick_and_serve` (per item) so the tally is
    comparable; assumes the robot is already at the bar (``_deliver_order`` drove here).
    """
    if not order.items:
        return
    items_str = ", ".join(order.items)
    n = len(order.items)

    # 1. At the bar: ask the barman to load the tray, then wait for the handoff.
    ctx.say(prompts.TRAY_ASK_BARMAN.format(items=items_str))
    _await_handoff(ctx, prompts.TRAY_LOADED_CONFIRM, "RESTAURANT_TRAY_LOAD_WAIT_SEC")
    order.status = OrderStatus.PICKED
    ctx.score("pickup_items", n)    # items received onto the tray (per item)
    ctx.score("first_pick_bonus")   # one-time (clamped to 1 by the sheet)

    # 2. Carry the whole order to the customer in ONE trip; re-acquire them (§5.1).
    fresh = return_to_customer(ctx, order.world_xy) if order.world_xy else None
    if fresh is None:
        ctx.say(prompts.SERVE_NO_CUSTOMER)
        return_to_bar(ctx)
        return
    order.world_xy = fresh
    ctx.score("return_table")       # returned to the customer's table with the order

    # 3. Present the tray; the customer takes their items off it.
    ctx.say(prompts.TRAY_PRESENT_CUSTOMER.format(items=items_str))
    _await_handoff(ctx, prompts.TRAY_TAKEN_CONFIRM, "RESTAURANT_TRAY_UNLOAD_WAIT_SEC")
    order.status = OrderStatus.SERVED
    ctx.score("serve_order", n)     # items served off the tray (per item)
    ctx.score("first_place_bonus")  # one-time (clamped to 1 by the sheet)
    ctx.say(prompts.SERVE_ANNOUNCE.format(items=items_str))

    # 4. Back to the bar for the next customer.
    return_to_bar(ctx)
    print(f"[restaurant] order #{order.id}: served {order.items} via tray")


def _take_one_order(ctx: TaskContext, caller, orders: dict[int, Order]) -> Order | None:
    """Approach one caller, capture + confirm their order; return the Order or None.

    The per-customer half of a serve cycle, shared by the serial and batched loops so
    they can't drift apart. The Order is always recorded in *orders* (so a FAILED one is
    still logged); returns it only on a parsed order (status ORDERED), else None with the
    Order left FAILED. Book-keeping (handled / give-up tracking) stays in the caller —
    only the serial loop re-scans, so only it needs give-up state.
    """
    order = Order(id=len(orders) + 1, world_xy=caller.world_xy, bearing=caller.bearing)
    orders[order.id] = order
    ctx.score("detect_customer")  # detected + selected a waving customer (claimed)
    if not approach_customer(ctx, caller.world_xy):
        order.status = OrderStatus.FAILED
        return None
    order.status = OrderStatus.APPROACHED
    ctx.score("reach_table")  # reached the customer's table
    order.appearance = capture_appearance(ctx, caller.world_xy)  # for re-ID/logging
    items = take_order(ctx, world_xy=order.world_xy)
    if not items:
        order.status = OrderStatus.FAILED
        return None
    order.items = items
    order.status = OrderStatus.ORDERED
    ctx.score("understand_order")  # captured + confirmed the order
    return order


def _deliver_order(ctx: TaskContext, order: Order) -> None:
    """The per-order delivery half of a serve cycle: relay at the bar, then pick + serve.

    Returns to the bar, relays the order to the barman, then runs the (Phase-2 gated)
    pick/serve. Shared by the serial and batched loops (see :func:`_take_one_order`).
    """
    return_to_bar(ctx)
    if relay_to_barman(ctx, order.items):
        order.status = OrderStatus.RELAYED
        ctx.score("communicate_barman")  # relayed the order to the barman
    if _tray_mode():
        _serve_with_tray(ctx, order)
    else:
        _pick_and_serve(ctx, order)


# ---------------------------------------------------------------------------
# Phase 0 states
# ---------------------------------------------------------------------------
class GoToStart(SubTask):
    """Anchor the Kitchen-bar start pose (where relayed orders are picked).

    Default (``RESTAURANT_KITCHEN_BAR_POSE`` unset or ``"current"``): treat
    wherever the robot is standing now as the bar anchor and DON'T drive. The
    arena isn't pre-mapped (rulebook 5.5), so the start pose is only a relative
    reference — anchoring on the current pose means the operator just places the
    robot at the bar and runs, with no map pose to type each time. Set an explicit
    ``"x,y,heading_rad"`` to drive to a fixed pose instead (manual override).

    This step never reads the shared LocationBook: a ``kitchen_bar`` waypoint in
    ``world.toml`` (e.g. defined for GPSR) is deliberately ignored here, so the
    pre-map can never make Restaurant drive somewhere the rulebook forbids.

    Later phases re-acquire the bar/barman visually on return rather than trusting
    this point blindly (design doc §5.1).
    """

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        raw = os.getenv("RESTAURANT_KITCHEN_BAR_POSE", "current").strip().lower()
        explicit = raw not in ("", "current", "here", "now")

        # Drive only when an explicit env pose is set. The shared pre-map is NEVER
        # consulted (rulebook 5.5 — the arena isn't surveyed in advance), so the
        # default is to anchor on wherever the robot stands now and stay put.
        if not explicit:
            # Start = current pose; stay put. Needs a genuine odometry fix to anchor.
            fix = _odom_fix(ctx)
            if not fix:
                print("[restaurant] GoToStart: no odometry fix; cannot anchor the bar here")
                return StepResult.RETRY
            ctx.data["bar_anchor"] = {"x": fix["x"], "y": fix["y"], "heading": fix["heading"]}
            print(f"[restaurant] bar anchor = current pose "
                  f"({fix['x']:.2f}, {fix['y']:.2f}, {math.degrees(fix['heading']):.0f}deg); staying put")
            return StepResult.DONE

        # Explicit env bar pose: drive there, then anchor on the pose we actually reached.
        x, y, h = _pose("RESTAURANT_KITCHEN_BAR_POSE")
        ok = ctx.goto(x, y, h)
        # Key off a genuine odometry fix (None) — NOT coordinate truthiness: (0,0) is a
        # valid pose at the SLAM origin, so `if pose.get("x")` would discard a real fix.
        fix = _odom_fix(ctx)
        ctx.data["bar_anchor"] = (
            {"x": fix["x"], "y": fix["y"], "heading": fix["heading"]}
            if fix else {"x": x, "y": y, "heading": h}
        )
        return StepResult.DONE if ok else StepResult.RETRY


class SignalReady(SubTask):
    """Announce that the robot is in position and ready to begin serving.

    A spoken go-signal emitted after GoToStart, so the operator/referee knows the
    robot is set before it starts working. Gated by RESTAURANT_SIGNAL_READY (default
    on); never blocks the run (always DONE).
    """

    def run(self, ctx: TaskContext) -> StepResult:
        if os.getenv("RESTAURANT_SIGNAL_READY", "1").strip().lower() in ("1", "true", "yes", "on"):
            ctx.say(prompts.READY_TO_START)
        return StepResult.DONE


class ScanAndApproach(SubTask):
    """Phase 0 core: sweep for a waving customer, then approach to a stand-off.

    Re-sweeps on an empty scan (callers come and go); aborts to the next step
    only after exhausting retries. Stores the chosen caller on ctx.data["target"].
    """

    max_retries = 2

    def run(self, ctx: TaskContext) -> StepResult:
        target = _next_caller(ctx, [], _f("RESTAURANT_HANDLED_RADIUS_M", "0.6"))
        if target is None:
            ctx.say(prompts.NO_CUSTOMER)
            return StepResult.RETRY
        ctx.data["target"] = target
        if approach_customer(ctx, target.world_xy):
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
        max_fails = _int("RESTAURANT_MAX_FAILS_PER_SPOT", "2")
        radius = float(os.getenv("RESTAURANT_HANDLED_RADIUS_M", "0.6"))
        orders: dict[int, Order] = ctx.data.setdefault("orders", {})
        # Map points of customers we've already taken an order from. We loop until
        # this many DISTINCT customers are handled — NOT a raw counter — so a
        # still-waving customer re-seen on a later sweep can't be served twice and
        # falsely satisfy the rulebook's "at least two customers" (design review).
        handled: list[tuple[float, float]] = []
        # Spots we keep failing at (approach refused, or order never parsed): [x, y,
        # fails]. Once fails >= max_fails the spot is skipped too, so one uncooperative
        # caller can't burn every attempt while a second waving customer goes unreached.
        giveups: list[list[float]] = []
        attempts = 0

        def note_failure(xy: tuple[float, float]) -> None:
            for g in giveups:
                if math.hypot(g[0] - xy[0], g[1] - xy[1]) <= radius:
                    g[2] += 1
                    return
            giveups.append([xy[0], xy[1], 1.0])

        while len(handled) < target and attempts < max_attempts:
            attempts += 1

            # 1. Detect + approach + take the order (Phase 0), skipping anyone already
            # handled and any spot we've given up on (failed max_fails times). By default
            # this stops at the FIRST waver and drives straight over (RESTAURANT_APPROACH_FIRST).
            blocked = handled + [(g[0], g[1]) for g in giveups if g[2] >= max_fails]
            caller = _next_caller(ctx, blocked, radius)
            if caller is None:
                ctx.say(prompts.NO_CUSTOMER)
                continue
            order = _take_one_order(ctx, caller, orders)
            if order is None:
                note_failure(caller.world_xy)
                continue
            # Mark this customer handled NOW (order secured) so the next sweep
            # won't re-select them even if they keep waving.
            handled.append(caller.world_xy)

            # 2. Relay at the bar, then pick + serve (Phase 2 — gated until calibrated).
            _deliver_order(ctx, order)

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
        want = min(max(1, _int("RESTAURANT_BATCH_SIZE", "2")), target)
        radius = float(os.getenv("RESTAURANT_HANDLED_RADIUS_M", "0.6"))
        orders: dict[int, Order] = ctx.data.setdefault("orders", {})

        # Phase A — gather up to `want` orders from ONE sweep, nearest callers first.
        callers = scan_for_callers(ctx)
        if not callers:
            ctx.say(prompts.NO_CUSTOMER)
            return StepResult.DONE
        p = ctx.current_pose()
        callers.sort(key=lambda c: math.hypot(c.world_xy[0] - p["x"], c.world_xy[1] - p["y"]))
        taken: list[Order] = []
        handled: list[tuple[float, float]] = []
        # Walk the WHOLE sorted list, not just the first `want`: if a near caller fails to
        # approach / order, fall through to the next instead of forfeiting that slot (the old
        # `callers[:want]` slice gave up a slot whenever a leading caller failed). Skip anyone
        # within RESTAURANT_HANDLED_RADIUS_M of one already taken this sweep (a second
        # detection of the same person that survived the scan dedup).
        for caller in callers:
            if len(taken) >= want:
                break
            if not exclude_handled([caller], handled, radius):
                continue  # duplicate of a customer already taken this sweep
            order = _take_one_order(ctx, caller, orders)
            if order is None:
                continue
            handled.append(caller.world_xy)
            taken.append(order)

        if not taken:
            ctx.say(prompts.ALL_DONE)
            return StepResult.DONE

        # Phase B — deliver each (per-order bar trip; tray would allow one trip).
        for order in taken:
            _deliver_order(ctx, order)

        ctx.say(prompts.ALL_DONE)
        print("[restaurant] batched orders: " + ", ".join(
            f"#{o.id}={o.status.name}({o.items})" for o in orders.values()))
        return StepResult.DONE


class TestTask(SubTask):
    """A simple test subtask, for quick manual testing of the infrastructure."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        # Full pick via the grasp skill: detect -> approach+aim -> de-deadzone ->
        # grasp on the auto-selected arm (per-move result checks inside).
        ctx.walkie.arm.go_to_home(group_name="right_arm", pose_name="standby", blocking=False)
        ok = pick_object(
            ctx, prompts=["red can"], arm="auto",
            pregrasp_standoff_m=0.2, approach_preference="side", approach_weight=2.0,
        )
        print(f"[test] pick_object -> {ok}")
        return StepResult.DONE if ok else StepResult.RETRY


class GraspPlanTestTask(SubTask):
    """Plan-only grasp test: call get_object_grasp_pos and print the candidate.

    Runs the grasp PLANNER directly (tasks.skills.get_object_grasp_pos) on the
    object(s) matching ``RESTAURANT_GRASP_PROMPTS`` (comma-separated; default
    ``"red can"``) and prints the winning GraspCandidate — map-frame grasp /
    pre-grasp points, gripper width, GraspNet score, approach axis, footprint, and
    support surface. No base, arm, or head motion beyond the planner's own snapshots,
    so it's the cheapest way to eyeball whether detection + GraspNet produce a sane
    grasp from where the robot stands, before trusting the full pick motion (TestTask).
    The approach bias mirrors TestTask (side / weight 2.0) so the printed candidate
    matches what ``pick_object`` would actually plan.
    """

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        raw = os.getenv("RESTAURANT_GRASP_PROMPTS", "red can")
        prompts = [p.strip() for p in raw.split(",") if p.strip()]
        if not prompts:
            print("[test] grasp-plan: RESTAURANT_GRASP_PROMPTS is empty")
            return StepResult.RETRY
        print(f"[test] grasp-plan: planning a grasp for {prompts}")

        cand = get_object_grasp_pos(
            ctx, prompts,
            approach_preference="side", approach_weight=2.0,
        )
        if cand is None:
            print("[test] get_object_grasp_pos -> None (no graspable detection)")
            return StepResult.RETRY

        gx, gy, gz = cand.grasp_xyz
        px, py, pz = cand.pregrasp_xyz
        print(f"[test] get_object_grasp_pos -> grasp=({gx:+.3f},{gy:+.3f},{gz:+.3f})m "
              f"pregrasp=({px:+.3f},{py:+.3f},{pz:+.3f})m "
              f"width={cand.width:.3f}m score={cand.score:.3f}")
        print(f"[test]   approach={cand.approach.round(3).tolist()} "
              f"footprint={cand.object_footprint_m} support_z={cand.support_surface_z} "
              f"grasp_to_surface_offset={cand.grasp_to_surface_offset}")
        return StepResult.DONE


class SurfaceScanTestTask(SubTask):
    """Read-only demo of surface perception (tasks.skills.detect_surfaces).

    Scans for horizontal surfaces (tables/shelves/floor) from one depth snapshot and
    prints each surface's height, footprint, and the objects sitting on it — exactly
    the structure a placement agent reads to decide *where* to put something. No arm
    or base motion, so it's safe to run during bring-up to eyeball detection before
    trusting the full place motion.
    """

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        # detect_objects=True also runs open-vocab detection and prints, per surface,
        # which of these prompts sit on it (with height + XY distance). Drop it for a
        # faster geometry-only scan.
        surfaces = detect_surfaces(
            ctx,
            detect_objects=True,
            object_prompts=["bottle", "cup", "can", "bowl", "plate"],
        )
        if not surfaces:
            print("[test] detect_surfaces -> no horizontal surfaces found")
            return StepResult.RETRY
        for s in surfaces:
            print(f"[test] surface {s.id}: z={s.z:.2f}m area={s.area:.2f}m^2 "
                  f"centroid=({s.centroid[0]:+.2f},{s.centroid[1]:+.2f}) n_points={s.n_points}")
        return StepResult.DONE


class PickAndPlaceTestTask(SubTask):
    """Full pick -> place demo (grasp memory + surface placement).

    Picks the target with the grasp skill — which records what it grabbed and how
    high above its support surface it sat (``ctx.data["held_objects"]``) — then
    places it on a clear spot of a detected surface, reconstructing that same height
    on the new surface so it lands upright. Placement is fully automatic here (nearest
    reachable surface + a free spot); see the commented overrides for agent-driven
    placement once you've inspected SurfaceScanTestTask's output.
    """

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        # ctx.walkie.arm.go_to_home(group_name="right_arm", pose_name="standby", blocking=False)

        # 1. Pick — on success this records the held object (per arm) for the placer.
        if not pick_object(
            ctx, prompts=["cereal"], arm="left",
            pregrasp_standoff_m=0.2, approach_preference="side", approach_weight=2.0,
        ):
            print("[test] pick_object -> False; nothing to place")
            return StepResult.RETRY

        arms = held_arms(ctx)
        held = recall_held_object(ctx, arms[0]) if arms else None
        if held is not None:
            print(f"[test] holding {held.label!r} in {held.arm} arm "
                  f"(grasp_to_surface_offset={held.grasp_to_surface_offset}, "
                  f"footprint={held.footprint_m})")
        
        input("Press Enter to continue to placement...")

        # 2. Place — auto-pick the nearest reachable surface and a free spot. To let an
        # agent steer it instead, pass an explicit surface or spot, e.g.:
        #   surfaces = detect_surfaces(ctx)
        #   ok = place_object(ctx, surface=surfaces[0])
        #   ok = place_object(ctx, target_xy=(1.2, 0.4))
        ok = place_object(ctx)
        print(f"[test] place_object -> {ok}")
        return StepResult.DONE if ok else StepResult.RETRY


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_phase0_slice(ctx: TaskContext) -> Task:
    """Phase 0 only: GoToStart -> ScanAndApproach. For on-robot bring-up."""
    return Task("Restaurant-Phase0", [GoToStart(), ScanAndApproach()], ctx)


def build_surface_demo(ctx: TaskContext) -> Task:
    """Read-only surface scan — list detected surfaces + what's on them. No motion."""
    return Task("Restaurant-SurfaceScan", [SurfaceScanTestTask()], ctx)


def build_pick_demo(ctx: TaskContext) -> Task:
    """Pick-only demo (grasp skill) — detect -> approach -> grasp. No placement."""
    return Task("Restaurant-PickDemo", [TestTask()], ctx)


def build_grasp_plan_demo(ctx: TaskContext) -> Task:
    """Plan-only grasp test (get_object_grasp_pos) — print the candidate, no motion."""
    return Task("Restaurant-GraspPlan", [GraspPlanTestTask()], ctx)


def build_place_demo(ctx: TaskContext) -> Task:
    """Demo of the grasp-memory + surface-placement pipeline (tasks.skills.place).

    Read-only scan first so you can eyeball detected surfaces, then the full
    pick -> place motion. For on-robot bring-up of the place skill.
    """
    return Task("Restaurant-PlaceDemo", [SurfaceScanTestTask(), PickAndPlaceTestTask()], ctx)


def build_restaurant_task(ctx: TaskContext) -> Task:
    """Full task. Serial loop by default; batched order-taking when RESTAURANT_BATCH=1.

    Pure: touches no hardware at build time. Run isolated slices for step-by-step
    on-robot bring-up via RESTAURANT_SLICE (see tasks/Restaurant/run.py).
    """
    batched = os.getenv("RESTAURANT_BATCH", "0").lower() in ("1", "true", "yes")
    serve = ServeCustomersBatched() if batched else ServeCustomers()
    return Task("Restaurant", [GoToStart(), SignalReady(), serve], ctx)
