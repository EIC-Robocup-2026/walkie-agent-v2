"""Offline tests for the Restaurant serve loops' control flow (no robot/LLM/network).

The serial (``ServeCustomers``) and batched (``ServeCustomersBatched``) loops carry the
no-arm scoring tier: distinct-customer accounting, giving up on a spot that keeps failing,
and (batched) walking past a failed caller instead of forfeiting that slot. That logic
only runs on the robot, so here we stub the perception / HRI / nav skills the loops call
and assert the pure control flow — the bookkeeping that decides how many DISTINCT
customers actually get served. Locks the A1 refactor (shared ``_take_one_order`` /
``_deliver_order``) against regression.
"""

from __future__ import annotations

import pytest

from tasks.base import StepResult
from tasks.Restaurant import prompts, subtasks
from tasks.Restaurant.skills import Caller
from tasks.Restaurant.subtasks import OrderStatus, ServeCustomers, ServeCustomersBatched, SignalReady

# Statuses that mean "we secured an order from this customer".
_SERVED_OR_BETTER = {
    OrderStatus.ORDERED, OrderStatus.RELAYED, OrderStatus.PICKED, OrderStatus.SERVED,
}


class FakeCtx:
    """Minimal TaskContext stand-in: just the surface the serve loops touch."""

    def __init__(self):
        self.data = {}
        self.said: list[str] = []

    def say(self, text):
        self.said.append(text)

    def current_pose(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    def score(self, key, n=1):  # serve loop awards against the live tally; no-op here
        pass


def _caller(x, y, conf=0.9):
    return Caller(world_xy=(x, y), bearing=0.0, bbox_xyxy=(0.0, 0.0, 1.0, 1.0), confidence=conf)


def _orders_taken(ctx):
    return [o for o in ctx.data["orders"].values() if o.status in _SERVED_OR_BETTER]


@pytest.fixture
def patched(monkeypatch):
    """Stub the nav/HRI/perception skills the loops call; return a call-recorder.

    The serve loops reference these as module globals in ``subtasks``, so patching the
    attribute there redirects the call at run time. ``exclude_handled`` / ``_take_one_order``
    / ``_deliver_order`` are left REAL — that's the shared bookkeeping under test.
    """
    calls = {"scan": 0, "approach": [], "take_order": []}

    def install(*, scan, approach, take_order, nearest=None, find_near=None):
        def _scan(ctx):
            calls["scan"] += 1
            return list(scan())

        def _approach(ctx, world_xy, **kw):
            calls["approach"].append(world_xy)
            return approach(world_xy)

        def _take_order(ctx, world_xy=None):
            calls["take_order"].append(world_xy)
            return list(take_order(world_xy))

        monkeypatch.setattr(subtasks, "scan_for_callers", _scan)
        monkeypatch.setattr(subtasks, "approach_customer", _approach)
        monkeypatch.setattr(subtasks, "take_order", _take_order)
        monkeypatch.setattr(subtasks, "capture_appearance", lambda ctx, xy: None)
        # Seat-refine after approach (issue 2): default = no refinement (keep the coarse
        # caller point, preserving the legacy assertions); a test can inject a refined point.
        monkeypatch.setattr(subtasks, "find_person_near",
                            find_near or (lambda ctx, xy, **kw: None))
        monkeypatch.setattr(subtasks, "return_to_bar", lambda ctx, **kw: True)
        monkeypatch.setattr(subtasks, "relay_to_barman", lambda ctx, items: True)
        # Stub BOTH delivery paths so the loop test is serve-mode agnostic
        # (_deliver_order picks tray vs gripper from RESTAURANT_TRAY_MODE).
        monkeypatch.setattr(subtasks, "_pick_and_serve", lambda ctx, order: None)
        monkeypatch.setattr(subtasks, "_serve_with_tray", lambda ctx, order: None)
        if nearest is not None:
            monkeypatch.setattr(subtasks, "nearest_caller", nearest)
        return calls

    return install


_NEAREST_FIRST = lambda ctx, callers: callers[0] if callers else None  # noqa: E731


# --- serial loop: distinct-customer accounting -----------------------------

def test_serial_does_not_double_serve_a_persistent_waver(monkeypatch, patched):
    """A customer who keeps waving after being served is taken exactly once.

    The rulebook needs >= 2 DISTINCT customers; counting one impatient waver twice would
    falsely satisfy it. ``exclude_handled`` must drop them on every later sweep.
    """
    monkeypatch.setenv("RESTAURANT_TARGET_CUSTOMERS", "2")
    monkeypatch.setenv("RESTAURANT_EXTRA_ATTEMPTS", "3")  # max_attempts = 5
    monkeypatch.setenv("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "0")  # exercise the full-sweep path this stubs
    calls = patched(
        scan=lambda: [_caller(2.0, 0.0)],   # the SAME customer, waving every sweep
        approach=lambda xy: True,
        take_order=lambda xy: ["coke"],
        nearest=_NEAREST_FIRST,
    )
    ctx = FakeCtx()
    ServeCustomers().run(ctx)

    assert len(calls["take_order"]) == 1            # took the order only once
    assert len(_orders_taken(ctx)) == 1
    assert calls["scan"] == 5                        # kept re-scanning to the attempt budget
    assert ctx.said.count(prompts.NO_CUSTOMER) >= 1  # later sweeps correctly found nobody new


def test_serial_gives_up_on_a_bad_spot_after_max_fails(monkeypatch, patched):
    """An un-approachable caller is retried max_fails times, then blocked.

    Without the give-up cap one uncooperative caller burns every attempt while a second
    waving customer is never reached.
    """
    monkeypatch.setenv("RESTAURANT_TARGET_CUSTOMERS", "2")
    monkeypatch.setenv("RESTAURANT_EXTRA_ATTEMPTS", "3")  # max_attempts = 5
    monkeypatch.setenv("RESTAURANT_MAX_FAILS_PER_SPOT", "2")
    monkeypatch.setenv("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "0")  # exercise the full-sweep path this stubs
    calls = patched(
        scan=lambda: [_caller(3.0, 0.0)],
        approach=lambda xy: False,          # this spot never lets us approach
        take_order=lambda xy: ["x"],
        nearest=_NEAREST_FIRST,
    )
    ctx = FakeCtx()
    ServeCustomers().run(ctx)

    assert len(calls["approach"]) == 2      # tried exactly max_fails times...
    assert calls["scan"] == 5               # ...even though it kept scanning the budget
    assert len(calls["take_order"]) == 0    # never got far enough to ask
    assert len(_orders_taken(ctx)) == 0


# --- batched loop: failure tolerance ---------------------------------------

def test_batched_walks_past_a_failed_leading_caller(monkeypatch, patched):
    """A failed nearest caller doesn't forfeit a batch slot — the next caller fills it.

    The old ``callers[:want]`` slice yielded only 1 order here (the leading caller failed
    and its slot was lost); the hardened loop walks the whole sorted list and fills both.
    """
    monkeypatch.setenv("RESTAURANT_TARGET_CUSTOMERS", "2")
    monkeypatch.setenv("RESTAURANT_BATCH_SIZE", "2")        # want = 2
    monkeypatch.setenv("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    callers = [_caller(1.0, 0.0), _caller(2.0, 0.0), _caller(3.0, 0.0)]  # nearest first

    def approach(xy):  # the nearest caller (1, 0) can't be approached
        return not (abs(xy[0] - 1.0) < 1e-6 and abs(xy[1]) < 1e-6)

    calls = patched(scan=lambda: callers, approach=approach, take_order=lambda xy: ["x"])
    ctx = FakeCtx()
    ServeCustomersBatched().run(ctx)

    assert len(_orders_taken(ctx)) == 2     # both slots filled (B + C), not just 1
    assert len(calls["approach"]) == 3      # A (fail), B, C
    assert len(calls["take_order"]) == 2    # only the two we reached
    assert calls["scan"] == 1               # single sweep, by design


# --- tray-mode delivery (_serve_with_tray) ---------------------------------

class _TrayCtx:
    """Records say/ask/score for the tray serve; ask returns a go-ahead (no dwell)."""

    def __init__(self, reply="ready"):
        self._reply = reply
        self.said: list[str] = []
        self.asked: list[str] = []
        self.scored: list[tuple[str, int]] = []

    def say(self, text):
        self.said.append(text)

    def ask(self, q, retries=1):
        self.asked.append(q)
        return self._reply

    def score(self, key, n=1):
        self.scored.append((key, n))


def test_serve_with_tray_one_trip_serves_whole_order(monkeypatch):
    """Tray mode: the barman loads, the robot carries the WHOLE order in one trip, the
    customer unloads — one return_to_customer, one return_to_bar, per-item scoring."""
    trips = {"to_customer": 0, "to_bar": 0}
    monkeypatch.setattr(subtasks, "return_to_customer",
                        lambda ctx, xy, **kw: trips.__setitem__("to_customer", trips["to_customer"] + 1) or xy)
    monkeypatch.setattr(subtasks, "return_to_bar",
                        lambda ctx, **kw: trips.__setitem__("to_bar", trips["to_bar"] + 1) or True)
    ctx = _TrayCtx()
    order = subtasks.Order(id=1, world_xy=(2.0, 3.0), bearing=0.0, items=["coke", "chips"])

    subtasks._serve_with_tray(ctx, order)

    assert order.status is OrderStatus.SERVED
    assert trips == {"to_customer": 1, "to_bar": 1}    # ONE round trip for both items
    assert ("pickup_items", 2) in ctx.scored          # scored per item (2)
    assert ("serve_order", 2) in ctx.scored
    assert any("tray" in s.lower() for s in ctx.said)  # asked the barman about the tray


def test_serve_with_tray_bails_when_customer_lost(monkeypatch):
    """Loaded the tray but can't re-find the customer: return to the bar, not SERVED."""
    monkeypatch.setattr(subtasks, "return_to_customer", lambda ctx, xy, **kw: None)
    bar = {"n": 0}
    monkeypatch.setattr(subtasks, "return_to_bar", lambda ctx, **kw: bar.__setitem__("n", bar["n"] + 1) or True)
    ctx = _TrayCtx()
    order = subtasks.Order(id=1, world_xy=(2.0, 3.0), bearing=0.0, items=["coke"])

    subtasks._serve_with_tray(ctx, order)

    assert order.status is OrderStatus.PICKED   # tray loaded, but the serve never happened
    assert bar["n"] == 1                          # bailed back to the bar
    assert ("serve_order", 1) not in ctx.scored


# --- issue 2: refined seat anchors handled-dedup / serve-return -------------

def test_take_one_order_anchors_on_refined_seat_not_coarse_caller(monkeypatch, patched):
    """The coarse pre-approach caller point is replaced by the up-close re-detection, so
    handled-dedup and the serve-time return use where the customer ACTUALLY is."""
    monkeypatch.setenv("RESTAURANT_TARGET_CUSTOMERS", "1")
    monkeypatch.setenv("RESTAURANT_EXTRA_ATTEMPTS", "0")   # max_attempts = 1
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "0")
    refined = _caller(5.0, 5.0)                            # far from the coarse (2, 0) lift
    patched(
        scan=lambda: [_caller(2.0, 0.0)],
        approach=lambda xy: True,
        take_order=lambda xy: ["coke"],
        nearest=_NEAREST_FIRST,
        find_near=lambda ctx, xy, **kw: refined,
    )
    ctx = FakeCtx()
    ServeCustomers().run(ctx)

    order = list(ctx.data["orders"].values())[0]
    assert order.world_xy == (5.0, 5.0)   # refined seat, NOT the coarse (2, 0) caller point


# --- issue 5: no redundant return_to_bar after delivery --------------------

def test_serial_returns_to_bar_once_per_delivery(monkeypatch, patched):
    """_deliver_order already ends at the bar; the loop must not drive there a second
    time (the old trailing return_to_bar double-drove the aisle every customer)."""
    monkeypatch.setenv("RESTAURANT_TARGET_CUSTOMERS", "1")
    monkeypatch.setenv("RESTAURANT_EXTRA_ATTEMPTS", "0")   # max_attempts = 1, one delivery
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "0")
    patched(
        scan=lambda: [_caller(2.0, 0.0)],
        approach=lambda xy: True,
        take_order=lambda xy: ["coke"],
        nearest=_NEAREST_FIRST,
    )
    bar = {"n": 0}
    # Override the fixture's return_to_bar with a counter (last setattr wins).
    monkeypatch.setattr(subtasks, "return_to_bar", lambda ctx, **kw: bar.__setitem__("n", bar["n"] + 1) or True)
    ctx = FakeCtx()
    ServeCustomers().run(ctx)

    assert bar["n"] == 1   # exactly one (the _deliver_order leading drive), not two


# --- issue 3: tray handoff waits for a positive go-ahead -------------------

class _ScriptedCtx:
    """ctx stub whose ask() pops scripted replies (then ''), recording say/ask/score."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.said: list[str] = []
        self.asked: list[str] = []
        self.scored: list[tuple[str, int]] = []

    def say(self, text):
        self.said.append(text)

    def ask(self, q, retries=1):
        self.asked.append(q)
        return self._replies.pop(0) if self._replies else ""

    def score(self, key, n=1):
        self.scored.append((key, n))


def test_await_handoff_keeps_waiting_through_not_yet(monkeypatch):
    monkeypatch.setenv("RESTAURANT_TRAY_HANDOFF_TRIES", "3")
    monkeypatch.setenv("RESTAURANT_TRAY_LOAD_WAIT_SEC", "0")  # no real sleep in tests
    ctx = _ScriptedCtx(["not yet, hold on", "ready"])
    assert subtasks._await_handoff(ctx, "loaded?", "RESTAURANT_TRAY_LOAD_WAIT_SEC") is True
    assert len(ctx.asked) == 2                       # didn't bail on "not yet"
    assert ctx.asked[1] == prompts.TRAY_STILL_WAITING


def test_await_handoff_times_out_on_silence(monkeypatch):
    monkeypatch.setenv("RESTAURANT_TRAY_HANDOFF_TRIES", "2")
    monkeypatch.setenv("RESTAURANT_TRAY_LOAD_WAIT_SEC", "0")
    ctx = _ScriptedCtx([])                            # pure silence every ask
    assert subtasks._await_handoff(ctx, "loaded?", "RESTAURANT_TRAY_LOAD_WAIT_SEC") is False
    assert len(ctx.asked) == 2                        # tried the full budget, then gave up


def test_await_handoff_accepts_any_non_wait_reply(monkeypatch):
    monkeypatch.setenv("RESTAURANT_TRAY_HANDOFF_TRIES", "3")
    monkeypatch.setenv("RESTAURANT_TRAY_LOAD_WAIT_SEC", "0")
    ctx = _ScriptedCtx(["the items are on the tray"])  # lenient: non-wait reply = done
    assert subtasks._await_handoff(ctx, "loaded?", "RESTAURANT_TRAY_LOAD_WAIT_SEC") is True
    assert len(ctx.asked) == 1


def test_serve_with_tray_unconfirmed_unload_is_not_scored(monkeypatch):
    """Load confirmed, but the customer never confirms taking the items: carry on (back
    to the bar) but do NOT claim the serve or announce it."""
    monkeypatch.setenv("RESTAURANT_TRAY_HANDOFF_TRIES", "2")
    monkeypatch.setenv("RESTAURANT_TRAY_LOAD_WAIT_SEC", "0")
    monkeypatch.setenv("RESTAURANT_TRAY_UNLOAD_WAIT_SEC", "0")
    monkeypatch.setattr(subtasks, "return_to_customer", lambda ctx, xy, **kw: xy)
    bar = {"n": 0}
    monkeypatch.setattr(subtasks, "return_to_bar", lambda ctx, **kw: bar.__setitem__("n", bar["n"] + 1) or True)
    ctx = _ScriptedCtx(["ready"])    # load confirmed; unload asks then hears silence
    order = subtasks.Order(id=1, world_xy=(2.0, 3.0), bearing=0.0, items=["coke", "chips"])

    subtasks._serve_with_tray(ctx, order)

    assert ("pickup_items", 2) in ctx.scored          # load WAS confirmed -> pickup scored
    assert order.status is OrderStatus.PICKED          # but the serve was never confirmed
    assert ("serve_order", 2) not in ctx.scored
    assert not any(prompts.SERVE_ANNOUNCE.format(items="coke, chips") == s for s in ctx.said)
    assert bar["n"] == 1                               # still returned to the bar (no stall)


# --- readiness go-signal (SignalReady) -------------------------------------

def test_signal_ready_announces_when_enabled(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SIGNAL_READY", "1")
    ctx = FakeCtx()
    assert SignalReady().run(ctx) is StepResult.DONE
    assert prompts.READY_TO_START in ctx.said


def test_signal_ready_is_silent_when_disabled(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SIGNAL_READY", "0")
    ctx = FakeCtx()
    assert SignalReady().run(ctx) is StepResult.DONE
    assert ctx.said == []
