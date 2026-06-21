"""Interleave SAFETY net: lock the *aggregation* half of "the +200 path can't
score worse than serial", so the team can move toward flipping
``GPSR_INTERLEAVE=1`` with the scoring math nailed down.

Two pure (no robot/LLM) guarantees, on top of the order properties in
``test_gpsr_schedule.py``:

1. **Permutation** — ``interleave`` emits every command's every step exactly once.
   A dropped step would forfeit part of a command; a duplicated one would re-drive.
2. **Aggregation order-invariance** — *given the same per-step success outcomes*,
   the per-command ``CmdStatus`` (hence partial score) computed from the
   interleaved order is IDENTICAL to the serial order, for *any* pattern of step
   successes. Interleaving only reorders steps; it cannot change which command a
   step belongs to, and ``summarize_status`` depends only on each command's set of
   outcomes — so the score aggregation is invariant.

SCOPE — what this does NOT prove: that a step's *outcome* is the same in both
orders. Interleaving shares one nav cache (``state["at"]``) across commands, so in
principle a step could behave differently. The design defends this with
per-command scratch isolation (only the robot's location is global —
``dispatch.execute_interleaved``), but that, and the wiring-level serial fallback
(``subtasks._run_interleaved`` returns False with no side effects on a scheduling
error), are design/integration properties — the on-robot interleave validation in
the runbook is still required. This file locks only the aggregation math.
"""

from __future__ import annotations

import textwrap

import pytest

from tasks.GPSR.parse import ground_step
from tasks.GPSR.plan import CmdStatus, Plan, summarize_status
from tasks.GPSR.prompts import RawStep
from tasks.GPSR.schedule import interleave
from tasks.GPSR.world import load_world

_WORLD_TOML = textwrap.dedent(
    """
    names = ["Charlie", "Jane"]

    [rooms]
    kitchen     = { pose = [1.0, 2.0, 0.0] }
    bedroom     = { pose = [3.0, 4.0, 0.0] }
    office      = { pose = [5.0, 6.0, 0.0] }
    living_room = { pose = [7.0, 8.0, 0.0] }

    [locations]
    kitchen_table = { room = "kitchen", placement = true, pose = [1.0, 2.0, 0.0], aliases = ["kitchen table"] }
    desk          = { room = "office",  placement = true, pose = [5.0, 6.0, 0.0] }

    [object_categories]
    drinks = ["cola"]
    fruits = ["apple"]
    dishes = ["cup"]
    """
)


@pytest.fixture
def world(tmp_path):
    p = tmp_path / "world.toml"
    p.write_text(_WORLD_TOML)
    return load_world(p)


def _r(**kw):
    kw.setdefault("raw", "x")
    return RawStep(**kw)


def _plan(world, *raws):
    return Plan(steps=[ground_step(r, world) for r in raws], source="x")


def _plan_sets(world):
    """A spread of multi-command batches that force real reordering: shared rooms,
    differing step counts, location-agnostic steps, and a single-command degenerate
    case."""
    sets = []

    # Two commands touching the same room (kitchen) -> interleave merges the visit.
    sets.append([
        (1, _plan(world, _r(primitive="navigate", room="kitchen"),
                  _r(primitive="count", object="cola", location="kitchen_table"))),
        (2, _plan(world, _r(primitive="navigate", room="kitchen"),
                  _r(primitive="find_person", person="Charlie", descriptor_kind="name", room="kitchen"))),
    ])

    # Three commands, uneven lengths, a location-agnostic say, rooms revisited serially.
    sets.append([
        (1, _plan(world, _r(primitive="navigate", room="kitchen"),
                  _r(primitive="say", info="the time"),
                  _r(primitive="count", object="cup", location="kitchen_table"))),
        (2, _plan(world, _r(primitive="navigate", room="bedroom"),
                  _r(primitive="find_person", person="Jane", descriptor_kind="name", room="bedroom"))),
        (3, _plan(world, _r(primitive="navigate", room="kitchen"),
                  _r(primitive="find_object", object="apple", room="kitchen"))),
    ])

    # Single command (degenerate — interleave must equal serial trivially).
    sets.append([
        (1, _plan(world, _r(primitive="navigate", room="office"),
                  _r(primitive="get_object_property", object="apple", which="color"))),
    ])

    # Four commands across four rooms, all single-step navigates + a follow (agnostic).
    sets.append([
        (1, _plan(world, _r(primitive="navigate", room="office"))),
        (2, _plan(world, _r(primitive="navigate", room="living_room"),
                  _r(primitive="follow", person="Charlie", descriptor_kind="name"))),
        (3, _plan(world, _r(primitive="navigate", room="bedroom"))),
        (4, _plan(world, _r(primitive="navigate", room="kitchen"))),
    ])
    return sets


# Per-step success patterns to test status-invariance against. Each maps
# (command_id, step_index) -> bool. They span all/none/mixed so DONE, PARTIAL and
# FAILED are all exercised on both sides.
_OK_PATTERNS = {
    "all_succeed": lambda cid, idx: True,
    "all_fail": lambda cid, idx: False,
    "alternate_by_step": lambda cid, idx: idx % 2 == 0,
    "first_step_only": lambda cid, idx: idx == 0,
    "by_command_parity": lambda cid, idx: cid % 2 == 0,
    "one_command_all_fail": lambda cid, idx: cid != 2,
}


def _all_steps(indexed):
    return sorted((cid, i) for cid, plan in indexed for i in range(len(plan.steps)))


def test_interleave_is_an_exact_permutation_of_every_step(world):
    for indexed in _plan_sets(world):
        order = interleave(indexed, world)
        assert sorted(order) == _all_steps(indexed), (
            f"order is not a permutation of all steps: {order}"
        )
        assert len(order) == len(set(order)), f"a step appears twice: {order}"


def _serial_status(indexed, ok):
    return {
        cid: summarize_status([ok(cid, i) for i in range(len(plan.steps))])
        for cid, plan in indexed
    }


def _interleaved_status(indexed, order, ok):
    oks: dict[int, list[bool]] = {cid: [] for cid, _ in indexed}
    for cid, idx in order:
        oks[cid].append(ok(cid, idx))
    return {cid: summarize_status(v) for cid, v in oks.items()}


def test_interleaved_status_equals_serial_for_every_outcome_pattern(world):
    """Aggregation order-invariance: identical per-command status (hence identical
    partial score) under any pattern of step successes — *given* the same outcomes.
    (Outcome identity itself is a design property; see the module docstring SCOPE.)"""
    for indexed in _plan_sets(world):
        order = interleave(indexed, world)
        for name, ok in _OK_PATTERNS.items():
            serial = _serial_status(indexed, ok)
            inter = _interleaved_status(indexed, order, ok)
            assert inter == serial, (
                f"pattern {name!r}: interleaved {inter} != serial {serial}"
            )


def test_status_patterns_actually_exercise_all_three_outcomes(world):
    """Guard the guard: make sure the patterns above really produce DONE, PARTIAL
    and FAILED somewhere (otherwise the equality test could pass vacuously)."""
    seen = set()
    for indexed in _plan_sets(world):
        for ok in _OK_PATTERNS.values():
            seen.update(_serial_status(indexed, ok).values())
    assert {CmdStatus.DONE, CmdStatus.PARTIAL, CmdStatus.FAILED} <= seen
