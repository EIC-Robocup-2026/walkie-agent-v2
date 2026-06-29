"""Pure offline tests for the interleave scheduler (schedule.interleave).

No robot, no LLM: plans are built with the real (pure) ground_step, scheduled,
and the resulting order is checked for the two properties that make the interleave
both correct and *meaningful* (the bonus condition): each command's internal step
order is preserved, and each room is visited as ONE contiguous block (no revisit).
"""

from __future__ import annotations

import textwrap

import pytest

from tasks.GPSR.parse import ground_step
from tasks.GPSR.plan import Plan, step_location
from tasks.GPSR.prompts import RawStep
from tasks.GPSR.schedule import _region, interleave
from walkie_world.map.vocab import load_world

_WORLD_TOML = textwrap.dedent(
    """
    names = ["Charlie"]

    [rooms]
    kitchen = { pose = [1.0, 2.0, 0.0] }
    bedroom = { pose = [3.0, 4.0, 0.0] }
    office  = { pose = [5.0, 6.0, 0.0] }

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


def _assert_intra_order(order):
    """Each command's step indices must appear in ascending order (0,1,2,...)."""
    nxt: dict[int, int] = {}
    for cid, idx in order:
        assert idx == nxt.get(cid, 0), f"command {cid} step {idx} is out of order in {order}"
        nxt[cid] = idx + 1


def _region_blocks(order, plans, world):
    """The room sequence the robot visits (drop None / consecutive duplicates)."""
    seq = []
    for cid, idx in order:
        r = _region(world, step_location(plans[cid].steps[idx]))
        if r is not None and (not seq or seq[-1] != r):
            seq.append(r)
    return seq


def test_step_location_semantics(world):
    assert step_location(ground_step(_r(primitive="navigate", room="kitchen"), world)) == "kitchen"
    assert step_location(ground_step(_r(primitive="say", info="the time"), world)) is None


def test_region_maps_placement_to_its_room(world):
    assert _region(world, "kitchen_table") == "kitchen"  # a placement -> its room
    assert _region(world, "kitchen") == "kitchen"        # a room -> itself
    assert _region(world, None) is None


def test_interleave_preserves_order_and_includes_every_step(world):
    p1 = _plan(world, _r(primitive="navigate", room="kitchen"),
               _r(primitive="find_object", object="cola", room="kitchen"))
    p2 = _plan(world, _r(primitive="navigate", room="bedroom"),
               _r(primitive="find_person", person="Charlie", descriptor_kind="name", room="bedroom"))
    order = interleave([(1, p1), (2, p2)], world)
    assert sorted(order) == [(1, 0), (1, 1), (2, 0), (2, 1)]  # every step exactly once
    _assert_intra_order(order)


def test_interleave_batches_each_room_into_one_visit(world):
    # cmd1 kitchen, cmd2 bedroom, cmd3 kitchen: serial would visit kitchen TWICE.
    plans = {
        1: _plan(world, _r(primitive="navigate", room="kitchen"),
                 _r(primitive="count", object="cup", room="kitchen")),
        2: _plan(world, _r(primitive="navigate", room="bedroom"),
                 _r(primitive="find_person", person="Charlie", descriptor_kind="name", room="bedroom")),
        3: _plan(world, _r(primitive="navigate", room="kitchen"),
                 _r(primitive="count", object="cola", room="kitchen")),
    }
    order = interleave(list(plans.items()), world)
    _assert_intra_order(order)
    blocks = _region_blocks(order, plans, world)
    assert len(blocks) == len(set(blocks)), f"a room is revisited: {blocks}"
    assert set(blocks) == {"kitchen", "bedroom"}


def test_interleave_keeps_location_agnostic_steps_in_place(world):
    # A 'say' (region None) must not force a room change — it runs wherever we are.
    plans = {
        1: _plan(world, _r(primitive="navigate", room="kitchen"), _r(primitive="say", info="hello")),
        2: _plan(world, _r(primitive="navigate", room="kitchen"), _r(primitive="say", info="bye")),
    }
    order = interleave(list(plans.items()), world)
    _assert_intra_order(order)
    assert _region_blocks(order, plans, world) == ["kitchen"]  # one room, never leaves
