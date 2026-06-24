"""Offline tests for the Pick and Place non-arm pipeline (no robot/LLM/network).

The arm is a separate skill under development, so PickAndPlace gates every
grasp/place behind PNP_ARM_CALIBRATED and the flow still has to earn its non-arm
budget: navigate, *recognize* each object, and *indicate the correct placement*
(rulebook 5.2 scoresheet + remark 16). These tests stub perception/sort/nav and
assert that pure control flow — that with the arm gated OFF the loops touch no
arm yet still communicate every object's placement, and that with the gate ON the
real pick/place fires. Locks the Phase A/B refactor against regression.
"""

from __future__ import annotations

import pytest

from tasks.base import StepResult
from tasks.PickAndPlace import prompts, subtasks
from tasks.PickAndPlace.skills import DetectedObject
from tasks.PickAndPlace.subtasks import (
    PerceiveDiningTable,
    ServeBreakfast,
    TidyDiningTable,
    TidyExtraSurface,
)


class FakeCtx:
    """Minimal TaskContext stand-in: just the surface the PnP subtasks touch."""

    def __init__(self):
        self.data = {}
        self.said: list[str] = []
        self.gotos: list[tuple[float, float, float]] = []

    def say(self, text):
        self.said.append(text)

    def goto(self, x, y, h):
        self.gotos.append((x, y, h))
        return True

    def current_pose(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}

    def rotate_to(self, heading):
        return True


def _obj(name, conf=0.9, xy=(1.0, 0.0)):
    return DetectedObject(
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0), class_name=name, confidence=conf,
        world_xy=xy, world_xyz=(xy[0], xy[1], 0.5),
    )


def _recognized(ctx):
    return [s for s in ctx.said if s.startswith("I can see a ")]


def _indicated(ctx):
    return [s for s in ctx.said if s.startswith("The ")]


@pytest.fixture
def stub(monkeypatch):
    """Stub perception/sort/nav/arm; leave announce_object + indicate_placement REAL.

    The real announce/indicate helpers (pure: say + optional base turn) are what
    communicate perception to the referee — the behaviour under test — so they
    stay real and we assert on ``ctx.said``. ``perceive_and_indicate_shelf`` is
    stubbed to a counter (the real one would call the camera).
    """
    calls = {"pick": [], "place": [], "place_at": [], "shelf": 0}

    def install(*, objects, sort, arm=False, breakfast_objects=None):
        def _perceive(ctx, classes=None):
            # breakfast slice passes explicit classes; the table path passes None.
            if classes is not None and breakfast_objects is not None:
                return list(breakfast_objects)
            return list(objects)

        monkeypatch.setattr(subtasks, "perceive_surface", _perceive)
        monkeypatch.setattr(subtasks, "sort_object", lambda ctx, o: sort(o))
        monkeypatch.setattr(subtasks, "arm_enabled", lambda: arm)
        monkeypatch.setattr(
            subtasks, "perceive_and_indicate_shelf",
            lambda ctx: calls.__setitem__("shelf", calls["shelf"] + 1) or [],
        )
        monkeypatch.setattr(subtasks, "pick_object",
                            lambda ctx, o: calls["pick"].append(o.class_name) or True)
        monkeypatch.setattr(subtasks, "place_object",
                            lambda ctx, dest, group=None: calls["place"].append((dest, group)) or True)
        monkeypatch.setattr(subtasks, "place_at",
                            lambda ctx, pose: calls["place_at"].append(pose) or True)
        return calls

    return install


def _sort(mapping, group="snacks"):
    """Build a sort() returning ObjectSort by class -> destination (default cabinet)."""
    def sort(o):
        dest = mapping.get(o.class_name, "cabinet")
        return prompts.ObjectSort(
            destination=dest, cabinet_group=(group if dest == "cabinet" else None)
        )
    return sort


# --- perception: recognize each object -------------------------------------

def test_perceive_announces_every_recognized_object(stub):
    """PerceiveDiningTable speaks each object — the 'correctly recognize' score."""
    objects = [_obj("cup"), _obj("plate"), _obj("apple")]
    stub(objects=objects, sort=_sort({}), arm=False)
    ctx = FakeCtx()
    PerceiveDiningTable().run(ctx)

    assert ctx.data["table_objects"] == objects
    assert len(_recognized(ctx)) == 3  # one "I can see a X." per object


# --- tidy: arm gated OFF (the non-arm scoring path) ------------------------

def test_tidy_arm_off_indicates_without_touching_arm(stub):
    """Gate off: indicate every placement, perceive the shelf once, NO arm calls."""
    objects = [_obj("cup"), _obj("plate"), _obj("apple")]
    calls = stub(objects=objects, sort=_sort({"cup": "dishwasher", "plate": "dishwasher"}), arm=False)
    ctx = FakeCtx()
    ctx.data["table_objects"] = objects
    res = TidyDiningTable().run(ctx)

    assert res is StepResult.DONE
    assert calls["pick"] == []                     # arm gated: never grasped
    assert calls["place"] == []                    # arm gated: never placed
    assert len(_indicated(ctx)) == 3               # every placement communicated
    assert calls["shelf"] == 1                     # one cabinet-bound item -> shelf perceived once
    assert ctx.data["sorted"]["dishwasher"]        # sort bookkeeping still recorded
    assert ctx.data.get("dishwasher_items", 0) == 0  # nothing actually loaded


def test_tidy_arm_off_no_shelf_when_no_cabinet_items(stub):
    """No cabinet-bound object -> no shelf perception (it would be a wasted trip)."""
    objects = [_obj("cup"), _obj("fork")]
    calls = stub(objects=objects, sort=_sort({"cup": "dishwasher", "fork": "dishwasher"}), arm=False)
    ctx = FakeCtx()
    ctx.data["table_objects"] = objects
    TidyDiningTable().run(ctx)

    assert calls["shelf"] == 0


# --- tidy: arm gated ON (the manipulation path is layered on cleanly) ------

def test_tidy_arm_on_picks_and_places_each(stub):
    """Gate on: the same flow now physically picks + places, loading the dishwasher."""
    objects = [_obj("cup"), _obj("apple")]
    calls = stub(objects=objects, sort=_sort({"cup": "dishwasher"}), arm=True)
    ctx = FakeCtx()
    ctx.data["table_objects"] = objects
    TidyDiningTable().run(ctx)

    assert calls["pick"] == ["cup", "apple"]
    assert ("dishwasher", None) in calls["place"]
    assert ctx.data["dishwasher_items"] == 1       # one item loaded -> CloseDishwasher will fire


# --- extra surface: same gating shape --------------------------------------

def test_extra_surface_arm_off_indicates_cabinet_only(stub):
    objects = [_obj("box"), _obj("can")]
    calls = stub(objects=objects, sort=_sort({}), arm=False)  # both -> cabinet
    ctx = FakeCtx()
    res = TidyExtraSurface().run(ctx)

    assert res is StepResult.DONE
    assert calls["pick"] == []
    assert len(_recognized(ctx)) == 2              # recognized each
    assert len(_indicated(ctx)) == 2              # indicated cabinet for each
    assert calls["shelf"] == 1


# --- breakfast: arm gated -> recognize + announce, never fetch -------------

def test_breakfast_arm_off_recognizes_without_fetching(stub):
    bowl_spoon = [_obj("bowl"), _obj("spoon")]
    calls = stub(objects=[], sort=_sort({}), arm=False, breakfast_objects=bowl_spoon)
    ctx = FakeCtx()
    ServeBreakfast().run(ctx)

    assert calls["pick"] == []                     # arm gated: no fetch
    assert calls["place_at"] == []
    assert len(_recognized(ctx)) == 4              # one recognized target per breakfast item


# --- shelf-indicate guard: at most once across tidy + extra ----------------

def test_shelf_indicated_once_across_steps(stub):
    table = [_obj("apple")]            # cabinet-bound
    calls = stub(objects=table, sort=_sort({}), arm=False)
    ctx = FakeCtx()
    ctx.data["table_objects"] = table
    TidyDiningTable().run(ctx)
    TidyExtraSurface().run(ctx)        # shares ctx.data -> guard must hold

    assert calls["shelf"] == 1


# --- slice builders construct (PNP_SLICE wiring) ---------------------------

def test_slice_builders_construct():
    ctx = FakeCtx()
    builders = (
        subtasks.build_nav_slice,
        subtasks.build_perceive_slice,
        subtasks.build_sort_slice,
        subtasks.build_breakfast_slice,
        subtasks.build_pick_and_place_task,
    )
    for build in builders:
        task = build(ctx)
        assert task.subtasks  # non-empty step list


# --- the gate at the seam: skills.pick_object / place_object ----------------

def test_skills_pick_object_gated_off_never_calls_arm(monkeypatch):
    """Gate off: the real arm primitive is never invoked; returns False (not held)."""
    from tasks.PickAndPlace import skills

    called = {"n": 0}
    monkeypatch.setattr(skills, "_pick_object", lambda ctx, o: called.__setitem__("n", called["n"] + 1) or True)
    monkeypatch.setenv("PNP_ARM_CALIBRATED", "0")
    ctx = FakeCtx()

    assert skills.pick_object(ctx, _obj("cup")) is False
    assert called["n"] == 0
    assert any("not enabled" in s for s in ctx.said)  # announced the gated indication


def test_skills_pick_object_gated_on_calls_arm(monkeypatch):
    from tasks.PickAndPlace import skills

    called = {"n": 0}
    monkeypatch.setattr(skills, "_pick_object", lambda ctx, o: called.__setitem__("n", called["n"] + 1) or True)
    monkeypatch.setenv("PNP_ARM_CALIBRATED", "1")
    ctx = FakeCtx()

    assert skills.pick_object(ctx, _obj("cup")) is True
    assert called["n"] == 1


def test_skills_place_object_gated_off_no_nav(monkeypatch):
    """Gate off: place_object no-ops without navigating to the furniture."""
    from tasks.PickAndPlace import skills

    monkeypatch.setenv("PNP_ARM_CALIBRATED", "0")
    ctx = FakeCtx()

    assert skills.place_object(ctx, "dishwasher") is False
    assert ctx.gotos == []
