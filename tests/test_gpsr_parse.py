"""Offline unit tests for the GPSR parser's pure core — no LLM, no robot.

Covers the world-model grounding, RawStep -> PlanStep grounding for every
primitive, and the deterministic plan -> speech render. The LLM end of the
parser (utterance -> RawPlan) is exercised separately by the coverage harness
(test_gpsr_coverage.py), which needs the model.
"""

from __future__ import annotations

import pytest

from tasks.GPSR.parse import ground_plan, ground_step
from tasks.GPSR.plan import Plan, PlanStep, Primitive, render_plan_speech
from tasks.GPSR.prompts import RawPlan, RawStep
from tasks.GPSR.world import load_world


@pytest.fixture(scope="module")
def world():
    return load_world()  # the default world.toml (CompetitionTemplate arena)


# --- world model ------------------------------------------------------------

def test_world_loads_vocabulary(world):
    assert "kitchen" in world.rooms
    assert "kitchen_table" in world.locations
    assert world.objects["cola"] == "drinks"
    assert "Charlie" in world.names
    assert "waving" in world.gestures


@pytest.mark.parametrize("text,expected", [
    ("the kitchen table", "kitchen_table"),
    ("Kitchen Table", "kitchen_table"),
    ("nightstand", "bedside_table"),     # alias
    ("the fridge", "refrigerator"),      # alias + article
    ("couch", "sofa"),
    ("nowhere in particular", None),
])
def test_location_grounding(world, text, expected):
    assert world.location(text) == expected


def test_room_alias_and_article(world):
    assert world.room("the living room") == "living_room"
    assert world.room("lounge") == "living_room"
    assert world.room("study") == "office"


def test_object_and_category_grounding(world):
    assert world.obj("the cola") == "cola"
    assert world.obj("orange juice") == "orange_juice"
    assert world.category("drinks") == "drinks"
    assert world.category("drink") == "drinks"      # singular
    # A bare category reference yields a concrete pickable item.
    assert world.obj("a drink") in world.categories["drinks"]
    assert world.obj("a unicorn") is None


def test_name_and_gesture_grounding(world):
    assert world.name("charlie") == "Charlie"
    assert world.gesture("waving person") == "waving"
    assert world.gesture("person raising their left arm") == "raising_left_arm"


def test_vocab_prompt_lists_canonical_terms(world):
    p = world.vocab_prompt()
    assert "cola" in p and "refrigerator" in p and "living room" in p
    assert "Charlie" in p and "waving" in p
    assert "coke" in p.lower()  # the synonym-mapping hint is present


# --- step grounding ---------------------------------------------------------

def _step(primitive, **fields):
    fields.setdefault("raw", primitive)
    return ground_step(RawStep(primitive=primitive, **fields), load_world())


def test_navigate_grounds_room_and_location():
    assert _step("navigate", room="the kitchen").args["target"] == "kitchen"
    assert _step("navigate", location="the sofa").args["target"] == "sofa"


def test_takeobj_decomposition_grounds():
    s = _step("pick", object="the cola", location="the cabinet")
    assert s.grounded
    assert s.args == {"object": "cola", "location": "cabinet"}


def test_deliver_to_me():
    s = _step("deliver", object="a coke", recipient="me")
    # "coke" is not in the vocabulary -> object unresolved, recipient is me.
    assert s.args["recipient"] == "me"


def test_find_person_by_gesture():
    s = _step("find_person", person="waving person", descriptor_kind="gesture", room="the kitchen")
    assert s.grounded
    assert s.args["descriptor"] == "waving"
    assert s.args["kind"] == "gesture"
    assert s.args["room"] == "kitchen"


def test_find_person_clothing_is_open_vocab():
    s = _step("find_person", person="the person in the red shirt", descriptor_kind="clothing")
    assert s.grounded  # clothing descriptions never fail to ground
    assert s.args["kind"] == "clothing"


def test_unknown_location_is_unresolved():
    s = _step("navigate", location="the dungeon")
    assert not s.grounded
    assert ("target", "the dungeon") in s.unresolved


def test_unknown_primitive_is_unresolved():
    s = ground_step(RawStep(primitive="say", info="hi", raw="hi"), load_world())
    assert s.grounded  # valid say
    # Force an invalid primitive via the enum path:
    bad = PlanStep(Primitive.SAY, {}, "x", [("primitive", "teleport")])
    assert not bad.grounded


def test_object_property_which_validation():
    assert _step("get_object_property", object="the cola", which="weight").grounded
    bad = _step("get_object_property", object="the cola", which="flavour")
    assert ("which", "flavour") in bad.unresolved


def test_superlative_query_object_grounds_as_placement_scoped():
    # "tell me the biggest object on the desk" — object is a query placeholder,
    # not a vocab item; it must NOT count as an ungrounded gap.
    raw = RawPlan(steps=[
        RawStep(primitive="navigate", location="the desk", raw="go to the desk"),
        RawStep(primitive="find_object", object="object", location="the desk", raw="find the objects"),
        RawStep(primitive="get_object_property", object="object", which="size", raw="the biggest one"),
        RawStep(primitive="say", info="the biggest object", raw="tell me"),
    ])
    plan = ground_plan(raw, load_world())
    assert plan.is_complete, [u for s in plan.steps for u in s.unresolved]


# --- whole-plan grounding + completeness ------------------------------------

def test_complete_plan_is_complete():
    raw = RawPlan(steps=[
        RawStep(primitive="navigate", room="the kitchen", raw="go to the kitchen"),
        RawStep(primitive="find_object", object="the cola", raw="find the cola"),
        RawStep(primitive="pick", object="the cola", raw="pick it up"),
        RawStep(primitive="deliver", object="the cola", recipient="me", raw="bring it to me"),
    ])
    plan = ground_plan(raw, load_world(), source="get me a coke from the kitchen")
    assert plan.is_complete
    assert plan.grounded_fraction == 1.0
    assert plan.needs_manipulation


def test_incomplete_plan_flags_gap():
    raw = RawPlan(steps=[RawStep(primitive="navigate", location="the dungeon", raw="go to the dungeon")])
    plan = ground_plan(raw, load_world())
    assert not plan.is_complete
    assert plan.grounded_fraction == 0.0


def test_empty_plan_not_complete():
    assert not Plan(steps=[]).is_complete


# --- plan -> speech ---------------------------------------------------------

def test_render_plan_speech_orders_clauses():
    raw = RawPlan(steps=[
        RawStep(primitive="navigate", room="the kitchen", raw="go to the kitchen"),
        RawStep(primitive="find_object", object="the cola", raw="find the cola"),
        RawStep(primitive="deliver", object="the cola", recipient="me", raw="bring it to me"),
    ])
    plan = ground_plan(raw, load_world())
    speech = render_plan_speech(plan)
    assert speech.startswith("Here is my plan.")
    assert "First I will go to the kitchen" in speech
    assert "then find the cola" in speech
    assert "finally bring the cola to you" in speech


def test_render_names_the_person():
    raw = RawPlan(steps=[
        RawStep(primitive="find_person", person="Charlie", descriptor_kind="name", raw="find charlie"),
        RawStep(primitive="guide", person="Charlie", descriptor_kind="name", to_location="the exit", raw="guide charlie to the exit"),
    ])
    speech = render_plan_speech(ground_plan(raw, load_world()))
    assert "find Charlie" in speech
    assert "guide Charlie to the exit" in speech


def test_render_gesture_person_as_the_x_person():
    raw = RawPlan(steps=[RawStep(primitive="find_person", person="waving person", descriptor_kind="gesture", raw="find the waver")])
    speech = render_plan_speech(ground_plan(raw, load_world()))
    assert "the waving person" in speech


def test_render_single_step():
    raw = RawPlan(steps=[RawStep(primitive="navigate", room="the bedroom", raw="go to the bedroom")])
    speech = render_plan_speech(ground_plan(raw, load_world()))
    assert speech == "Here is my plan. I will go to the bedroom."


def test_render_empty_plan_apologizes():
    assert "could not" in render_plan_speech(Plan(steps=[])).lower()
