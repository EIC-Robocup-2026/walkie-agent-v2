"""Offline end-to-end integration of GPSR dispatch + the real Tier-1 skills.

The pure unit tests (test_gpsr_parse / _dispatch / _gestures) cover grounding and
routing *policy* in isolation. This file closes the seam between them: it drives
the **real** `execute_plan` over the **real** skill bodies with a mock
TaskContext, so parser↔skill integration bugs that neither side sees alone are
caught — chiefly that an explicit `navigate` step followed by a find/count/greet
naming the same place does NOT make the robot drive there twice.

It runs with no robot and no LLM (the dev box has no CUDA): the heavy hardware
imports are kept out of `tasks.base` (type-only), the two geometry helpers
`skills.py` borrows from HRI are monkeypatched, and every perception/LLM call is
served by a fake. Plans are built with the real `ground_step` (pure) so the
grounding→dispatch→skill chain is exercised exactly as on the robot.
"""

from __future__ import annotations

import textwrap

import pytest

from client.object_detection import DetectedObject
from client.pose_estimation import PersonPose, PoseKeypoint
from tasks.GPSR import skills
from tasks.GPSR.dispatch import execute_plan
from tasks.GPSR.parse import ground_step
from tasks.GPSR.plan import CmdStatus, Plan
from tasks.GPSR.prompts import RawStep
from tasks.GPSR.world import load_world

# --- fixtures: a small arena with DISTINCT poses (so "which place" is provable) -

_WORLD_TOML = textwrap.dedent(
    """
    names = ["Charlie", "Robin"]

    [rooms]
    kitchen     = { pose = [1.0, 2.0, 0.0] }
    living_room = { pose = [3.0, 4.0, 1.5], aliases = ["living room", "lounge"] }
    office      = { pose = [7.0, 8.0, 0.0] }

    [locations]
    kitchen_table = { room = "kitchen", placement = true, category = "dishes", pose = [1.5, 2.5, 0.0], aliases = ["kitchen table"] }
    desk          = { room = "office",  placement = true, pose = [5.0, 6.0, 0.0] }

    [object_categories]
    drinks = ["cola", "milk"]
    dishes = ["cup", "bowl", "plate"]

    [gestures]
    waving  = { aliases = ["waving person", "person waving"] }
    sitting = { aliases = ["sitting person", "person sitting"] }
    """
)


@pytest.fixture
def world(tmp_path):
    p = tmp_path / "world.toml"
    p.write_text(_WORLD_TOML)
    return load_world(p)


# --- fakes ------------------------------------------------------------------

class _FakeImg:
    width = 640
    height = 480

    def crop(self, box):  # get_*_property/_person_info crop before captioning
        return self


class _FakeSnap:
    def __init__(self):
        self.img = _FakeImg()


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    """Stand-in ChatOpenAI: records prompts, returns a canned line (or raises)."""

    def __init__(self, reply="It is a lovely day.", raise_on_invoke=False):
        self.reply = reply
        self.raise_on_invoke = raise_on_invoke
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raise_on_invoke:
            raise RuntimeError("stub model offline")
        return _FakeMsg(self.reply)


class _FakeDetector:
    def __init__(self, dets):
        self.dets = dets
        self.prompts_seen: list = []

    def detect(self, img, prompts=None, return_mask=False):
        self.prompts_seen.append(prompts)
        return list(self.dets)


class _FakePoseEst:
    def __init__(self, people):
        self.people = people

    def estimate(self, img):
        return list(self.people)


class _FakeCaptioner:
    def __init__(self, text="a blue striped shirt"):
        self.text = text

    def caption(self, img, prompt=None):
        return self.text


class _FakeAI:
    def __init__(self, dets=(), people=(), caption="a blue striped shirt"):
        self.object_detection = _FakeDetector(list(dets))
        self.pose_estimation = _FakePoseEst(list(people))
        self.image_caption = _FakeCaptioner(caption)


class _FakeBrain:
    """Tier-2 stand-in: records the clauses dispatch hands to the agent stack."""

    def __init__(self):
        self.clauses: list[str] = []

        outer = self

        class _Agent:
            def invoke(self, payload, config=None):
                outer.clauses.append(payload["messages"][0].content)
                return {"messages": []}

        self.walkie_agent = _Agent()


class _FakeCtx:
    """Mock TaskContext recording the robot's outward actions."""

    def __init__(self, *, ai=None, model=None, snap=True, goto_ok=True, ask_reply=""):
        self.walkieAI = ai or _FakeAI()
        self.model = model or _FakeModel()
        self._snap = _FakeSnap() if snap else None
        self._goto_ok = goto_ok
        self._ask_reply = ask_reply
        self.gotos: list[tuple[float, float, float]] = []
        self.saids: list[str] = []
        self.asked: list[str] = []

    def goto(self, x, y, h):
        self.gotos.append((x, y, h))
        return self._goto_ok

    def snapshot(self):
        return self._snap

    def say(self, text):
        self.saids.append(text)

    def ask(self, question, retries=1):
        self.asked.append(question)
        return self._ask_reply


# --- helpers ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_geometry(monkeypatch):
    """Stub the two HRI geometry helpers skills.py borrows (need depth + odom).

    Keeps the test on control flow: lift always yields a point, facing is a
    recorded no-op. The lift/face math itself is HRI's to test.
    """
    faced: list[tuple[float, float]] = []
    monkeypatch.setattr(skills, "lift_bbox_world_xy", lambda ctx, snap, bbox: (9.0, 9.0))

    def _face(ctx, x, y):
        faced.append((x, y))
        return True

    monkeypatch.setattr(skills, "face_point", _face)
    return faced


def _kp(name, x, y, c=0.9):
    return PoseKeypoint(x=x, y=y, confidence=c, name=name, index=0)


def _waving_person(bbox=(320, 240, 80, 160)):
    """A pose whose left wrist is well above its shoulder (→ waving/raised arm)."""
    return PersonPose(
        bbox=bbox,
        confidence=0.9,
        keypoints=[
            _kp("left_shoulder", 100, 100),
            _kp("left_hip", 100, 200),
            _kp("left_wrist", 100, 40),  # 60px above shoulder, torso=100 → raised
        ],
    )


def _det(bbox=(0, 0, 10, 10), conf=0.9, name="cola"):
    return DetectedObject(mask=None, bbox=bbox, area_ratio=0.01, class_name=name, confidence=conf)


def _run(ctx, world, *raw_steps, brain=None, manip=False):
    plan = Plan(steps=[ground_step(r, world) for r in raw_steps], source="test")
    return plan, execute_plan(ctx, plan, world, brain, manip_enabled=manip)


# --- navigation dedup (the headline integration concern) --------------------

def test_navigate_then_find_object_does_not_double_navigate(world):
    """navigate(kitchen) + find_object(cola, room=kitchen) → ONE drive, not two."""
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()]))
    plan, status = _run(
        ctx, world,
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        RawStep(primitive="find_object", object="cola", room="kitchen", raw="find a cola"),
    )
    assert ctx.gotos == [(1.0, 2.0, 0.0)]  # kitchen pose, exactly once
    assert status is CmdStatus.DONE
    assert any("found the cola" in s for s in ctx.saids)


def test_find_object_alone_navigates_when_it_carries_the_location(world):
    """With no preceding navigate, find_object must drive to its own location."""
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()]))
    _run(ctx, world, RawStep(primitive="find_object", object="cola", room="kitchen", raw="find a cola in the kitchen"))
    assert ctx.gotos == [(1.0, 2.0, 0.0)]


def test_find_object_without_a_place_does_not_navigate(world):
    """A bare find_object (parser split the nav into its own step) drives nowhere."""
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()]))
    _run(ctx, world, RawStep(primitive="find_object", object="cola", raw="find the cola"))
    assert ctx.gotos == []


def test_distinct_places_are_not_deduped(world):
    """kitchen then kitchen_table are different targets → two drives."""
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()]))
    _run(
        ctx, world,
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        RawStep(primitive="find_object", object="cup", location="kitchen table", raw="find a cup on the kitchen table"),
    )
    assert ctx.gotos == [(1.0, 2.0, 0.0), (1.5, 2.5, 0.0)]


def test_repeated_navigate_same_room_deduped(world):
    ctx = _FakeCtx()
    _run(
        ctx, world,
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen again"),
    )
    assert ctx.gotos == [(1.0, 2.0, 0.0)]


# --- other no-arm primitives over the real skill bodies ---------------------

def test_count_objects_reports_count_and_navigates_once(world):
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det(name="cup"), _det(name="cup")]))
    _run(ctx, world, RawStep(primitive="count", object="cups", location="kitchen table", raw="count the cups on the kitchen table"))
    assert ctx.gotos == [(1.5, 2.5, 0.0)]
    assert any("count 2" in s for s in ctx.saids)


def test_find_person_by_gesture_matches_and_faces(world, _patch_geometry):
    ctx = _FakeCtx(ai=_FakeAI(people=[_waving_person()]))
    _, status = _run(ctx, world, RawStep(primitive="find_person", person="waving person", descriptor_kind="gesture", raw="find the waving person"))
    assert status is CmdStatus.DONE
    assert any("found you" in s.lower() for s in ctx.saids)
    assert _patch_geometry == [(9.0, 9.0)]  # faced the lifted point


def test_find_person_absent_is_an_honest_negative_not_a_failure(world):
    """No people detected → the skill still 'ran' (speaks, returns True → DONE)."""
    ctx = _FakeCtx(ai=_FakeAI(people=[]))
    _, status = _run(ctx, world, RawStep(primitive="find_person", person="Charlie", descriptor_kind="name", raw="find Charlie"))
    assert status is CmdStatus.DONE
    assert any("could not find" in s.lower() for s in ctx.saids)


def test_get_object_property_category_uses_world_model_no_perception(world):
    """category is known from the world model — answered with zero detections."""
    ctx = _FakeCtx(ai=_FakeAI(dets=[]))
    _run(ctx, world, RawStep(primitive="get_object_property", object="cola", which="category", raw="what category is the cola"))
    assert ctx.gotos == []
    assert any("drinks" in s for s in ctx.saids)


# --- say / tell knowledge source --------------------------------------------

def test_say_routes_through_llm_with_known_facts(world):
    model = _FakeModel(reply="Today is a great day at RoboCup.")
    ctx = _FakeCtx(model=model)
    _run(ctx, world, RawStep(primitive="say", info="what day is it", raw="tell me what day it is"))
    assert ctx.saids == ["Today is a great day at RoboCup."]
    # The prompt was grounded with identity + the live clock, not bare.
    sent = model.calls[0][0].content
    assert "RoboCup@Home" in sent and "what day is it" in sent


def test_say_falls_back_to_literal_when_the_llm_fails(world):
    ctx = _FakeCtx(model=_FakeModel(raise_on_invoke=True))
    _run(ctx, world, RawStep(primitive="say", info="hello everyone", raw="say hello to everyone"))
    assert ctx.saids == ["hello everyone"]


# --- Tier-2 fallback routing ------------------------------------------------

def test_gated_manipulation_falls_through_to_tier2(world):
    """pick is gated off (no arm) → dispatch hands the clause to the agent stack."""
    ctx = _FakeCtx()
    brain = _FakeBrain()
    _, status = _run(ctx, world, RawStep(primitive="pick", object="cola", raw="pick up the cola"), brain=brain, manip=False)
    assert ctx.gotos == []  # no Tier-1 skill ran
    assert brain.clauses == ["pick up the cola"]
    assert status is CmdStatus.DONE  # Tier-2 reported handled


def test_gated_manipulation_without_a_brain_is_a_failure(world):
    ctx = _FakeCtx()
    _, status = _run(ctx, world, RawStep(primitive="pick", object="cola", raw="pick up the cola"), brain=None, manip=False)
    assert status is CmdStatus.FAILED


def test_tier2_invalidates_the_nav_cache(world):
    """A Tier-2 step between two navigate(kitchen)s must NOT dedup the second.

    The agent fallback can drive the robot anywhere, so the deterministic
    `state["at"]` cache is stale afterwards — the robot must re-navigate.
    """
    ctx = _FakeCtx()
    brain = _FakeBrain()
    _run(
        ctx, world,
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        RawStep(primitive="pick", object="cola", raw="pick up the cola"),  # gated -> Tier-2
        RawStep(primitive="navigate", room="kitchen", raw="go back to the kitchen"),
        brain=brain, manip=False,
    )
    assert brain.clauses == ["pick up the cola"]
    assert ctx.gotos == [(1.0, 2.0, 0.0), (1.0, 2.0, 0.0)]  # drove to kitchen twice


def test_mixed_plan_with_one_failing_step_is_partial(world):
    """navigate ok + gated pick with no brain → PARTIAL (partial scoring)."""
    ctx = _FakeCtx()
    _, status = _run(
        ctx, world,
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        RawStep(primitive="pick", object="cola", raw="pick up the cola"),
        brain=None, manip=False,
    )
    assert status is CmdStatus.PARTIAL
