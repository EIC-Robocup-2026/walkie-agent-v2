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

from client.image import DetectedObject, PersonPose, PoseKeypoint
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


class _FakeImage:
    """Stand-in for the unified ImageClient (walkieAI.image)."""

    def __init__(self, dets=(), people=(), caption="a blue striped shirt"):
        self.dets = list(dets)
        self.people = list(people)
        self.caption_text = caption
        self.prompts_seen: list = []

    def detect(self, img, *, prompts=None, return_mask=False):
        self.prompts_seen.append(prompts)
        return list(self.dets)

    def estimate_poses(self, img):
        return list(self.people)

    def caption(self, img, *, prompt=None):
        return self.caption_text


class _FakeAI:
    def __init__(self, dets=(), people=(), caption="a blue striped shirt"):
        self.image = _FakeImage(dets, people, caption)


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

    def __init__(self, *, ai=None, model=None, snap=True, goto_ok=True, ask_reply="", pose=(0.0, 0.0)):
        self.walkieAI = ai or _FakeAI()
        self.model = model or _FakeModel()
        self._snap = _FakeSnap() if snap else None
        self._goto_ok = goto_ok
        self._ask_reply = ask_reply
        self._pose = pose
        self.gotos: list[tuple[float, float, float]] = []
        self.rotations: list[float] = []
        self.saids: list[str] = []
        self.asked: list[str] = []
        self.data: dict = {}  # ctx.data["brain"].graphs is the scene memory (run.py)

    def goto(self, x, y, h):
        self.gotos.append((x, y, h))
        return self._goto_ok

    def current_pose(self):
        return {"x": self._pose[0], "y": self._pose[1], "heading": 0.0}

    def rotate_to(self, heading_rad):
        self.rotations.append(heading_rad)
        return True

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


def _plain_person(bbox):
    """A pose with no informative keypoints (clothing match ignores gesture)."""
    return PersonPose(bbox=bbox, confidence=0.9, keypoints=[])


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


# --- find_object scene-memory fallback (option A) ---------------------------

class _RecallGraphs:
    """Fake walkie_graphs: query_text returns one node at a fixed centroid."""

    def __init__(self, centroid=(3.0, 4.0, 0.5)):
        self._c = centroid
        self.queries: list[str] = []

    def query_text(self, query, k=1):
        self.queries.append(query)
        return [type("N", (), {"centroid": self._c})()]


class _ArriveThenSeeImage:
    """Detect returns nothing until approach_point flips state['arrived'] True —
    i.e. the object is only visible once the robot has driven to the recalled spot."""

    def __init__(self, state, name="cola"):
        self.state, self.name = state, name

    def detect(self, img, *, prompts=None, return_mask=False):
        return [_det(name=self.name)] if self.state["arrived"] else []

    def estimate_poses(self, img):
        return []

    def caption(self, img, *, prompt=None):
        return ""


def test_find_object_no_location_recalls_from_scene_memory(world, monkeypatch):
    """No location named -> ask the scene graph where the cola is, drive there
    (approach_point), and confirm with a live detect. Position comes from memory."""
    monkeypatch.setenv("GPSR_FIND_USE_MEMORY", "1")
    state = {"arrived": False}
    approached: list[tuple[float, float]] = []
    monkeypatch.setattr(skills, "approach_point",
                        lambda ctx, x, y, **kw: (approached.append((x, y)), state.__setitem__("arrived", True), True)[-1])
    graphs = _RecallGraphs((3.0, 4.0, 0.5))
    ctx = _FakeCtx(ai=type("AI", (), {"image": _ArriveThenSeeImage(state)})())
    ctx.data["brain"] = type("B", (), {"graphs": graphs})()
    _, status = _run(ctx, world, RawStep(primitive="find_object", object="cola", raw="find the cola"))
    assert status is CmdStatus.DONE
    assert graphs.queries == ["cola"]        # consulted the scene memory
    assert approached == [(3.0, 4.0)]        # drove to the recalled position
    assert any("found the cola" in s for s in ctx.saids)


def test_find_object_at_named_place_does_not_touch_memory(world, monkeypatch):
    """Command names a place and the object is there -> the scene graph is never
    queried (the operator's location wins — option A)."""
    monkeypatch.setenv("GPSR_FIND_USE_MEMORY", "1")
    graphs = _RecallGraphs()
    ctx = _FakeCtx(ai=_FakeAI(dets=[_det(name="cola")]))
    ctx.data["brain"] = type("B", (), {"graphs": graphs})()
    _, status = _run(ctx, world, RawStep(
        primitive="find_object", object="cola", room="living_room",
        raw="find the cola in the living room"))
    assert status is CmdStatus.DONE
    assert graphs.queries == []              # memory NOT consulted
    assert ctx.gotos == [(3.0, 4.0, 1.5)]    # only the living_room drive
    assert any("found the cola" in s for s in ctx.saids)


def test_find_object_falls_back_to_memory_when_named_place_empty(world, monkeypatch):
    """Named place comes up empty -> recall from the scene graph and confirm."""
    monkeypatch.setenv("GPSR_FIND_USE_MEMORY", "1")
    state = {"arrived": False}
    approached: list[tuple[float, float]] = []
    monkeypatch.setattr(skills, "approach_point",
                        lambda ctx, x, y, **kw: (approached.append((x, y)), state.__setitem__("arrived", True), True)[-1])
    graphs = _RecallGraphs((7.0, 1.0, 0.3))
    ctx = _FakeCtx(ai=type("AI", (), {"image": _ArriveThenSeeImage(state)})())
    ctx.data["brain"] = type("B", (), {"graphs": graphs})()
    _, status = _run(ctx, world, RawStep(
        primitive="find_object", object="cola", room="living_room",
        raw="find the cola in the living room"))
    assert status is CmdStatus.DONE
    assert ctx.gotos[0] == (3.0, 4.0, 1.5)   # first drove to the named place
    assert graphs.queries == ["cola"]        # then recalled from memory
    assert approached == [(7.0, 1.0)]        # and approached the recalled spot
    assert any("found the cola" in s for s in ctx.saids)


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
    # Speech is now descriptor-specific (honest), not a generic "found you".
    assert any("found the waving person" in s.lower() for s in ctx.saids)
    assert _patch_geometry == [(9.0, 9.0)]  # faced the lifted point


def test_find_person_absent_is_an_honest_negative_not_a_failure(world):
    """No people detected → the skill still 'ran' (speaks, returns True → DONE)."""
    ctx = _FakeCtx(ai=_FakeAI(people=[]))
    _, status = _run(ctx, world, RawStep(primitive="find_person", person="Charlie", descriptor_kind="name", raw="find Charlie"))
    assert status is CmdStatus.DONE
    assert any("could not find" in s.lower() for s in ctx.saids)


def test_find_person_by_clothing_picks_the_llm_chosen_candidate(world, monkeypatch):
    """Clothing isn't re-ID (no gallery): caption each person, LLM picks the index.

    Two visible people; the model replies '1' → the *second* person's box is the
    one faced. Asserts the bbox lifted is candidate 1's, not just the nearest.
    """
    lifted: list[tuple] = []
    monkeypatch.setattr(skills, "lift_bbox_world_xy",
                        lambda ctx, snap, bbox: lifted.append(tuple(bbox)) or (9.0, 9.0))
    p0 = _plain_person((100, 100, 40, 120))
    p1 = _plain_person((400, 100, 40, 120))
    ctx = _FakeCtx(ai=_FakeAI(people=[p0, p1]), model=_FakeModel(reply="1"))
    _, status = _run(ctx, world, RawStep(
        primitive="find_person", person="person in a red shirt",
        descriptor_kind="clothing", raw="find the person in a red shirt"))
    assert status is CmdStatus.DONE
    assert lifted == [skills.cxcywh_to_xyxy(p1.bbox)]  # candidate 1, by LLM choice
    assert any("found" in s.lower() for s in ctx.saids)


def test_find_person_by_clothing_no_match_is_honest(world):
    """People are present but none match the attire → '-1' → honest negative."""
    ctx = _FakeCtx(
        ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]),
        model=_FakeModel(reply="-1"),
    )
    _, status = _run(ctx, world, RawStep(
        primitive="find_person", person="person in a red shirt",
        descriptor_kind="clothing", raw="find the person in a red shirt"))
    assert status is CmdStatus.DONE
    assert any("could not find" in s.lower() for s in ctx.saids)


def test_follow_with_destination_uses_arrival_stopper(world, monkeypatch):
    """`follow ... to X` is Tier-1 (no brain) over HRI's follow_person with
    select_largest_person, and passes an ArrivalStopper so the loop ends on
    arrival ('stopped') -> arrival announce. We stub the loop (it's HRI's to test)."""
    calls = {}

    def _fake_follow(ctx, select, *, stopper=None, on_warmup=None, **kw):
        calls["select"] = select
        calls["stopper"] = stopper
        if on_warmup:
            on_warmup()
        return "stopped"  # simulate the arrival stopper firing

    monkeypatch.setattr(skills, "follow_person", _fake_follow)
    ctx = _FakeCtx(ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]))
    _, status = _run(ctx, world, RawStep(
        primitive="follow", person="me", to_location="kitchen",
        raw="follow me to the kitchen"))
    assert status is CmdStatus.DONE                       # Tier-1, not a Tier-2 miss
    assert calls["select"] is skills.select_largest_person
    assert calls["stopper"] is not None                   # ArrivalStopper wired for the destination
    assert any("follow you" in s.lower() for s in ctx.saids)         # warmup ack
    assert any("arrived at kitchen" in s.lower() for s in ctx.saids)  # arrival on 'stopped'


def test_follow_without_destination_has_no_stopper(world, monkeypatch):
    """`follow me` (no destination) passes no stopper and does not claim arrival."""
    calls = {}

    def _fake_follow(ctx, select, *, stopper=None, on_warmup=None, **kw):
        calls["stopper"] = stopper
        return "timeout"

    monkeypatch.setattr(skills, "follow_person", _fake_follow)
    ctx = _FakeCtx(ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]))
    _, status = _run(ctx, world, RawStep(primitive="follow", person="me", raw="follow me"))
    assert status is CmdStatus.DONE
    assert calls["stopper"] is None                       # no destination -> no arrival stopper
    assert any("stopped following" in s.lower() for s in ctx.saids)  # not a false 'arrived'


def test_guide_leads_person_from_start_to_destination(world, _patch_geometry):
    """guide: drive to `from`, confirm/face the person, lead to `to`, announce."""
    ctx = _FakeCtx(ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]))
    _, status = _run(ctx, world, RawStep(
        primitive="guide", person="Charlie", descriptor_kind="name",
        from_location="office", to_location="kitchen",
        raw="guide Charlie from the office to the kitchen"))
    assert status is CmdStatus.DONE
    assert ctx.gotos == [(7.0, 8.0, 0.0), (1.0, 2.0, 0.0)]  # office (from) then kitchen (to)
    assert _patch_geometry == [(9.0, 9.0)]                  # faced the person
    assert any("hello charlie" in s.lower() for s in ctx.saids)
    assert any("guide you to kitchen" in s.lower() for s in ctx.saids)
    assert any("arrived at kitchen" in s.lower() for s in ctx.saids)


def test_guide_reacquire_leads_in_segments_and_looks_back(world, _patch_geometry, monkeypatch):
    """With GPSR_GUIDE_REACQUIRE on, guide drives to `to` in capped hops and turns
    back between them (look-back). A visible follower → no waiting; final hop lands
    on the destination pose."""
    monkeypatch.setenv("GPSR_GUIDE_REACQUIRE", "1")
    monkeypatch.setenv("GPSR_GUIDE_SEGMENT_M", "1.0")
    # start at the origin; kitchen is (1,2) → dist 2.24 → ceil(2.24/1)=3 hops.
    ctx = _FakeCtx(ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]), pose=(0.0, 0.0))
    _, status = _run(ctx, world, RawStep(
        primitive="guide", person="Charlie", descriptor_kind="name",
        to_location="kitchen", raw="guide Charlie to the kitchen"))
    assert status is CmdStatus.DONE
    assert len(ctx.gotos) == 3                       # segmented, not one blocking drive
    assert ctx.gotos[-1] == (1.0, 2.0, 0.0)          # final hop = kitchen surveyed pose
    assert len(ctx.rotations) == 2                   # looked back after each non-final hop
    assert any("arrived at kitchen" in s.lower() for s in ctx.saids)


def test_guide_reacquire_leads_on_best_effort_when_follower_lost(world, _patch_geometry, monkeypatch):
    """Follower never re-appears at a look-back → prompt + wait (bounded), then lead
    on. Must still reach the destination (DONE), never hang."""
    monkeypatch.setenv("GPSR_GUIDE_REACQUIRE", "1")
    monkeypatch.setenv("GPSR_GUIDE_SEGMENT_M", "1.0")
    monkeypatch.setenv("GPSR_GUIDE_MAX_MISSES", "2")
    monkeypatch.setattr(skills.time, "sleep", lambda *_: None)  # don't actually wait
    ctx = _FakeCtx(ai=_FakeAI(people=[]), pose=(0.0, 0.0))      # nobody ever visible
    _, status = _run(ctx, world, RawStep(
        primitive="guide", person="Charlie", descriptor_kind="name",
        to_location="kitchen", raw="guide Charlie to the kitchen"))
    assert status is CmdStatus.DONE
    assert ctx.gotos[-1] == (1.0, 2.0, 0.0)          # reached the destination anyway
    assert any("keep up" in s.lower() for s in ctx.saids)  # asked the follower to catch up
    assert any("arrived at kitchen" in s.lower() for s in ctx.saids)


def test_guide_unreachable_destination_falls_back_to_tier2(world):
    """If the lead-nav fails, guide returns False so the agent stack (Tier-2) tries."""
    brain = _FakeBrain()
    ctx = _FakeCtx(ai=_FakeAI(people=[_plain_person((100, 100, 40, 120))]), goto_ok=False)
    _, status = _run(ctx, world, RawStep(
        primitive="guide", person="Charlie", descriptor_kind="name",
        to_location="kitchen", raw="guide Charlie to the kitchen"), brain=brain)
    assert brain.clauses                       # clause handed to Tier-2
    assert status is CmdStatus.DONE            # Tier-2 stub 'handled' it


def test_guide_with_no_person_visible_still_leads_open_loop(world, _patch_geometry):
    """Person not in the first frame (the realistic case): guide leads anyway —
    no facing, but still announces and reaches `to` (DONE). This locks the
    documented 'no follow-back tracking' degraded behaviour as intended."""
    ctx = _FakeCtx(ai=_FakeAI(people=[]))
    _, status = _run(ctx, world, RawStep(
        primitive="guide", person="Charlie", descriptor_kind="name",
        to_location="kitchen", raw="guide Charlie to the kitchen"))
    assert status is CmdStatus.DONE
    assert ctx.gotos == [(1.0, 2.0, 0.0)]      # led to kitchen (no `from` given)
    assert _patch_geometry == []                # nobody seen -> nobody faced
    assert any("guide you to kitchen" in s.lower() for s in ctx.saids)
    assert any("arrived at kitchen" in s.lower() for s in ctx.saids)


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


# --- interleave (the bonus path) --------------------------------------------

def _kitchen_plan(world, src, second):
    return Plan(steps=[ground_step(r, world) for r in (
        RawStep(primitive="navigate", room="kitchen", raw="go to the kitchen"),
        second,
    )], source=src)


def test_interleave_shares_nav_dedup_across_commands(world):
    """The point of interleaving: a room entered for one command is not re-entered
    for another. Two commands both starting in the kitchen -> kitchen driven ONCE
    (serial gives each command its own state and would drive there twice)."""
    from tasks.GPSR.dispatch import execute_interleaved
    from tasks.GPSR.schedule import interleave

    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()], people=[_plain_person((100, 100, 40, 120))]))
    p1 = _kitchen_plan(world, "c1", RawStep(primitive="find_object", object="cola", room="kitchen", raw="find the cola"))
    p2 = _kitchen_plan(world, "c2", RawStep(primitive="count", object="cup", room="kitchen", raw="count the cups"))
    indexed = [(1, p1), (2, p2)]
    statuses = execute_interleaved(ctx, indexed, world, None, manip_enabled=False, order=interleave(indexed, world))
    assert ctx.gotos == [(1.0, 2.0, 0.0)]            # kitchen exactly once
    assert statuses[1] is CmdStatus.DONE
    assert statuses[2] is CmdStatus.DONE


def test_execute_commands_uses_interleave_when_enabled(world, monkeypatch):
    """GPSR_INTERLEAVE=1 + >=2 planned commands -> ExecuteCommands interleaves:
    announces it, and the shared nav-dedup drives the shared kitchen once."""
    monkeypatch.setenv("GPSR_INTERLEAVE", "1")
    from tasks.GPSR.subtasks import Command, ExecuteCommands

    ctx = _FakeCtx(ai=_FakeAI(dets=[_det()], people=[_plain_person((100, 100, 40, 120))]))
    p1 = _kitchen_plan(world, "c1", RawStep(primitive="find_object", object="cola", room="kitchen", raw="find the cola"))
    p2 = _kitchen_plan(world, "c2", RawStep(primitive="find_object", object="cola", room="kitchen", raw="find the cola"))
    cmds = [Command(1, "c1", p1), Command(2, "c2", p2)]
    ctx.data = {"world": world, "brain": None, "commands": cmds}
    ExecuteCommands().run(ctx)
    assert any("interleav" in s.lower() for s in ctx.saids)   # announced the interleave
    assert ctx.gotos == [(1.0, 2.0, 0.0)]                     # kitchen once, not per-command
    assert all(c.status is CmdStatus.DONE for c in cmds)


def test_interleave_isolates_per_command_scratch(world, monkeypatch):
    """Only the nav location is global across commands; per-command scratch is
    isolated — so an interleaved step of command B cannot clobber the target
    command A stashed for its own next step. A probe skill records the scratch it
    sees on entry: command 1's 2nd step sees its OWN 1st step's mark, while
    command 2 sees nothing. (This FAILS if execute_interleaved shares one state.)"""
    from tasks.GPSR.dispatch import execute_interleaved
    from tasks.GPSR.schedule import interleave

    seen = []

    def _probe(ctx, step, world, state):
        seen.append(state.get("mark"))
        state["mark"] = step.args.get("info")
        return True

    monkeypatch.setitem(skills.SKILLS, "say", _probe)  # 'say' has no location -> all "here"
    ctx = _FakeCtx()
    p1 = Plan(steps=[ground_step(RawStep(primitive="say", info="A1", raw="x"), world),
                     ground_step(RawStep(primitive="say", info="A2", raw="x"), world)], source="c1")
    p2 = Plan(steps=[ground_step(RawStep(primitive="say", info="B1", raw="x"), world)], source="c2")
    indexed = [(1, p1), (2, p2)]
    execute_interleaved(ctx, indexed, world, None, manip_enabled=False, order=interleave(indexed, world))
    assert seen == [None, "A1", None]  # cmd1.step2 sees cmd1.step1; cmd2 sees its own (empty)
