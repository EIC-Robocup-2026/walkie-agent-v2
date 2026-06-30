"""Offline unit tests for the GPSR scoring-hardening helpers — all pure logic
(no robot/LLM/network), so they are tested here directly:

- ``world._fuzzy_match`` / ``WorldModel._lookup`` fuzzy fallback — recover an
  STT/LLM near-miss noun that would otherwise be an ungrounded gap (protects the
  per-command 250 + the plan 100).
- ``skills._median_count`` — object-count stabilization across flickery frames.
- ``skills._superlative_dir`` / ``_bbox_area`` / ``_pick_by_size`` — selecting the
  biggest/smallest object by image area for a "what is the biggest object" query.
- The deterministic plan render stays clean (no leaked raw token) for the kinds of
  commands the hardening touches (the scored 300).
"""

from __future__ import annotations

import pytest

from tasks.GPSR.parse import ground_step
from tasks.GPSR.plan import render_plan_speech, Plan
from tasks.GPSR.prompts import RawStep
from tasks.GPSR.skills import (
    _bbox_area,
    _count_objects_stable,
    _median_count,
    _pick_by_size,
    _superlative_dir,
)
from walkie_world.map.vocab import _fuzzy_match, load_world


@pytest.fixture(scope="module")
def world():
    return load_world(include_absent=True)  # full grammar, like the parser tests


# --- grounding fuzzy fallback ----------------------------------------------

def test_fuzzy_match_recovers_near_miss():
    cands = ["kitchen_table", "dining_table", "sofa", "cabinet"]
    assert _fuzzy_match("kitchen_tabel", cands, 0.8) == "kitchen_table"  # transposed
    assert _fuzzy_match("cabinett", cands, 0.8) == "cabinet"             # doubled char


def test_fuzzy_match_rejects_a_genuinely_different_word():
    cands = ["kitchen_table", "sofa", "cabinet"]
    assert _fuzzy_match("refrigerator", cands, 0.8) is None


def test_fuzzy_match_disabled_at_zero_cutoff():
    cands = ["kitchen_table"]
    assert _fuzzy_match("kitchen_tabel", cands, 0.0) is None  # 0 -> exact-only


def test_lookup_exact_still_wins_and_is_unchanged(world):
    # Fuzzy is consulted ONLY on a miss, so exact behaviour is untouched.
    assert world.location("the kitchen table") == "kitchen_table"
    assert world.room("kitchen") == "kitchen"


def test_lookup_grounds_a_misspelled_noun_via_fuzzy(world, monkeypatch):
    monkeypatch.setenv("GPSR_GROUNDING_FUZZY_CUTOFF", "0.8")
    assert world.location("kitchen tabel") == "kitchen_table"  # would miss exactly


def test_lookup_fuzzy_off_leaves_a_miss_unresolved(world, monkeypatch):
    monkeypatch.setenv("GPSR_GROUNDING_FUZZY_CUTOFF", "0")
    assert world.location("kitchen tabel") is None  # exact-only -> a real gap


# --- object-count stabilization (median over frames) -----------------------

def test_median_count_is_robust_to_one_flickery_frame():
    assert _median_count([3, 3, 4]) == 3   # a single +1 flicker is ignored
    assert _median_count([2, 3, 3]) == 3   # a single dropped box is ignored


def test_median_count_even_and_empty():
    assert _median_count([2, 4]) == 3      # even -> rounded mean of the middle two
    assert _median_count([]) == 0
    assert _median_count([5]) == 5


class _Snap:
    def __init__(self, img):
        self.img = img


class _CountCtx:
    """Snapshots a fixed sequence of detection-count frames (None = dropped frame)."""

    def __init__(self, per_frame_counts):
        self._counts = list(per_frame_counts)
        self._i = 0

    def snapshot(self):
        return _Snap(img="frame")  # never None here; dropped frames simulated below


def test_count_objects_stable_takes_the_median(monkeypatch):
    monkeypatch.setenv("GPSR_COUNT_OBJ_FRAMES", "3")
    monkeypatch.setenv("GPSR_COUNT_OBJ_SETTLE_SEC", "0")
    frames = iter([[object()] * 3, [object()] * 4, [object()] * 3])  # 3,4,3 -> 3
    monkeypatch.setattr("tasks.GPSR.skills._detect", lambda ctx, img, classes: next(frames))
    assert _count_objects_stable(_CountCtx([]), ["cup"]) == 3


def test_count_objects_stable_none_when_no_frame(monkeypatch):
    monkeypatch.setenv("GPSR_COUNT_OBJ_FRAMES", "3")
    monkeypatch.setenv("GPSR_COUNT_OBJ_SETTLE_SEC", "0")

    class _NoFrameCtx:
        def snapshot(self):
            return None

    assert _count_objects_stable(_NoFrameCtx(), ["cup"]) is None


# --- superlative object selection by size ----------------------------------

def test_superlative_dir_reads_direction_from_raw():
    assert _superlative_dir("the biggest object on the desk") == "large"
    assert _superlative_dir("what's the largest item") == "large"
    assert _superlative_dir("the smallest one") == "small"
    assert _superlative_dir("the thinnest object") == "small"
    assert _superlative_dir("the red cup") is None
    assert _superlative_dir(None) is None


class _Det:
    def __init__(self, bbox, confidence=0.5):
        self.bbox = bbox
        self.confidence = confidence


def test_bbox_area_and_pick_by_size():
    small = _Det((0, 0, 10, 10))    # area 100
    big = _Det((0, 0, 30, 20))      # area 600
    assert _bbox_area(big.bbox) == 600
    assert _pick_by_size([small, big], "large") is big
    assert _pick_by_size([small, big], "small") is small


def test_bbox_area_clamps_inverted_box():
    assert _bbox_area((30, 30, 10, 10)) == 0.0  # never negative


# --- render stays clean for the touched command shapes ---------------------

@pytest.mark.parametrize("steps", [
    [RawStep(primitive="count", object="cups", location="the desk", raw="count the cups on the desk")],
    [RawStep(primitive="get_object_property", object="object", which="size", raw="the biggest object"),
     RawStep(primitive="say", info="the biggest object on the desk", raw="tell me")],
])
def test_render_is_clean_for_count_and_superlative(world, steps):
    plan = Plan(steps=[ground_step(s, world) for s in steps])
    speech = render_plan_speech(plan)
    assert speech and "could not work out a plan" not in speech
    assert "_" not in speech  # no leaked raw primitive token
