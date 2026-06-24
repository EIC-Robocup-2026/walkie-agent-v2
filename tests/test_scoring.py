"""Offline tests for the shared scoring framework (tasks/scoring.py) + PNP_SHEET.

Two halves: (1) the framework mechanics — award clamping, penalties, the non-arm
ceiling, the planning estimate, JSON snapshot; (2) a *real* reconciliation of the
PickAndPlace sheet against the rulebook (positives sum to 3515, non-arm ceiling is
195), which catches a mis-keyed line rather than being fudged to pass.
"""

from __future__ import annotations

import json

import pytest

from tasks.scoring import (
    Capture,
    LineKind,
    ScoreLine,
    ScoreSheet,
    ScoreTracker,
    estimate,
)
from tasks.PickAndPlace.scoring import PNP_CAPTURES, PNP_SHEET


def _toy_sheet():
    return ScoreSheet(
        challenge="Toy",
        rulebook_total=130,
        lines=[
            ScoreLine("nav", "Navigate", 15, 1),
            ScoreLine("recognize", "Recognize object", 10, 10),       # 100 non-arm
            ScoreLine("place", "Place object", 40, 3, arm=True),       # 120 arm
            ScoreLine("drop", "Dropped while placing", -40, 3, arm=True, kind=LineKind.PENALTY),
        ],
    )


# --- framework mechanics ----------------------------------------------------

def test_award_clamps_to_max_units():
    t = ScoreTracker(_toy_sheet())
    assert t.award("recognize", 4) == 40       # 4 x 10
    assert t.award("recognize", 100) == 60     # only 6 left to cap at 10
    assert t.units_of("recognize") == 10
    assert t.award("recognize", 5) == 0        # already capped
    assert t.earned() == 100


def test_award_unknown_key_raises():
    t = ScoreTracker(_toy_sheet())
    with pytest.raises(KeyError):
        t.award("does_not_exist")


def test_penalty_nets_into_earned_but_not_positive():
    t = ScoreTracker(_toy_sheet())
    t.award("nav")           # +15
    t.award("place", 2)      # +80
    t.penalize("drop", 1)    # -40
    assert t.earned_positive() == 95
    assert t.penalties() == -40
    assert t.earned() == 55


def test_non_arm_ceiling_excludes_arm_lines():
    sheet = _toy_sheet()
    assert sheet.non_arm_ceiling() == 115        # nav 15 + recognize 100
    assert sheet.positive_total() == 235         # + place 120 (penalties excluded)


def test_duplicate_keys_raise():
    with pytest.raises(ValueError):
        ScoreSheet("Dup", 0, [ScoreLine("k", "a", 1), ScoreLine("k", "b", 2)])


def test_estimate_excludes_arm_by_default():
    sheet = _toy_sheet()
    caps = {"nav": Capture(1.0, 1.0, 1.0), "recognize": Capture(0.5, 0.5, 0.5),
            "place": Capture(1.0, 1.0, 1.0)}
    non_arm = estimate(sheet, caps)
    assert non_arm["total"]["exp"] == pytest.approx(15 + 50)     # arm 'place' excluded
    with_arm = estimate(sheet, caps, include_arm=True)
    assert with_arm["total"]["exp"] == pytest.approx(15 + 50 + 120)


def test_snapshot_and_write(tmp_path):
    path = tmp_path / "scorecard.json"
    t = ScoreTracker(_toy_sheet(), path=str(path))
    t.award("nav")
    t.award("recognize", 3)
    t.write()
    data = json.loads(path.read_text())
    assert data["challenge"] == "Toy"
    assert data["claimed"] == 45
    assert "attempted" in data["disclaimer"].lower()      # honesty label present
    assert {r["key"] for r in data["breakdown"]} == {"nav", "recognize"}


# --- PNP_SHEET reconciliation (real, not fudged) ----------------------------

def test_pnp_positive_total_matches_rulebook():
    # Sum of every positive line at full units must equal the official 3515 —
    # a mis-keyed line (wrong points/max_units) trips this.
    assert PNP_SHEET.positive_total() == PNP_SHEET.rulebook_total == 3515


def test_pnp_non_arm_ceiling_is_195():
    # navigate 15 + recognize 12x10 + shelf-indicate 2x30.
    assert PNP_SHEET.non_arm_ceiling() == 195
    assert {ln.key for ln in PNP_SHEET.non_arm_lines()} == {
        "navigate_table", "recognize_object", "shelf_indicate"}


def test_pnp_non_arm_estimate_matches_scoring_doc():
    # 15x0.90 + 120x0.65 + 60x0.55 = 124.5 — the ~125 planning figure.
    est = estimate(PNP_SHEET, PNP_CAPTURES)
    assert est["total"]["exp"] == pytest.approx(124.5)
    assert est["total"]["low"] == pytest.approx(15 * 0.70 + 120 * 0.40 + 60 * 0.30)
    assert est["total"]["high"] == pytest.approx(15 * 1.0 + 120 * 0.85 + 60 * 0.80)


def test_pnp_penalties_are_negative_and_excluded_from_total():
    pens = PNP_SHEET.penalties()
    assert pens and all(ln.points < 0 for ln in pens)
    assert all(ln.key.startswith("pen_") for ln in pens)
    # penalties must not inflate the positive total
    assert PNP_SHEET.positive_total() == 3515


# --- every challenge sheet reconciles against its rulebook total ------------

from tasks.GPSR.scoring import GPSR_SHEET                       # noqa: E402
from tasks.HRI.scoring import HRI_SHEET                         # noqa: E402
from tasks.Laundry.scoring import LAUNDRY_SHEET                 # noqa: E402
from tasks.Restaurant.scoring import RESTAURANT_SHEET           # noqa: E402

ALL_SHEETS = [PNP_SHEET, RESTAURANT_SHEET, HRI_SHEET, LAUNDRY_SHEET, GPSR_SHEET]


@pytest.mark.parametrize("sheet", ALL_SHEETS, ids=lambda s: s.challenge)
def test_sheet_positive_total_reconciles(sheet):
    # Sum of every positive line at full units must equal the official total
    # transcribed from the PDF — a mis-keyed points/count value trips this.
    assert sheet.positive_total() == sheet.rulebook_total


@pytest.mark.parametrize("sheet", ALL_SHEETS, ids=lambda s: s.challenge)
def test_sheet_penalties_are_negative(sheet):
    assert all(ln.points < 0 for ln in sheet.penalties())


def test_non_arm_ceilings_match_transcription():
    # The non-arm budget each challenge can score with the arm gated (PDF-derived).
    assert RESTAURANT_SHEET.non_arm_ceiling() == 960   # detect+reach+order+relay+return
    assert HRI_SHEET.non_arm_ceiling() == 950          # gaze+seat+intro+follow (door/bag are arm)
    assert LAUNDRY_SHEET.non_arm_ceiling() == 15       # navigate only — Laundry is all-arm
    assert GPSR_SHEET.non_arm_ceiling() == 1490        # best-case all-non-arm draw (see module docstring)
    assert PNP_SHEET.non_arm_ceiling() == 195


# --- the TaskContext.score() hook -------------------------------------------

def test_ctx_score_awards_to_attached_tracker():
    # ctx.score() is the runtime integration point — tasks award against the
    # sheet through it. A bad key is swallowed (logged), never raised, so the
    # live tally can never break a run.
    from tasks.base import TaskContext

    tracker = ScoreTracker(PNP_SHEET)
    ctx = TaskContext(walkie=None, walkieAI=None, model=None, scorer=tracker)
    ctx.score("recognize_object", 3)
    assert tracker.units_of("recognize_object") == 3
    ctx.score("does_not_exist")            # bad key -> logged, must NOT raise
    assert tracker.earned() == 30          # only the valid award counted


def test_ctx_score_is_noop_without_a_scorer():
    from tasks.base import TaskContext

    ctx = TaskContext(walkie=None, walkieAI=None, model=None)  # scorer=None
    ctx.score("anything")                  # must be a silent no-op, not raise
