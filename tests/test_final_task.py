"""Offline checks for the Finals task (tasks/Final): scoresheet + task assembly.

No robot / CUDA: the scoresheet is pure data and build_final_task touches no hardware
(the heavy skill/agent imports are lazy inside the step bodies)."""

from __future__ import annotations

from tasks.base import TaskContext
from tasks.scoring import estimate
from tasks.Final import build_final_task
from tasks.Final.scoring import FINAL_CAPTURES, FINAL_SHEET


def test_sheet_positive_total_matches_rulebook():
    # 3*150 + 10*650 + 600 + 600 + 300 = 8450 (the scoresheet's example tally).
    assert FINAL_SHEET.positive_total() == FINAL_SHEET.rulebook_total == 8450


def test_sheet_has_the_fixed_lines_and_penalties():
    keys = {ln.key for ln in FINAL_SHEET.lines}
    assert {"open_apartment_door", "move_laundry_basket", "close_dishwasher"} <= keys
    # The two manipulation problems are arm-gated; the door open is not.
    assert FINAL_SHEET.line("move_laundry_basket").arm is True
    assert FINAL_SHEET.line("close_dishwasher").arm is True
    assert FINAL_SHEET.line("open_apartment_door").arm is False
    # Repetition + assistance penalties are negative and excluded from the total.
    assert FINAL_SHEET.line("pen_solve_repeat_3rd").points == -500
    assert all(ln.points < 0 for ln in FINAL_SHEET.penalties())


def test_estimate_runs_over_captures():
    est = estimate(FINAL_SHEET, FINAL_CAPTURES, include_arm=True)
    assert est["challenge"] == "Final"
    assert est["total"]["exp"] > 0


def test_build_final_task_assembles_without_hardware():
    ctx = TaskContext(walkie=None, walkieAI=None, model=None)
    task = build_final_task(ctx)
    assert task.name == "Final"
    assert [s.name for s in task.subtasks] == [
        "EnterArena",
        "WelcomeGuest",
        "MoveLaundryBasket",
        "CloseDishwasher",
        "PatrolAndSolve",
        "Finish",
    ]
    # EnterArena is the only critical step (entering is required to begin).
    assert task.subtasks[0].critical is True
