"""RoboCup@Home 2026 Finals (rulebook chapter 6) scoresheet as data — FINAL_SHEET.

The Finals has **no maximum score** (it can be repeated unlimited times); the
``rulebook_total`` 8450 is the scoresheet's own example tally — the multipliers
printed on the sheet (find+state a problem 3×150, solve a problem 10×650) plus the
three fixed lines (laundry basket 600, open door 600, dishwasher 300):

    3×150 + 10×650 + 600 + 600 + 300 = 8450

so ``positive_total()`` reproduces it. ``solve_problem`` is the open-ended problem
budget (trash → bin, misplaced object → default location, a person's request); like
GPSR's ``solve_command`` its draw is mixed-arm, so it is left ``arm=False`` and the
per-problem reality lives in the capture %. The named fixed lines are scored by the
deterministic handlers (tasks/Final/skills.py); the open-ended lines are claimed by
the agent-driven patrol and are optimistic (see tasks/scoring.py module docstring).
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

FINAL_SHEET = ScoreSheet(
    challenge="Final",
    rulebook_total=8450,
    lines=[
        # --- main goal (repeatable; multipliers are the sheet's example tally) ----
        ScoreLine("find_state_problem", "Find and clearly state a problem", 150, 3),
        ScoreLine("solve_problem", "Solve a problem (partial scoring)", 650, 10,
                  note="open-ended: trash / misplaced object / person request; mixed-arm"),
        # --- fixed, position-known problems --------------------------------------
        ScoreLine("move_laundry_basket", "Move the laundry basket to the washing machine",
                  600, 1, arm=True),
        ScoreLine("open_apartment_door", "Open the apartment door (welcome the guest)", 600, 1),
        ScoreLine("close_dishwasher", "Close the dishwasher door", 300, 1, arm=True),
        # --- penalties (excluded from rulebook_total) ----------------------------
        ScoreLine("pen_find_repeat", "Find a repeated problem category", -100, 10,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_human_assist", "Human assistance: asking for a problem's location",
                  -150, 10, kind=LineKind.PENALTY),
        ScoreLine("pen_solve_repeat_2nd", "Solve a repeated problem category (2nd time)",
                  -300, 10, kind=LineKind.PENALTY),
        ScoreLine("pen_solve_repeat_3rd", "Solve a repeated problem category (3rd+ time)",
                  -500, 10, kind=LineKind.PENALTY),
        ScoreLine("pen_restart", "Restart (only if scoring continues after)", -50, 10,
                  kind=LineKind.PENALTY),
    ],
)

# Rough capture %: the fixed lines are deterministic (high once surveyed + arm
# calibrated); the open-ended lines depend heavily on the arena draw and autonomy.
FINAL_CAPTURES = {
    "find_state_problem": Capture(0.40, 0.70, 1.00),
    "solve_problem": Capture(0.15, 0.40, 0.75),
    "move_laundry_basket": Capture(0.0, 0.50, 1.00),
    "open_apartment_door": Capture(0.30, 0.80, 1.00),
    "close_dishwasher": Capture(0.0, 0.50, 1.00),
}
