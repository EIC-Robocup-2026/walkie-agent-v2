"""GPSR (rulebook 5.3) scoresheet as data — GPSR_SHEET + GPSR_CAPTURES.

Total 1490 = the draw-independent budget (understand 3×80, speak-a-plan 3×100,
interleave bonus 200 = 740) + the solve budget (3 commands × 250 = 750).

Caveat unique to GPSR: the **solve budget is draw-dependent**. The operator draws
3 commands; ~⅓ of categories need the arm (take/place/bring/deliver), ~⅔ don't.
A single ScoreLine can't express that mix, so ``solve_command`` is left ``arm=False``
and ``non_arm_ceiling()`` reads as a *best-case all-non-arm draw* (1490), NOT the
expectation. The realistic per-category capture % lives in the authoritative GPSR
table in ``docs/SCORING.md``; GPSR_CAPTURES["solve_command"] is a blended estimate
across draws so :func:`tasks.scoring.estimate` returns a sensible whole-challenge
number (~1030 exp), consistent with that table's run scenarios.
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

GPSR_SHEET = ScoreSheet(
    challenge="GPSR",
    rulebook_total=1490,
    lines=[
        # --- draw-independent budget (all non-arm) -------------------------------
        ScoreLine("understand_stt", "Understand a command (STT)", 80, 3),
        ScoreLine("speak_plan", "Speak a plan (parse + TTS)", 100, 3),
        ScoreLine("interleave_bonus", "Interleave bonus (take all 3 at once)", 200, 1,
                  kind=LineKind.BONUS),
        # --- solve budget (draw-dependent; see docstring) ------------------------
        ScoreLine("solve_command", "Execute a drawn command (partial scoring)", 250, 3,
                  note="draw-dependent: ~1/3 of categories need the arm"),
        # --- penalties (excluded from rulebook_total) ----------------------------
        ScoreLine("pen_custom_operator", "Custom operator", -20, 3, kind=LineKind.PENALTY),
        ScoreLine("pen_rephrasing", "Requested rephrasing", -30, 6, kind=LineKind.PENALTY),
        ScoreLine("pen_bypass_stt", "Bypass STT (typed)", -50, 3, kind=LineKind.PENALTY),
    ],
)

# understand/speak/interleave from docs/SCORING.md; solve_command is a BLENDED
# estimate across the draw (not a single category) — the per-category table in
# SCORING.md is authoritative for a known draw.
GPSR_CAPTURES = {
    "understand_stt": Capture(0.83, 0.92, 1.00),
    "speak_plan": Capture(0.80, 0.90, 1.00),
    "interleave_bonus": Capture(0.0, 1.00, 1.00),
    "solve_command": Capture(0.20, 0.45, 0.80),
}
