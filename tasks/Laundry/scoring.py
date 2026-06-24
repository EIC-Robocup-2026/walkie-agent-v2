"""Laundry (rulebook) scoresheet as data — LAUNDRY_SHEET + LAUNDRY_CAPTURES.

Transcribed from the rulebook score sheet (Total 4415 = every positive line,
INCLUDING the Extra-Reward bonuses open-washer/remove/basket, EXCLUDING the Special
section: −500 not-attending, −100 alt-start, +441 outstanding).

Laundry is an almost-pure manipulation challenge: ``arm=True`` on everything except
**navigating to the laundry area (15)**. The non-arm ceiling is therefore just 15 —
which is exactly why the non-arm roadmap deprioritises Laundry: there is no
"communicate perception" budget here the way PickAndPlace has (no recognize/indicate
lines on the sheet), so an arm-gated run scores almost nothing. The honest move is to
gate the manipulation behind a LAUNDRY_ARM_CALIBRATED flag (parity with the others)
but expect ~15 pts until the arm + folding skill lands.
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

LAUNDRY_SHEET = ScoreSheet(
    challenge="Laundry",
    rulebook_total=4415,
    lines=[
        # --- non-arm (the entire non-arm budget) ---------------------------------
        ScoreLine("navigate_laundry_area", "Navigate to the laundry area", 15, 1),
        # --- arm: retrieve / fold / stack ----------------------------------------
        ScoreLine("pick_up_clothing", "Pick up a piece of clothing from the basket", 100, 1, arm=True),
        ScoreLine("fold_clothing", "Fold a piece of clothing", 800, 1, arm=True),
        ScoreLine("fold_additional", "Fold an additional piece of clothing", 400, 5, arm=True),
        ScoreLine("stack_folded", "Stack a folded piece of clothing neatly", 100, 6, arm=True),
        # --- arm: extra rewards ---------------------------------------------------
        ScoreLine("open_washer_door", "Open the washing machine door", 300, 1, arm=True, kind=LineKind.BONUS),
        ScoreLine("remove_from_washer", "Remove clothing from the washing machine", 300, 1,
                  arm=True, kind=LineKind.BONUS),
        ScoreLine("use_basket", "Use the basket for transportation", 300, 1, arm=True, kind=LineKind.BONUS),
        # --- penalties (excluded from rulebook_total) ----------------------------
        ScoreLine("pen_pick_multiple", "Picking up multiple pieces at once", -100, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_quality", "Folding quality penalty", -800, 1, arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_flatten", "Human assistance: flattening/arranging before folding", -200, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_during", "Human assistance during folding", -800, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_remove_floor", "Clothing touches the floor / lost in transport", -200, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_basket_dropped", "Laundry dropped / lost during transport", -200, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_add_quality", "Additional-fold quality penalty", -400, 5,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_add_flatten", "Human assistance: flatten before additional fold", -100, 5,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_fold_add_during", "Human assistance during additional folding", -400, 5,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_assist_env", "Human assistance: environment change (per item)", -40, 1,
                  arm=True, kind=LineKind.PENALTY),
    ],
)

LAUNDRY_CAPTURES = {
    "navigate_laundry_area": Capture(0.70, 0.90, 1.00),
}
