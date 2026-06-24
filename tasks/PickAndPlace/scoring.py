"""Pick and Place (rulebook 5.2) scoresheet as data — PNP_SHEET + PNP_CAPTURES.

The single source of truth for the PickAndPlace numbers: the runtime ScoreTracker
awards against these keys, and the planning estimate (docs/SCORING.md) is derived
from them. PURE DATA — no runtime/hardware coupling, so it imports anywhere.

Encoded from the rulebook 5.2 scoresheet (Total 3515 = sum of positive lines,
excluding the special −500/−100/+351 lines). ``arm=True`` marks every line that
needs the manipulator (gated behind PNP_ARM_CALIBRATED today); the three non-arm
positives — navigate + recognize + shelf-indicate — are the 195-pt budget the
flow scores with the arm gated off (see rulebook remark 16).

Award keys (used by the deferred ctx.score() wiring on the PnP branch):
    navigate_table · recognize_object · shelf_indicate   (non-arm)
    pick_* / place_* / rack_pull_push / dishwasher_door / milk_open / pour (arm)
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

PNP_SHEET = ScoreSheet(
    challenge="PickAndPlace",
    rulebook_total=3515,
    lines=[
        # --- non-arm: the budget scorable with the arm gated (remark 16) ---------
        ScoreLine("navigate_table", "Navigate to the table", 15, 1),
        ScoreLine("recognize_object", "Correctly recognize an object", 10, 12),
        ScoreLine("shelf_indicate", "Perceive on a shelf + indicate placement", 30, 2),
        # --- arm: picking --------------------------------------------------------
        ScoreLine("pick_transport", "Pick up an object for transportation", 50, 12, arm=True),
        ScoreLine("first_pick_bonus", "First Pick Bonus", 100, 1, arm=True, kind=LineKind.BONUS),
        ScoreLine("pick_floor", "Pick an object from the floor", 30, 1, arm=True),
        ScoreLine("pick_cutlery", "Pick cutlery", 50, 2, arm=True),
        ScoreLine("pick_plate", "Pick the plate", 100, 1, arm=True),
        ScoreLine("pick_tab", "Pick the dishwasher tab", 100, 1, arm=True),
        # --- arm: placing --------------------------------------------------------
        ScoreLine("place_designated", "Place an object in its designated location", 40, 12, arm=True),
        ScoreLine("place_dishwasher", "Correctly placed in the dishwasher", 70, 3, arm=True),
        ScoreLine("place_cabinet_similar", "Placed next to similar objects in the cabinet", 20, 2, arm=True),
        ScoreLine("place_tab_slot", "Dishwasher tab in the slot inside the dishwasher", 160, 1, arm=True),
        # --- arm/appliance: extra rewards (autonomous, not ask-referee) ----------
        ScoreLine("rack_pull_push", "Pull or push the dishwasher rack", 100, 2, arm=True),
        ScoreLine("dishwasher_door", "Open/close the dishwasher door without assistance", 200, 2, arm=True),
        ScoreLine("milk_open", "Open the milk container without assistance", 400, 1, arm=True),
        ScoreLine("pour", "Pour cereal or milk into the bowl without assistance", 200, 2, arm=True),
        # --- penalties (not netted into rulebook_total; arm-coupled) -------------
        ScoreLine("pen_aux_pick", "Picked a common object from the auxiliary table", -20, 2,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_aux_place", "Placed a common object from the auxiliary table", -20, 2,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_breakfast_area", "Area around breakfast items not cleaned", -30, 4,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_thrown", "Object thrown or dropped while placing", -40, 12,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_dropped_floor", "Object dropped on the floor", -40, 12,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_breakfast_not_typical", "Breakfast not in a typical meal setting", -50, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_spill", "Spilling cereal/milk while pouring", -100, 2,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_assist_reposition", "Human assistance: object repositioned", -30, 12,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_assist_handover", "Human assistance: handover", -100, 24,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_assist_env", "Human assistance: environment change (per item)", -40, 1,
                  kind=LineKind.PENALTY),
    ],
)

# Per-line capture % for the NON-ARM lines (matches docs/SCORING.md PnP section).
# Arm lines have no capture entry -> they contribute 0 to the non-arm estimate.
PNP_CAPTURES = {
    "navigate_table": Capture(0.70, 0.90, 1.00),
    "recognize_object": Capture(0.40, 0.65, 0.85),
    "shelf_indicate": Capture(0.30, 0.55, 0.80),
}
