"""Restaurant (rulebook 5.5) scoresheet as data — RESTAURANT_SHEET + RESTAURANT_CAPTURES.

Transcribed from the rulebook 5.5 score sheet (Total 2360 = every positive Regular
+ Extra reward line, INCLUDING the First Pick / First Place bonuses, EXCLUDING the
Special section: −500 not-attending, −100 alt-start, +236 outstanding). ``arm=True``
marks the lines that need the manipulator (picking from the bar, serving, tray); the
non-arm budget — detect customer, reach table, take+confirm order, relay to barman,
return — is the 960-pt Phase-0 budget the serve pipeline scores with the arm gated.
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

RESTAURANT_SHEET = ScoreSheet(
    challenge="Restaurant",
    rulebook_total=2360,
    lines=[
        # --- non-arm: the Phase-0 serve budget (no manipulator) ------------------
        ScoreLine("detect_customer", "Detect a calling or waving customer", 80, 2),
        ScoreLine("reach_table", "Reach a customer's table", 80, 2),
        ScoreLine("understand_order", "Understand + confirm the order to the customer", 160, 2),
        ScoreLine("communicate_barman", "Communicate the order to the barman", 80, 2),
        ScoreLine("return_table", "Return to the customer table with the order", 80, 2),
        # --- arm: pick from the bar, serve, tray ---------------------------------
        ScoreLine("pickup_items", "Pick up the requested items from the Kitchen-bar", 100, 4, arm=True),
        ScoreLine("first_pick_bonus", "First Pick Bonus", 100, 1, arm=True, kind=LineKind.BONUS),
        ScoreLine("serve_order", "Serve the order to the customer", 100, 4, arm=True),
        ScoreLine("first_place_bonus", "First Place Bonus", 100, 1, arm=True, kind=LineKind.BONUS),
        ScoreLine("use_tray", "Use an unattached tray to transport", 200, 2, arm=True),
        # --- penalties (excluded from rulebook_total) ----------------------------
        ScoreLine("pen_guided_to_table", "Human assistance: being guided to a table", -80, 2,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_alternative_hri", "Alternative HRI", -80, 2, kind=LineKind.PENALTY),
        ScoreLine("pen_no_eye_contact", "Not making eye-contact when taking the order", -60, 2,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_handover_barman", "Human assistance: barman handover to the robot", -100, 4,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_guest_takes", "Human assistance: guest takes the object from tray/hand", -100, 4,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_not_reaching_bar", "Not reaching the bar (barman has to move)", -60, 2,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_directional", "Human assistance: asking for directional confirmation", -30, 2,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_told_location", "Human assistance: told/pointed where a table/bar is", -40, 2,
                  kind=LineKind.PENALTY),
    ],
)

# Capture % for the NON-ARM lines (Phase-0 serve pipeline; see restaurant memories).
RESTAURANT_CAPTURES = {
    "detect_customer": Capture(0.50, 0.75, 0.90),
    "reach_table": Capture(0.60, 0.85, 0.95),
    "understand_order": Capture(0.45, 0.70, 0.90),
    "communicate_barman": Capture(0.60, 0.85, 0.95),
    "return_table": Capture(0.60, 0.85, 0.95),
}
