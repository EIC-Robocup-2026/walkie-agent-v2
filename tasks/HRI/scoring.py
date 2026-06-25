"""HRI / Receptionist (rulebook 5.x) scoresheet as data — HRI_SHEET + HRI_CAPTURES.

Transcribed from the rulebook score sheet (Total 1450 = every positive line,
EXCLUDING the Special section: −500 not-attending, +145 outstanding). ``arm=True``
marks the manipulator lines (open the entrance door, receive/drop the bag). The
non-arm budget is 950 — gaze, seating, guest recognition + introduction, and
following the host — i.e. almost the whole challenge; the dark part is that the
12-step flow is currently commented out (tasks/HRI/subtasks.py).

Note: ``follow_host`` (200) is a non-arm nav + re-ID skill, but earning it presumes
the bag was received (an arm step) — a *flow* dependency, not an arm requirement of
the follow itself; kept non-arm because that is what the HRI dev work targets.

The big NON-ARM penalties dwarf the positive deltas: not acknowledging people
(2×−200) and wrong guest info (4×−40) are why robust gaze + re-ID guard ~560 pts.
"""

from __future__ import annotations

from tasks.scoring import Capture, LineKind, ScoreLine, ScoreSheet

HRI_SHEET = ScoreSheet(
    challenge="HRI",
    rulebook_total=1450,
    lines=[
        # --- non-arm: greeting / gaze / seating ----------------------------------
        ScoreLine("doorbell_detect", "Detect the doorbell sound", 30, 2),
        ScoreLine("gaze_greeting", "Look at the person talking when receiving a guest", 50, 2),
        ScoreLine("seat_offer", "Offer a free seat to the new guest", 100, 2),
        ScoreLine("gaze_navigation", "Look in the navigation direction / at the goal", 15, 2),
        ScoreLine("visual_attribute_correct", "Tell a correct visual attribute of guest 1 to guest 2", 20, 4),
        ScoreLine("no_confirmation_questions", "Not asking non-essential confirm questions", 15, 4),
        # --- non-arm: introduction -----------------------------------------------
        ScoreLine("intro_name_drink", "Say name + favourite drink of each guest", 30, 4),
        ScoreLine("intro_gaze_correct", "Look at the correct guest while introducing", 50, 2),
        # --- non-arm: guiding / following ----------------------------------------
        ScoreLine("follow_host", "Follow the host to the bag-drop area", 200, 1),
        # --- arm ------------------------------------------------------------------
        ScoreLine("door_open", "Open the entrance door for a guest", 200, 2, arm=True),
        ScoreLine("bag_handover", "Receive the bag from the guest via handover", 50, 1, arm=True),
        ScoreLine("drop_correct_area", "Drop the bag in the correct area", 50, 1, arm=True),
        # --- penalties (excluded from rulebook_total) ----------------------------
        ScoreLine("pen_visual_incorrect", "Tell an incorrect visual attribute", -20, 4, kind=LineKind.PENALTY),
        ScoreLine("pen_handover_assist", "Requesting handover assistance from the guest", -25, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_bag_on_structure", "Bag placed on the robot structure, not handed over", -50, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_drop_bag", "Drop the bag while following the host", -50, 1,
                  arm=True, kind=LineKind.PENALTY),
        ScoreLine("pen_rediscover", "Rediscovering the operator by natural interaction", -50, 1,
                  kind=LineKind.PENALTY),
        ScoreLine("pen_ask_wait", "Asking the operator to wait", -50, 1, kind=LineKind.PENALTY),
        ScoreLine("pen_physical_guide", "Guiding the robot with physical contact", -150, 1, kind=LineKind.PENALTY),
        ScoreLine("pen_wrong_guest_info", "Wrong guest information was memorized", -40, 4, kind=LineKind.PENALTY),
        ScoreLine("pen_alternative_hri", "Alternative HRI", -20, 6, kind=LineKind.PENALTY),
        ScoreLine("pen_not_acknowledging", "Not acknowledging people", -200, 2, kind=LineKind.PENALTY),
    ],
)

# Capture % for the NON-ARM lines. doorbell is ~0 today (the flow punts it).
HRI_CAPTURES = {
    "doorbell_detect": Capture(0.0, 0.0, 0.5),
    "gaze_greeting": Capture(0.50, 0.80, 0.95),
    "seat_offer": Capture(0.40, 0.70, 0.90),
    "gaze_navigation": Capture(0.50, 0.80, 0.95),
    "visual_attribute_correct": Capture(0.30, 0.60, 0.85),
    "no_confirmation_questions": Capture(0.50, 0.80, 0.95),
    "intro_name_drink": Capture(0.40, 0.70, 0.90),
    "intro_gaze_correct": Capture(0.40, 0.70, 0.90),
    "follow_host": Capture(0.30, 0.60, 0.85),
}
