"""Spoken lines + the agent mission prompt for the Finals task."""

from __future__ import annotations

# --- spoken lines (deterministic scaffold) ----------------------------------
ENTER_ANNOUNCE = "Hello, I am Walkie. I am ready to help tidy up the house."
ENTER_DOOR_PROMPT = "Please open the arena door so I can come in and begin."

WELCOME_ANNOUNCE = "I will go and welcome the guest at the door."
WELCOME_GREETING = "Hello and welcome! Please come in. How can I help you today?"
WELCOME_NO_DOOR = "I do not have the exit door surveyed, so I cannot welcome the guest yet."

LAUNDRY_ANNOUNCE = "I will move the laundry basket to the washing machine."
LAUNDRY_NO_ARM = "My arm is not calibrated, so I can only point out the laundry basket, not carry it."
LAUNDRY_NO_TARGET = "I do not have the washing machine surveyed, so I cannot place the basket there."

DISHWASHER_ANNOUNCE = "I will close the dishwasher."
DISHWASHER_NO_ARM = "My arm is not calibrated, so I cannot close the dishwasher yet."
DISHWASHER_NO_TARGET = "I do not have the dishwasher surveyed, so I cannot reach it."

FINISH_ANNOUNCE = "I have done my rounds of the house. Thank you."

# --- per-room agent mission (handed to the Walkie orchestrator) --------------
# The scaffold drives to each room; the agent then finds + solves ONE problem there
# using its tools. It must SPEAK each problem it finds (the rulebook scores "find and
# clearly state a problem") and must not repeat a problem category it already solved.
FINAL_PATROL_MISSION = """You are competing in the RoboCup@Home Finals: keep the \
house tidy and help people. You are now in the {room}. There is no operator to ask — \
act autonomously; if unsure, pick the most likely option and proceed.

Do the following, in order, then stop:
1. Check whether anyone here is raising a hand (delegate_to_vision \
"is anyone raising a hand?"). If someone is, go to them and call \
`handle_person_request()` to take and carry out their request.
2. Otherwise look for ONE problem to solve in this room:
   - trash / an object on the FLOOR → pick it up and put it in the trash bin;
   - an object that is NOT where it belongs → look up where it belongs \
(delegate_to_database "where does the <object> belong?") and take it there.
3. SPEAK a clear statement of the problem you found before you solve it \
(e.g. "There is a cup on the floor; I will throw it away."). Stating the problem \
out loud is required.

Already-solved problem categories this run: {solved}. Do NOT solve a problem of a \
category you have already solved — pick a different kind, or move on. Solve at most \
one problem here, then finish (call speak with a short summary and stop)."""
