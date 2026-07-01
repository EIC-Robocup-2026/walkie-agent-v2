"""Spoken lines + the custom-model-call prompts for the Inspection task.

Everything the robot SAYS during the inspection is centralized here so wording can
be tuned without touching control flow. The two ``*_INSTRUCTIONS`` / ``QNA_SYSTEM``
blocks drive the lightweight custom model calls (exit-signal classification and the
referee Q&A) that stand in for the full agent stack (see tasks/Inspection/skills.py).
"""

from __future__ import annotations

# --- entry door --------------------------------------------------------------
READY_ANNOUNCE = "Hello, I am Walkie. I am set up at the entry door and ready for inspection."
# request_open_door asks ONCE, then watches the depth camera and drives in on its own
# the moment the doorway reads clear — so this is a calm heads-up, not a plea.
ENTRY_DOOR_PROMPT = (
    "I am at the entry. Please open the door whenever you are ready, and I will "
    "notice it and come in."
)
ENTERED_ANNOUNCE = "The door is open. I am coming through now."

# --- safety stop -------------------------------------------------------------
SAFETY_ANNOUNCE = (
    "Before I move on, please note: if anyone steps in front of me, I will stop and "
    "wait until my path is clear."
)
SAFETY_PERSON_DETECTED = "I see you in front of me. I will wait here until you step aside."
SAFETY_OBSTACLE_DETECTED = "Something is close in front of me. I will wait until my path is clear."
SAFETY_CLEAR = "Thank you, my path is clear. I will continue."
SAFETY_NOONE = "I do not see anyone in my path. I will continue to the inspection points."

# --- inspection points -------------------------------------------------------
ARRIVED_AT_POINT = "I have reached inspection point {n}."

# --- external devices --------------------------------------------------------
DEVICES_INTRO = "Here are the external devices I am using for this test:"
DEVICES_NONE = "I am not using any external devices for this test."
DEVICES_OUTRO = "That is the complete list of external devices I am using."

# --- loudness ----------------------------------------------------------------
LOUDNESS_ASK_CONFIRM = "Could you hear me clearly just now? Please say yes or no."
LOUDNESS_REPEAT = "I will say it again, a little louder."
LOUDNESS_OK = "Great, thank you."

# --- exit --------------------------------------------------------------------
EXIT_WAIT_ANNOUNCE = (
    "I have finished the inspection items. Please press the green button whenever you want me to move to "
    "the exit."
)
EXIT_NO_SIGNAL = "I have not received a signal for a while."
EXIT_ACK = "Understood. I will move to the exit now."
EXIT_ARRIVED = "I have reached the exit. You may press the stop button. Thank you."

# --- custom model call: exit-signal classification ---------------------------
EXIT_CLASSIFY_INSTRUCTIONS = (
    "You are Walkie, a service robot being inspected. The referee just said something "
    "to you. Decide whether they are telling you that the inspection is finished and "
    "you should now move to the exit / leave. A question or an unrelated remark is NOT "
    "a signal to leave. Answer strictly with the requested JSON."
)

# --- custom model call: referee Q&A ------------------------------------------
QNA_SYSTEM = (
    "You are Walkie, an omnidirectional service robot from Chulalongkorn University's "
    "EIC team, currently being inspected by a RoboCup@Home referee. Answer the "
    "referee's question briefly and clearly, in one or two spoken sentences — no "
    "markdown, no lists, no headings. Be factual and concise.\n\n"
    "The external devices you are using for this test are: {devices}.\n"
    "If you do not know something, say so briefly rather than guessing."
)

# --- agent mode: per-turn message handed to the Walkie orchestrator -----------
# Only used when INSPECTION_AGENT_MODE=1: the scaffold still owns the listen /
# exit-signal loop and just routes each non-signal utterance to the full agent,
# which replies via its speak tool. The devices are threaded in so the agent can
# answer "what devices are you using?" without extra wiring.
AGENT_TURN = (
    "You are Walkie, being inspected by a RoboCup@Home referee. The referee just "
    'said: "{utterance}". Reply to them briefly using the speak tool (one or two '
    "sentences). Your external devices for this test are: {devices}."
)
