"""All spoken language and LLM prompts for the HRI (receptionist) task.

Rulebook notes baked into the wording:
- Name + favorite drink are asked in ONE question; confirmation questions
  ("did you say James?") cost the non-essential-question bonus, so there are
  none. A targeted follow-up for a genuinely missing field is allowed.
- Missing info degrades to the GENERIC_* fallbacks rather than blocking.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GuestInfo(BaseModel):
    """Extraction schema for a guest's self-introduction."""

    name: str | None = Field(None, description="The guest's first name, or null if not stated")
    drink: str | None = Field(None, description="The guest's favorite drink, or null if not stated")


EXTRACT_GUEST_INFO_INSTRUCTIONS = (
    "You are parsing a speech-to-text transcript of a party guest introducing "
    "themselves to a receptionist robot that asked for their name and favorite "
    "drink. The transcript may contain recognition errors, fillers, or partial "
    "answers. Extract the guest's name and favorite drink. Use null for "
    "anything not actually stated. Normalize obvious STT mistakes only when "
    "the intended word is clear (e.g. 'cocacola' -> 'Coca-Cola')."
)


class SeatChoice(BaseModel):
    """LLM decision for which scanned seat to offer a guest."""

    seat_index: int | None = Field(
        None,
        description=(
            "Zero-based index of the chosen seat from the numbered list, "
            "or null if no detected seat is suitable to offer"
        ),
    )
    seat_part: Literal["LEFT", "MIDDLE", "RIGHT"] | None = Field(
        None,
        description=(
            "Only when the chosen seat is a sofa with cushions: which FREE "
            "cushion to offer (LEFT, MIDDLE, or RIGHT, as listed for that "
            "seat). Null for an ordinary single seat, or to let the robot take "
            "the first free cushion."
        ),
    )
    announcement: str | None = Field(
        None,
        description=(
            "The exact sentence the robot should say to direct the guest to "
            "the chosen seat — it is spoken verbatim. Null to use a default "
            "line."
        ),
    )
    reason: str | None = Field(
        None, description="One short sentence explaining the choice"
    )


PICK_SEAT_INSTRUCTIONS = (
    "You are the seating planner for a receptionist robot at a party. The "
    "robot stands in the living room facing the seating area and must point a "
    "newly arrived guest to one specific seat. You are given a text "
    "description of the robot's current camera frame: every detected seat "
    "(numbered, with class, position in the frame, size, detection "
    "confidence, and occupancy) and every detected person. The party host is "
    "always in the room and already seated, so at least one seat is taken by "
    "the host — even if occupancy detection missed them.\n"
    "Choosing the seat:\n"
    "- A seat is occupied only when someone is actually sitting on it. Trust "
    "the per-seat status given; a person merely standing in front of, or whose "
    "box edges onto, a seat does NOT make it occupied. Never pick a seat or "
    "cushion marked taken.\n"
    "- A sofa seats several people: it lists its cushions (LEFT/MIDDLE/RIGHT) "
    "with each one's status. Offer a sofa whenever it has a FREE cushion, even "
    "if someone is already on another cushion, and set seat_part to that free "
    "cushion's label. Pick the free cushion next to a seated host/guest so they "
    "can talk. Never offer a cushion marked taken.\n"
    "- Prefer confidently detected, larger seats over marginal detections.\n"
    "- Prefer a free seat near the host (and near an earlier-seated guest) so "
    "people can talk face to face — but never their own seats.\n"
    "Composing the announcement:\n"
    "- One short, warm spoken sentence telling the guest to take the chosen "
    "seat, addressing the guest by name when known.\n"
    "- For a sofa cushion, name the side naturally (e.g. 'the right side of the "
    "sofa, next to our host James').\n"
    "- The robot rotates to face the chosen seat BEFORE speaking, so never "
    "say 'to my left' or 'to my right' — describe the seat as right in front, "
    "and/or relative to the people already seated (e.g. 'the armchair next "
    "to our host James').\n"
    "- Referring to the host helps the guest find the seat; do so when it "
    "reads naturally. The host's favorite drink and appearance may be given "
    "and can be woven in when they help (e.g. 'next to our host James, in "
    "the red sweater').\n"
    "Return null for seat_index only when every detected seat is unusable."
)

GO_TO_GREETING_SPOT = "Please wait here while I greet the next guest at the door."

# --- Greeting ---------------------------------------------------------------

LOOKING_FOR_GUEST = "I am looking for the next guest."

GREET_ASK_BOTH = (
    "Hello, welcome to the party! I am Walkie, your receptionist. "
    "May I have your name and your favorite drink, please?"
)
ASK_MISSING_NAME = "And may I have your name, please?"
ASK_MISSING_DRINK = "And what is your favorite drink?"

APPEARANCE_CAPTION_PROMPT = (
    "Describe this person's visible appearance in one sentence for someone who "
    "must recognize them at a party: clothing and its colors, hair, glasses, "
    "and any other distinctive feature."
)

PHOTO_SAY_CHEESE = (
    "Let me take a picture so I can recognize you later. "
    "Please look at me and stand still — say cheese!"
)

YAY_I_REMEMBER = "I've successfully taken a picture of you"

HOST_APPEARANCE_CAPTION_PROMPT = (
    "Describe the visible appearance of the SEATED person in this image in one "
    "sentence for someone who must recognize them at a party: clothing and its "
    "colors, hair, glasses, and any other distinctive feature."
)

# --- Guiding ----------------------------------------------------------------

FOLLOW_ME = "Please follow me to the living room."
OFFER_SEAT_TEMPLATE = "Please take a seat on the {seat_class} {direction}."
OFFER_SEAT_FACING = "right here in front of me"  # used after rotating to face the seat
OFFER_SEAT_FALLBACK = "Please take any free seat you like."

# --- Introductions ------------------------------------------------------------

GENERIC_DRINK = "an unknown drink"
GENERIC_OTHER_GUEST = "our other guest"


class GuestIntroSpeeches(BaseModel):
    """The two guest-to-guest introductions, generated in a single LLM call.

    Each field is spoken while the robot FACES that guest (the listener),
    presenting the OTHER guest sitting beside them.
    """

    facing_guest_1: str = Field(
        default="",
        description="Spoken while facing guest 1, telling them who the other guest beside them is",
    )
    facing_guest_2: str = Field(
        default="",
        description="Spoken while facing guest 2, telling them who the other guest beside them is",
    )


GUEST_INTRO_INSTRUCTIONS = (
    "You are the receptionist robot at a party, introducing the two guests to "
    "each other. For each line below you are told which guest the robot is "
    "FACING (the listener), the name and favorite drink of the OTHER guest "
    "sitting beside them, and which side (left or right) that other guest is on "
    "FROM THE LISTENER'S OWN POINT OF VIEW. Write the exact words the robot "
    "should speak to the listener, presenting the other guest.\n"
    "Rules:\n"
    "- Address the listener and tell them who is beside them, e.g. 'Alice, the "
    "person on your left is Bob, and his favorite drink is cola.'\n"
    "- Use the EXACT side word given (left or right); never flip it. If the "
    "side is unknown, say 'next to you' instead of naming a side.\n"
    "- ALWAYS state the other guest's name and favorite drink explicitly — both "
    "are scored. For an unknown name say something natural like 'our other "
    "guest'; for an unknown drink omit the drink sentence rather than inventing "
    "one.\n"
    "- Warm, natural spoken sentences only. No stage directions, no emoji, "
    "nothing that cannot be said aloud.\n"
    "- Never invent facts; use only what is given.\n"
    "- Vary the phrasing between the two lines so it sounds human."
)

# --- Bag handover (gated by HRI_ENABLE_BAG) ----------------------------------

BAG_ASK_HANDOVER = (
    "I see you brought a bag. I will open my gripper now — please hang the bag "
    "on it."
)
BAG_CLOSING_WARNING = "I will close my gripper in 1 second, please be careful."
BAG_RECEIVED = "Thank you, I have the bag."
FOLLOW_HOST_NOT_AVAILABLE = (
    "I am sorry, I cannot follow the host yet. I will leave the bag here."
)

# --- Follow host & place the bag ---------------------------------------------

# Asked once the robot is holding the bag: the host answers by walking off
# ("follow me") or, if the spot is right here, by telling it to put it down.
BAG_ASK_WHERE = "Where would you like me to put the bag? You can say follow me, and I will come with you."
# Acknowledgement before following — also coaches the host to walk slowly,
# since each follow step costs a detect + recognize + listen round-trip.
FOLLOW_HOST_ACK = (
    "Okay, please lead the way and walk slowly. Tell me to put the bag down when we get there."
)
# Spoken when the host can't be found in the frame for a while.
FOLLOW_HOST_LOST = "I have lost sight of you. Please stand in front of me so I can follow."
# Acknowledgement right before lowering the bag.
BAG_PLACE_ACK = "Okay, I will put the bag down right here."

FINISH_TASK = "I finished the task. Please enjoy the party!"


class HostCommand(BaseModel):
    """LLM classification of one heard utterance during the bag handover."""

    intent: Literal["follow", "place", "other"] = Field(
        description=(
            "What the host is telling the robot to do. 'follow' = come with "
            "me / this way / over here / keep following. 'place' = put the bag "
            "down here / this is the spot / stop / leave it here. 'other' = "
            "anything that is not a clear instruction to the robot, including "
            "background party chatter and empty or garbled transcripts."
        )
    )


CLASSIFY_HOST_COMMAND_INSTRUCTIONS = (
    "A receptionist robot is holding a guest's bag, and the party host is "
    "leading it to where the bag should be left. The room is full of people "
    "talking, so the robot's microphone often picks up background chatter that "
    "is NOT addressed to it. You are given a single speech-to-text transcript "
    "(it may contain recognition errors). Decide what, if anything, the host "
    "is instructing the robot to do:\n"
    "- 'follow': follow me, come this way, over here, this way, keep coming.\n"
    "- 'place': put the bag (down) here, this is the spot, right here, leave "
    "it here, stop, that's far enough.\n"
    "- 'other': small talk, questions, unrelated chatter, anything not a clear "
    "command to the robot, or an empty/garbled transcript.\n"
    "When in doubt, answer 'other' so the robot never acts on crowd noise."
)
