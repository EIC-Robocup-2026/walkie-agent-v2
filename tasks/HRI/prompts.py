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
            "The exact words the robot should say to the guest — a warm "
            "suggestion of the chosen seat (they may sit anywhere free), "
            "spoken verbatim. Null to use a default line."
        ),
    )
    reason: str | None = Field(
        None, description="One short sentence explaining the choice"
    )


PICK_SEAT_INSTRUCTIONS = (
    "You are the seating planner for a receptionist robot at a party. The "
    "robot stands in the living room facing the seating area and suggests a "
    "seat to a newly arrived guest. You are given a text description of the "
    "robot's scan of the room — possibly taken from several camera headings "
    "(views) — listing every detected seat (numbered, with class, which view "
    "it was seen in, position in that view's frame, size, detection "
    "confidence, and occupancy) and every detected person.\n"
    "Reading the scene:\n"
    "- Everyone detected in the scan is already SEATED — the newly arrived "
    "guest is standing next to the robot, outside the seating area. So a "
    "person on or overlapping a seat means that seat (or cushion) is taken, "
    "and the party host is one of the seated people even if occupancy "
    "detection missed them.\n"
    "- A sofa seats several people: it lists its cushions (LEFT/MIDDLE/RIGHT) "
    "with each one's status. A sofa with a FREE cushion can still be offered "
    "even if someone is already on another cushion; set seat_part to the free "
    "cushion's label then.\n"
    "Choosing the seat:\n"
    "- The guest may sit anywhere that is free — every free seat or cushion "
    "is acceptable. Never pick a seat or cushion marked taken, or one someone "
    "is on.\n"
    "- It is nicer for people to sit next to one another, so when there is a "
    "choice, prefer a free seat or cushion beside the host or an earlier-"
    "seated guest — they can talk face to face. This is a preference, not a "
    "requirement.\n"
    "- Prefer confidently detected, larger seats over marginal detections.\n"
    "Composing the announcement:\n"
    "- One or two short, warm spoken sentences. Tell the guest they are "
    "welcome to sit anywhere that is free, and suggest the chosen seat rather "
    "than command it (e.g. 'Feel free to sit anywhere you like, Alice — the "
    "armchair right in front of me, next to our host James, would be a nice "
    "spot.'). Address the guest by name when known.\n"
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
ASK_MISSING_NAME = "Can you say your name again please?"
ASK_MISSING_DRINK = "Can you say your favorite drink again please?"

APPEARANCE_CAPTION_PROMPT = (
    "Describe one the person in this image in detail."
    "state: the type and color of their clothing (top and bottom), their hair "
    "color and style, whether they wear glasses, and any other distinctive "
    "feature such as a hat, beard, or bag they are carrying."
)

# The caption model on walkie-ai-server is a weak instruction-follower — it
# tends to narrate the whole scene ("The image shows a woman standing in front
# of a projector screen in a room with chairs...") no matter what the prompt
# says. So the raw caption is distilled by the LLM into a person-only
# description before it is stored/spoken (gated by HRI_APPEARANCE_DISTILL).


class PersonAppearance(BaseModel):
    """Extraction schema distilling a raw VLM caption to person-only details."""

    description: str | None = Field(
        None,
        description=(
            "One flowing sentence describing ONLY the person's own appearance: "
            "clothing type and colors, hair color/style, glasses, and other "
            "distinctive personal features. Null when the caption contains no "
            "usable detail about the person."
        ),
    )


APPEARANCE_DISTILL_INSTRUCTIONS = (
    "You are given a raw automatic caption of a camera image whose subject is "
    "ONE person, but the caption may ramble about the whole scene. Extract a "
    "single-sentence description of THAT PERSON ONLY — the person the caption "
    "focuses on (usually the first one mentioned, e.g. 'a woman standing...'). "
    "Keep every detail about their own appearance: clothing type and colors, "
    "hair color and style, glasses, hat, beard, a bag they are HOLDING or "
    "WEARING, and similar personal features. Drop everything else: the room, "
    "background, walls, doors, windows, furniture, screens, tables, objects "
    "lying around, and any OTHER people. Never invent details that are not in "
    "the caption. Return null if the caption says nothing usable about the "
    "person's appearance."
)

PHOTO_SAY_CHEESE = (
    "Let me take a picture so I can recognize you later. "
    "Please look at me and stand still — say cheese!"
)

YAY_I_REMEMBER = "I've successfully taken a picture of you"

HOST_APPEARANCE_CAPTION_PROMPT = (
    "Describe ONLY the SEATED person in this image, so someone could pick them "
    "out of a crowd at a party. Do NOT describe the room, background, "
    "furniture, screens, or any other people. In one sentence, state: the type "
    "and color of their clothing (top and bottom), their hair color and style, "
    "whether they wear glasses, and any other distinctive feature such as a "
    "hat or beard."
)

# --- Guiding ----------------------------------------------------------------

FOLLOW_ME = "Please follow me to the living room."
OFFER_SEAT_TEMPLATE = (
    "You can sit anywhere you like — how about the {seat_class} {direction}?"
)
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
    "FACING (the listener), the name, favorite drink, and appearance of the "
    "OTHER guest sitting beside them, and which side (left or right) that other "
    "guest is on FROM THE LISTENER'S OWN POINT OF VIEW. Write the exact words "
    "the robot should speak to the listener, presenting the other guest.\n"
    "Rules:\n"
    "- Address the listener and tell them who is beside them, e.g. 'Alice, the "
    "person on your left is Bob — he's the one in the blue checked shirt with "
    "glasses and short dark hair — and his favorite drink is cola.'\n"
    "- Use the EXACT side word given (left or right); never flip it. If the "
    "side is unknown, say 'next to you' instead of naming a side.\n"
    "- ALWAYS state the other guest's name and favorite drink explicitly — both "
    "are scored. For an unknown name say something natural like 'our other "
    "guest'; for an unknown drink omit the drink sentence rather than inventing "
    "one.\n"
    "- ALWAYS describe the other guest's appearance IN DETAIL using the given "
    "appearance description — clothing and its colors, hair, glasses, and any "
    "other distinctive feature mentioned. Keep every concrete detail from the "
    "description (reworded naturally into speech), so the listener can actually "
    "spot the person. If the appearance is unknown, skip it rather than "
    "inventing one.\n"
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
BAG_PLACE = "Please place the bag on my hand. After you placed the bag, gently push down on my hand."
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
