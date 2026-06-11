"""All spoken language and LLM prompts for the HRI (receptionist) task.

Rulebook notes baked into the wording:
- Name + favorite drink are asked in ONE question; confirmation questions
  ("did you say James?") cost the non-essential-question bonus, so there are
  none. A targeted follow-up for a genuinely missing field is allowed.
- Missing info degrades to the GENERIC_* fallbacks rather than blocking.
"""

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
    "- Never pick an occupied seat. If a person's box overlaps a seat, treat "
    "it as occupied even when it is listed as free.\n"
    "- A sofa counts as a single seat; skip it if anyone is on it.\n"
    "- Prefer confidently detected, larger seats over marginal detections.\n"
    "- Prefer a free seat near the host (and near an earlier-seated guest) so "
    "people can talk face to face — but never their own seats.\n"
    "Composing the announcement:\n"
    "- One short, warm spoken sentence telling the guest to take the chosen "
    "seat, addressing the guest by name when known.\n"
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

# --- Greeting ---------------------------------------------------------------

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

GENERIC_GUEST = "our other guest"
GENERIC_DRINK = "an unknown drink"

INTRO_TEMPLATE = "{listener_name}, this is {other_name}. Their favorite drink is {other_drink}."
DESCRIBE_GUEST1_TEMPLATE = (
    "Inside, you will meet our first guest, {name}. {appearance}"
)

# --- Bag handover (gated by HRI_ENABLE_BAG) ----------------------------------

BAG_ASK_HANDOVER = (
    "I see you brought a bag. I will open my gripper now — please hang the bag "
    "on it."
)
BAG_CLOSING_WARNING = "I will close my gripper in three seconds, please be careful."
BAG_RECEIVED = "Thank you, I have the bag."
FOLLOW_HOST_NOT_AVAILABLE = (
    "I am sorry, I cannot follow the host yet. I will leave the bag here."
)
