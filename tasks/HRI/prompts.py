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

# --- Guiding ----------------------------------------------------------------

FOLLOW_ME = "Please follow me to the living room."
OFFER_SEAT_TEMPLATE = "Please take a seat on the {seat_class} {direction}."
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
