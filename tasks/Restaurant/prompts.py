"""Spoken language + LLM schemas for the Restaurant task (rulebook 5.5).

PLACEHOLDER. The robot must take an order spoken by a customer (two objects),
politely confirm it, relay it to the barman, then serve. Order parsing is the
one piece of real NLP; gesture detection, navigation and manipulation are stubs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Order(BaseModel):
    """A customer's order — two edible/drinkable objects (rulebook 5.5 setup)."""

    items: list[str] = Field(
        default_factory=list,
        description=(
            "The food/drink items the customer asked for, normalized to their "
            "object names, in the order spoken. Usually two. Empty if none understood."
        ),
    )


EXTRACT_ORDER_INSTRUCTIONS = (
    "You are parsing a speech-to-text transcript of a restaurant customer giving "
    "an order to a service robot. Extract the items they asked for (normally two "
    "food or drink items), normalized to simple object names (e.g. 'a can of "
    "coke' -> 'coke'). Ignore pleasantries. Return an empty list if nothing "
    "orderable was said."
)


# --- Spoken lines -----------------------------------------------------------
# Readiness go-signal, spoken once the robot is in position at the bar (RESTAURANT_SIGNAL_READY).
READY_TO_START = "I am in position and ready to start serving."
# Spoken the moment a calling customer is detected and the robot starts heading over.
FOUND_CUSTOMER = "I see you waving! I'm coming over to take your order."
GREET_CUSTOMER = "Hello! I am Walkie. What would you like to order?"
ASK_REPEAT = "Sorry, I did not catch that. Could you repeat your order, please?"
# Used after the first re-ask: a poor accent garbles STT, so nudge the customer to slow
# down rather than repeating the identical line many times.
ASK_REPEAT_SLOW = "I am still having trouble hearing your order. Could you say it again, a little more slowly?"
CONFIRM_ORDER = "Let me confirm: you would like {items}. Is that right?"
ORDER_TAKEN = "Thank you, I will bring that right over."
RELAY_TO_BARMAN = "Order for a customer: {items}, please."
SERVE_ANNOUNCE = "Here is your order: {items}. Enjoy!"
ALL_DONE = "I have served all the customers I could reach."

GREET_BARMAN = "Hello! I have an order to place."
# Spoken to the barman the moment the order is actually in hand — the tray load is
# confirmed, or an item has been picked from the bar. Courtesy only; never scored,
# and only said on a CONFIRMED handoff (no thanks for items we never received).
# Gated by RESTAURANT_THANK_BARMAN.
THANK_BARMAN = "Thank you very much! I'll take this to the customer now."
SERVE_NO_CUSTOMER = "I could not find you again to serve the order, sorry."

# Caption prompt to remember a customer's look, so we can re-find them on return.
CUSTOMER_APPEARANCE_PROMPT = (
    "Describe this seated restaurant customer's visible appearance in one short "
    "sentence for someone who must recognize them again: clothing and its colors, "
    "hair, glasses, and any other distinctive feature."
)

# Live continuous scan (RESTAURANT_LIVE_SCAN): spoken once the running count of distinct
# waving customers ticks up, as the robot sweeps the room ({n} = callers seen so far).
SEE_N_CUSTOMERS = "I can see {n} customer{plural} waving."
# Spoken when the live approach loses the customer for good (no waver re-found) and the
# robot heads back to the bar instead of driving blind.
CUSTOMER_LOST = "I lost sight of the customer. I'll head back to the bar and try again."

# LLM system prompt: turn a one-line appearance caption (captured up close during the
# approach) into a single spoken call-out that names one distinctive visible feature and
# asks to take the order. Keep it short, warm, and natural; no markdown, one sentence.
GREETING_INSTRUCTIONS = (
    "You are a friendly restaurant service robot greeting a customer you have just "
    "walked up to. You are given a one-line description of the customer's visible "
    "appearance. Write ONE short, polite spoken greeting that references a single "
    "distinctive visible feature (e.g. the colour of their shirt) so they know you "
    "mean them, then ask what they would like to order. Output only the sentence — "
    "no quotes, no markdown, no preamble. Example: 'Hello! You in the red jacket — "
    "what would you like to order?'"
)

NO_CUSTOMER = "I do not see anyone calling right now."
# Spoken when the serve loop runs with the arm gated off (RESTAURANT_ARM_CALIBRATED unset).
PICK_NOT_AVAILABLE = "I cannot bring your order just yet — my arm is not ready."

# --- Tray mode (RESTAURANT_TRAY_MODE): the robot carries a tray; the barman loads
# the items onto it and the customer takes them off, so no arm grasp/place is used.
TRAY_ASK_BARMAN = "Could you please place {items} on my tray?"
TRAY_LOADED_CONFIRM = "Please tell me when the items are on my tray."
TRAY_PRESENT_CUSTOMER = "Here is your order: {items}. Please take them from my tray."
TRAY_TAKEN_CONFIRM = "Please tell me when you have taken your items."
# Re-asked when the human said "not yet" / stayed silent during a tray handoff.
TRAY_STILL_WAITING = "No problem, take your time. Just say 'ready' when it's done."
