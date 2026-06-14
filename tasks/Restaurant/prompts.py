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
GREET_CUSTOMER = "Hello! I am Walkie. What would you like to order?"
ASK_REPEAT = "Sorry, I did not catch that. Could you repeat your order, please?"
CONFIRM_ORDER = "Let me confirm: you would like {items}. Is that right?"
ORDER_TAKEN = "Thank you, I will bring that right over."
RELAY_TO_BARMAN = "Order for a customer: {items}, please."
SERVE_ANNOUNCE = "Here is your order: {items}. Enjoy!"
ALL_DONE = "I have served all the customers I could reach."

# Placeholder lines for the not-yet-implemented perception/manipulation steps.
NO_CUSTOMER = "I do not see anyone calling right now."
PICK_NOT_AVAILABLE = "I cannot pick up the items yet — manipulation is not implemented."
