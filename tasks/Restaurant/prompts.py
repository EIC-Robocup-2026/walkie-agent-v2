"""Spoken language + LLM schemas for the Restaurant task (rulebook 5.5).

The robot detects a waving customer, navigates to their table, takes an order
spoken by the customer (two objects), politely confirms it, relays it to the
barman, then collects + serves the items one at a time. Order parsing is the NLP
piece (the `Order` schema below); the grasp planner that backs collect/serve is a
stub (see tasks/manipulation.py) but the arm motion is real.
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
COLLECTING_ITEM = "Let me get the {item} for you."
SERVE_ITEM = "Here is your {item}."
SERVE_ANNOUNCE = "Here is your order: {items}. Enjoy!"
ALL_DONE = "I have served all the customers I could reach."

# Identify fallback: clearly point out a detected customer the robot could not reach.
IDENTIFY_CUSTOMER = "I can see you are {desc}. I am coming to take your order."
IDENTIFY_CAPTION_PROMPT = (
    "Briefly describe this person's appearance (clothing colour and any standout "
    "feature) in a short phrase, e.g. 'the person in the red shirt'."
)

# Placeholder lines for the not-yet-served / not-found steps.
NO_CUSTOMER = "I do not see anyone calling right now."
ITEM_NOT_FOUND = "I could not find the {item} on the bar, so I will skip it."
