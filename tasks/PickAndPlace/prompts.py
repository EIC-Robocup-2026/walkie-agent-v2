"""Spoken language + LLM schemas for the Pick and Place task (rulebook 5.2).

PLACEHOLDER. The flow is laid out in subtasks.py; the perception/manipulation
bodies are stubs. Spoken lines are real (the robot must announce its perception
and any help requests per the rulebook), so they are filled in where the wording
is already determined by the rules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Destination sorting ----------------------------------------------------
class ObjectSort(BaseModel):
    """Where a single dining-table object belongs (rulebook 5.2 main goal 1)."""

    destination: str | None = Field(
        None,
        description=(
            "One of: 'dishwasher' (dirty tableware/cutlery), 'trash' (the "
            "designated trash category), or 'cabinet' (everything else, grouped "
            "with similar items). Null if undecidable."
        ),
    )
    cabinet_group: str | None = Field(
        None,
        description=(
            "When destination is 'cabinet', the category/shelf group to place "
            "it with (e.g. 'snacks', 'drinks'); null otherwise."
        ),
    )
    reason: str | None = Field(None, description="One short sentence explaining the choice")


SORT_OBJECT_INSTRUCTIONS = (
    "You are sorting objects a service robot cleared from a dining table. Each "
    "object goes to exactly one place: the DISHWASHER for dirty tableware and "
    "cutlery (mugs, cups, plates, forks, knives, spoons); the TRASH bin for the "
    "one object category designated as trash for this run; otherwise the CABINET, "
    "grouped with semantically similar items already on the shelves. Objects that "
    "match no shelf category go to an empty part of the shelf. Return null only "
    "when the object is genuinely unidentifiable."
)


# --- Spoken lines (the robot must voice perception + help requests) ----------
PERCEPTION_ANNOUNCE = "I can see {count} objects on the table. Let me sort them."
ASK_OPEN_DISHWASHER = (
    "I could not open the dishwasher door myself. Could you please open it for me?"
)
ASK_OPEN_MILK = "I cannot open the milk container. Could you please open it for me?"
BREAKFAST_DONE = "Breakfast is served: a bowl, spoon, cereal, and milk."
TASK_DONE = "I have finished tidying the kitchen and serving breakfast."

# Placeholder lines for not-yet-implemented manipulation extension points.
PICK_NOT_AVAILABLE = (
    "I cannot pick up the {obj} yet — manipulation is not implemented. "
    "Skipping it for now."
)
POUR_NOT_AVAILABLE = "I cannot pour yet — pouring is not implemented."
