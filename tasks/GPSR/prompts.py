"""Spoken language + LLM schemas for the GPSR task (rulebook 5.3).

PLACEHOLDER. GPSR is "understand and execute three arbitrary operator commands",
which is exactly what the existing Walkie agent stack does — so this task is a
thin shell around it (see subtasks.py), not a fixed sequence of hardcoded steps.
The only task-level NLP here is splitting the operator's utterance into the
individual commands; the actual planning/execution is the agent's job.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CommandList(BaseModel):
    """The operator's commands, split out of one (possibly run-on) utterance."""

    commands: list[str] = Field(
        default_factory=list,
        description=(
            "Each distinct command the operator gave, verbatim and self-contained, "
            "in the order spoken. Usually up to three. Empty if none understood."
        ),
    )


SPLIT_COMMANDS_INSTRUCTIONS = (
    "You are parsing a speech-to-text transcript of an operator giving a service "
    "robot up to three commands at once. Split it into the individual, "
    "self-contained commands in the order spoken, each rewritten so it stands "
    "alone (resolve 'then', 'after that', shared objects). Do not invent commands "
    "and do not merge two distinct ones. Return an empty list if nothing is "
    "understandable."
)


# --- Spoken lines -----------------------------------------------------------
GREET_OPERATOR = (
    "Hello, I am Walkie. I am ready for your commands. You can give me up to "
    "three at once, or one at a time."
)
ASK_FOR_COMMANDS = "What would you like me to do?"
ASK_REPEAT = "Sorry, I did not catch that. Could you please repeat the command?"
CONFIRM_RECEIVED = "Understood. I will get to work."
COMMAND_ANNOUNCE = "Working on command {n}: {command}"
RETURN_ANNOUNCE = "I have finished. Returning to the instruction point."
ALL_DONE = "I have completed all the commands."
