"""Spoken language + LLM schemas for the GPSR task (rulebook 5.3).

GPSR is "understand and execute three arbitrary operator commands". The design
(docs/GPSR_DESIGN.md) parses each command into a typed `Plan` over a small set of
atomic primitives (the §3.1 vocabulary), speaks that plan to score "demonstrate a
plan has been generated", then executes it deterministically (Tier 1) with an
agent fallback (Tier 2). This module holds the LLM schemas/instructions for two
edges: splitting an utterance into commands, and parsing one command into a raw
plan that `parse.py` grounds against the world model.
"""

from __future__ import annotations

from typing import Literal, Optional

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


# --- Command -> typed plan (the parser) -------------------------------------

class RawStep(BaseModel):
    """One atomic action the LLM extracts from a command, BEFORE world grounding.

    Fill only the fields relevant to `primitive`; leave the rest null. Use the
    operator's own words for nouns — `parse.py` maps them onto canonical arena
    entities (so "the kitchen table"/"nightstand"/"a drink" are all fine here).
    """

    primitive: Literal[
        "navigate", "find_object", "find_person", "pick", "place", "deliver",
        "follow", "guide", "count", "get_person_info", "get_object_property",
        "say", "greet",
    ] = Field(description="The atomic action this step performs.")
    object: Optional[str] = Field(None, description="Object/category referenced (find_object, pick, place, deliver, count, get_object_property).")
    location: Optional[str] = Field(None, description="A specific placement/furniture/beacon (e.g. 'kitchen table', 'shelf').")
    room: Optional[str] = Field(None, description="A room (e.g. 'kitchen', 'living room').")
    to_location: Optional[str] = Field(None, description="Destination for guide/follow (a room or location).")
    from_location: Optional[str] = Field(None, description="Origin for guide (where the person currently is).")
    person: Optional[str] = Field(None, description="Person reference: a name, a gesture/pose, or a clothing/color description.")
    descriptor_kind: Optional[Literal["name", "gesture", "pose", "clothing"]] = Field(None, description="How `person` identifies the person.")
    recipient: Optional[str] = Field(None, description="Who receives a delivered object: 'me' for the operator, else a name or description.")
    which: Optional[str] = Field(None, description="For get_object_property: size|weight|category|color. For get_person_info: name|pose|gesture|clothing.")
    info: Optional[str] = Field(None, description="For say: the thing to tell (a fact, a joke, the time, an answer to a question).")
    raw: str = Field(description="The exact clause of the command this step came from.")


class RawPlan(BaseModel):
    """An ordered list of atomic steps that accomplish one command."""

    steps: list[RawStep] = Field(
        default_factory=list,
        description="The steps to perform, in order. Empty if the command is not understandable.",
    )


PARSE_INSTRUCTIONS = (
    "You are the planner for a domestic service robot in a RoboCup@Home GPSR test. "
    "Turn ONE operator command into an ordered list of atomic steps the robot can "
    "execute. The command came from a speech-to-text transcript and may have "
    "recognition errors or be phrased loosely — interpret intent generously.\n"
    "\n"
    "Decompose into these primitives only:\n"
    "- navigate: go to a room or location.\n"
    "- find_object: locate an object (optionally in a room/at a location).\n"
    "- find_person: locate a person by name, gesture/pose, or clothing.\n"
    "- pick: pick up an object (optionally from a location).\n"
    "- place: put an object on a location.\n"
    "- deliver: bring/give an object to someone (recipient 'me' = the operator).\n"
    "- follow: follow a person (optionally to a room).\n"
    "- guide: guide/escort/lead a person to a location.\n"
    "- count: count objects at a location, or persons (by gesture/pose) in a room.\n"
    "- get_person_info: determine a person's name, pose, gesture, or clothing.\n"
    "- get_object_property: determine an object's size, weight, category, or color.\n"
    "- say: tell/announce information (a fact, the time, a joke, an answer).\n"
    "- greet: greet a person by name or clothing in a room.\n"
    "\n"
    "Rules:\n"
    "- Make implicit navigation EXPLICIT: 'bring me the cola from the kitchen' = "
    "navigate(kitchen) -> find_object(cola) -> pick(cola) -> deliver(cola, me). A "
    "'tell/count' that names a place still needs a navigate step first.\n"
    "- Use the operator's words for nouns; do not normalize or invent entities.\n"
    "- Keep steps minimal and ordered; do not add steps the command didn't ask for "
    "(no returning to the operator — the task harness handles that).\n"
    "- Return an empty steps list ONLY if nothing is understandable."
)


# Spoken when the parse yields no usable plan even after grounding.
PLAN_NOT_UNDERSTOOD = (
    "I'm sorry, I did not understand that command well enough to make a plan."
)


# --- Spoken lines -----------------------------------------------------------
# The robot decides + advises how commands are issued (rulebook 5.3): it actively
# requests all three at once to keep the interleave bonus reachable and save the
# round-trips of returning between commands (§5.5).
GREET_OPERATOR = (
    "Hello, I am Walkie. Please give me all three of your commands now, one after "
    "another, and I will plan and carry them out."
)
ASK_FOR_COMMANDS = "What would you like me to do?"
ASK_REPEAT = "Sorry, I did not catch that. Could you please repeat the command?"
PLAN_PREAMBLE = "For command {n}, here is my plan."
CONFIRM_RECEIVED = "Understood. I will get to work."
COMMAND_ANNOUNCE = "Working on command {n}: {command}"
RETURN_ANNOUNCE = "I have finished. Returning to the instruction point."
ALL_DONE = "I have completed all the commands."
