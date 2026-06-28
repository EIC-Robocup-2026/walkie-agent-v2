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
            "Each DISTINCT command (separate task/goal) the operator gave, "
            "self-contained, in the order spoken. A single command that chains "
            "several actions toward one goal stays as ONE entry — do not split it. "
            "At most three. Empty if none understood."
        ),
    )


SPLIT_COMMANDS_INSTRUCTIONS = (
    "You receive a speech-to-text transcript of an operator giving a service robot "
    "UP TO THREE commands. Return each DISTINCT command as one self-contained "
    "string, in the order spoken.\n"
    "\n"
    "CRITICAL — do NOT over-split. A single command usually chains SEVERAL actions "
    "toward ONE goal (go somewhere, find something, then carry/deliver it). Those "
    "chained actions are ONE command — keep them together. Start a new command only "
    "when the operator clearly moves on to a SEPARATE, independent task.\n"
    "\n"
    "Examples:\n"
    "- 'go to the kitchen, find a coke, and bring it to me' -> ONE command (a "
    "single goal: deliver a coke).\n"
    "- 'navigate to the bedroom and tell me how many people are there' -> ONE "
    "command.\n"
    "- 'bring me a coke from the kitchen. then guide Charlie to the bedroom. finally "
    "count the apples on the desk' -> THREE commands (three separate goals).\n"
    "\n"
    "Rewrite each command so it stands alone (resolve 'then'/'after that'/shared "
    "nouns). Never return more than three commands, and do not invent or drop any. "
    "If unsure whether something is one command or several, prefer FEWER — keeping "
    "chained actions together. Return an empty list only if nothing is understandable."
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
# Recovery escalation (rulebook 5.3). Each rephrasing request costs −30, so we
# re-ask only on an empty parse and only a few times; then we request a custom
# operator (−20/command, but recovers the command) rather than silently failing.
ASK_REPHRASE = "Sorry, I did not understand. Could you please say the command again, more simply?"
REQUEST_CUSTOM_OPERATOR = (
    "I am having trouble understanding the commands. Could a custom operator "
    "please come and give me the commands?"
)
GIVE_UP_ON_COMMANDS = (
    "I am sorry, I could not understand the commands. I will return to the "
    "instruction point."
)
PLAN_PREAMBLE = "For command {n}, here is my plan."
CONFIRM_RECEIVED = "Understood. I will get to work."
# Plan-confirmation gate (GPSR_CONFIRM_PLAN): after speaking a command's plan the
# robot asks a human to approve it before executing. Off by default (the rulebook
# run is autonomous); turn on for supervised practice/demos.
ASK_CONFIRM_PLAN = "Should I carry out this plan for command {n}? Please say yes or no."
PLAN_CONFIRMED = "Okay, I will carry it out."
PLAN_REJECTED = "Okay, I will skip this command."
COMMAND_SKIPPED = "Skipping command {n} — the plan was not approved."
COMMAND_ANNOUNCE = "Working on command {n}: {command}"
# Interleaved mode (GPSR_INTERLEAVE): spoken once before the merged execution to
# demonstrate the robot is interleaving the commands (the bonus condition).
INTERLEAVE_ANNOUNCE = (
    "I will carry out all the commands together, interleaving them to save time "
    "and avoid unnecessary trips."
)
RETURN_ANNOUNCE = "I have finished. Returning to the instruction point."
ALL_DONE = "I have completed all the commands."
