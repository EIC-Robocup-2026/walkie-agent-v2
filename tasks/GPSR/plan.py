"""GPSR typed plan: the artifact that scores the 300 and drives execution.

A command parses into a `Plan` — an ordered list of `PlanStep`s, each one an
atomic `Primitive` (the §3.1 vocabulary in docs/GPSR_DESIGN.md) with grounded
args. The plan is both:

- **demonstrable** — `render_plan_speech` turns it into a spoken sentence, which
  the robot says to score "demonstrate a plan has been generated" (rulebook 5.3);
- **executable** — Phase 1's executor dispatches each step's primitive to a skill.

This module is pure (no robot/LLM/network): building, grounding-status, and
wording a plan are all offline-testable, which is what makes the Phase-0 coverage
gate possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Primitive(str, Enum):
    """The atomic actions every generated command decomposes into.

    str-valued so the LLM parser can emit the value directly and it round-trips
    through structured output / JSON without conversion.
    """

    NAVIGATE = "navigate"
    FIND_OBJECT = "find_object"
    FIND_PERSON = "find_person"
    PICK = "pick"
    PLACE = "place"
    DELIVER = "deliver"
    FOLLOW = "follow"
    GUIDE = "guide"
    COUNT = "count"
    GET_PERSON_INFO = "get_person_info"
    GET_OBJECT_PROPERTY = "get_object_property"
    SAY = "say"
    GREET = "greet"


# Primitives that need manipulation (gated by GPSR_ENABLE_MANIPULATION until the
# arm is calibrated) — used by the executor/scheduler to flag arm-dependent work.
MANIPULATION_PRIMITIVES = frozenset({Primitive.PICK, Primitive.PLACE, Primitive.DELIVER})


class CmdStatus(Enum):
    """Lifecycle of one operator command through the executor."""

    RECEIVED = auto()
    PLANNED = auto()
    IN_PROGRESS = auto()
    DONE = auto()       # every step succeeded (Tier-1 or Tier-2)
    PARTIAL = auto()    # some steps succeeded (partial scoring applies)
    FAILED = auto()     # no step succeeded


@dataclass
class PlanStep:
    """One atomic action with grounded args and a record of what didn't ground.

    `args` holds canonical world values where grounding succeeded. `unresolved`
    lists ``(field, original_text)`` for each referenced noun that did NOT match
    the world model — the step's grounding gap, and the signal for whether this
    step needs the Tier-2 agent fallback. `raw` is the source clause, kept for the
    spoken plan and debugging.
    """

    primitive: Primitive
    args: dict = field(default_factory=dict)
    raw: str = ""
    unresolved: list[tuple[str, str]] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.unresolved


@dataclass
class Plan:
    """An ordered list of steps parsed from one operator command."""

    steps: list[PlanStep] = field(default_factory=list)
    source: str = ""  # the command utterance this came from

    def __bool__(self) -> bool:
        return bool(self.steps)

    @property
    def is_complete(self) -> bool:
        """True iff the command parsed into ≥1 fully-grounded typed steps.

        This is the per-command "covered with no Tier-2 fallback" predicate the
        Phase-0 coverage metric counts (docs/GPSR_DESIGN.md §10).
        """
        return bool(self.steps) and all(s.grounded for s in self.steps)

    @property
    def grounded_fraction(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.grounded for s in self.steps) / len(self.steps)

    @property
    def needs_manipulation(self) -> bool:
        return any(s.primitive in MANIPULATION_PRIMITIVES for s in self.steps)


# --- dispatch policy (pure, offline-testable) -------------------------------

def prefer_tier1(step: PlanStep, *, manip_enabled: bool) -> bool:
    """Whether to attempt this step with a deterministic skill (Tier 1) first.

    Eligible when the step is fully grounded and not a manipulation primitive
    that's currently gated off. Otherwise it goes straight to the Tier-2 agent
    fallback (which can also handle ungrounded / exotic clauses).
    """
    if not step.grounded:
        return False
    if step.primitive in MANIPULATION_PRIMITIVES and not manip_enabled:
        return False
    return True


def summarize_status(step_oks: list[bool]) -> CmdStatus:
    """Aggregate per-step success into a command status (partial scoring aware)."""
    if not step_oks:
        return CmdStatus.FAILED
    if all(step_oks):
        return CmdStatus.DONE
    if any(step_oks):
        return CmdStatus.PARTIAL
    return CmdStatus.FAILED


# --- deterministic plan -> speech (scores the 300) --------------------------

def _the(name: str | None) -> str:
    """Canonical world name -> spoken phrase: 'kitchen_table' -> 'the kitchen table'."""
    if not name:
        return "it"
    return "the " + name.replace("_", " ")


def _bare(name: str | None) -> str:
    return (name or "").replace("_", " ")


def _person_phrase(descriptor: str | None, kind: str | None) -> str:
    """Spoken reference to a person from a grounded descriptor + its kind.

    name -> "Charlie"; gesture/pose -> "the waving person"; clothing -> "the
    person in a red shirt". The plan stores the descriptor under args
    'descriptor' (find_person/follow/guide/greet) or 'recipient' (deliver).
    """
    desc = _bare(descriptor)
    if not desc:
        return "the person"
    if kind == "name":
        return desc
    if kind in ("gesture", "pose"):
        return f"the {desc} person"
    # clothing / free description: turn a bare attire phrase ("red shirt") into a
    # person reference, unless the descriptor already names a human ("the person
    # in a red shirt") — which would otherwise double up.
    if desc.lower().startswith(("person", "the person", "a person", "someone", "people")):
        return desc
    return f"the person in {desc}"


def _count_people_phrase(descriptor: str | None, kind: str | None) -> str:
    """Spoken noun for counting people: 'waving people' / 'people in red' / 'people'."""
    desc = _bare(descriptor)
    if kind in ("gesture", "pose") and desc:
        return f"{desc} people"
    if kind == "clothing" and desc:
        return f"people in {desc}"
    return "people"


def _step_phrase(step: PlanStep) -> str:
    """One imperative clause for a step, from canonical args (loose-key tolerant)."""
    a = step.args
    p = step.primitive
    if p is Primitive.NAVIGATE:
        return f"go to {_the(a.get('target'))}"
    if p is Primitive.FIND_OBJECT:
        where = a.get("location") or a.get("room")
        return f"find {_the(a.get('object'))}" + (f" at {_the(where)}" if where else "")
    if p is Primitive.FIND_PERSON:
        who = _person_phrase(a.get("descriptor"), a.get("kind"))
        room = a.get("room")
        return f"find {who}" + (f" in {_the(room)}" if room else "")
    if p is Primitive.PICK:
        where = a.get("location")
        return f"pick up {_the(a.get('object'))}" + (f" from {_the(where)}" if where else "")
    if p is Primitive.PLACE:
        return f"place {_the(a.get('object'))} on {_the(a.get('location'))}"
    if p is Primitive.DELIVER:
        recipient = a.get("recipient")
        if recipient in (None, "me", "you"):
            return f"bring {_the(a.get('object'))} to you"
        return f"deliver {_the(a.get('object'))} to {_person_phrase(recipient, a.get('kind'))}"
    if p is Primitive.FOLLOW:
        who = _person_phrase(a.get("descriptor"), a.get("kind"))
        to = a.get("to")
        return f"follow {who}" + (f" to {_the(to)}" if to else "")
    if p is Primitive.GUIDE:
        who = _person_phrase(a.get("descriptor"), a.get("kind"))
        return f"guide {who} to {_the(a.get('to'))}"
    if p is Primitive.COUNT:
        where = a.get("location") or a.get("room")
        if a.get("what") == "persons":
            what = _count_people_phrase(a.get("descriptor"), a.get("kind"))
        else:
            what = _bare(a.get("object") or a.get("category") or "objects")
        return f"count how many {what} there are" + (f" at {_the(where)}" if where else "")
    if p is Primitive.GET_PERSON_INFO:
        return f"find out the person's {_bare(a.get('which')) or 'details'}"
    if p is Primitive.GET_OBJECT_PROPERTY:
        obj = a.get("object")
        return f"check the {_bare(a.get('which')) or 'property'} of {_the(obj)}"
    if p is Primitive.SAY:
        return f"say {_bare(a.get('info')) or 'the information'}"
    if p is Primitive.GREET:
        who = _person_phrase(a.get("descriptor"), a.get("kind"))
        room = a.get("room")
        return f"greet {who}" + (f" in {_the(room)}" if room else "")
    return step.raw or p.value


def render_plan_speech(plan: Plan, *, preamble: str = "Here is my plan.") -> str:
    """A single spoken sentence describing the plan, in order.

    Deterministic (no LLM): "Here is my plan. First I will go to the kitchen,
    then find the cola, and finally bring it to you." Empty plan -> a brief
    can't-understand line so the caller always has something to speak.
    """
    if not plan.steps:
        return "I'm sorry, I could not work out a plan for that command."
    phrases = [_step_phrase(s) for s in plan.steps]
    if len(phrases) == 1:
        body = f"I will {phrases[0]}."
    else:
        seq = []
        for i, ph in enumerate(phrases):
            if i == 0:
                seq.append(f"first I will {ph}")
            elif i == len(phrases) - 1:
                seq.append(f"and finally {ph}")
            else:
                seq.append(f"then {ph}")
        body = ", ".join(seq) + "."
    return f"{preamble} {body[0].upper()}{body[1:]}"
