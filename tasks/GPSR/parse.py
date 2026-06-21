"""GPSR command parser: utterance -> typed, grounded `Plan`.

Two layers, split so the load-bearing logic is offline-testable:

- `ground_plan` / `ground_step` — **pure**: map a `RawPlan` (the LLM's loose
  extraction) onto canonical world entities, recording every noun that failed to
  ground. This is what the Phase-0 coverage gate measures (docs/GPSR_DESIGN.md
  §10) and what unit tests exercise without any LLM.
- `parse_command` / `parse_commands` — the LLM edge: `ctx.extract` a `RawPlan`
  from the utterance, then ground it. Needs the model (OpenRouter), not the robot.

The parser is the single most load-bearing component in GPSR: it gates the
draw-independent 540 (understand + speak-a-plan) and feeds every solve.
"""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from . import prompts
from .plan import Plan, PlanStep, Primitive
from .prompts import RawPlan, RawStep
from .world import WorldModel, load_world

if TYPE_CHECKING:  # type-only; importing these at runtime would pull no hardware,
    from pydantic import BaseModel  # but keep the pure core import-light regardless.

# Closed value sets for the property/info "which" fields (loose-normalized).
_OBJECT_PROPERTIES = {"size", "weight", "category", "color", "colour"}
_PERSON_INFO = {"name", "pose", "gesture", "clothing"}

# Generic object placeholders: a superlative/query reference ("the biggest object
# on the desk") names no concrete item — it's discovered at the placement. These
# ground to None *without* a gap, so a placement-scoped query still completes.
_GENERIC_OBJECTS = {"object", "item", "thing", "one", "something", "anything", "stuff"}

# Person nouns the LLM sometimes drops into `object` instead of `person` — most
# often on "count the people in X", where there is no gesture/name to trigger the
# person field. We detect them so the count grounds as persons (the pose detector),
# not as an object the detector can't resolve (which would forfeit the command).
_PERSON_NOUNS = {
    "person", "people", "persons", "someone", "somebody", "anyone", "anybody",
    "human", "humans", "guest", "guests", "man", "men", "woman", "women",
    "boy", "boys", "girl", "girls", "child", "children", "everyone", "everybody",
}


def _is_person_noun(text: str | None) -> bool:
    """True when the head noun names a person generically ("people", "guests")."""
    if not text:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and words[-1] in _PERSON_NOUNS


def _is_generic_object(text: str | None) -> bool:
    """True for a non-specific object reference ("the biggest object", "something").

    Tests the **head noun** (last word) so a leading superlative/qualifier
    ("biggest object", "heaviest item") doesn't hide the generic noun — the LLM
    emits the bare "object" for some phrasings and "biggest object" for question
    forms ("what's the biggest object on the desk").
    """
    if not text:
        return False
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    return words[-1].rstrip("s") in {g.rstrip("s") for g in _GENERIC_OBJECTS}


def _ground(unresolved: list[tuple[str, str]], field: str, text: str | None, resolver) -> str | None:
    """Resolve `text` via `resolver`; log (field, text) to `unresolved` on a miss.

    Empty/None text grounds to None silently (the field simply wasn't given).
    """
    if not text:
        return None
    val = resolver(text)
    if val is None:
        unresolved.append((field, text))
    return val


def _ground_target(unresolved, raw: RawStep, world: WorldModel) -> str | None:
    """A navigation target: a specific location wins, else a room."""
    text = raw.location or raw.room or raw.to_location
    if not text:
        return None
    val = world.location(text) or world.room(text)
    if val is None:
        unresolved.append(("target", text))
    return val


# The operator refers to themselves in a person slot ("follow me", "guide me to
# the exit"). It names no enrolled person, so don't try to ground it against the
# names list (that records a spurious gap and breaks the spoken plan — "follow me");
# it grounds to the operator, whom follow/guide track as the nearest person anyway.
_OPERATOR_REFS = {"me", "myself", "i", "operator", "you", "yourself"}


def _ground_person(unresolved, raw: RawStep, world: WorldModel) -> dict:
    """Ground a person reference. Returns args fragment {descriptor, kind}.

    name -> must match the names list; gesture/pose -> must match the gesture
    set; clothing (or an unclassified free description) is open-vocab and always
    grounds. The operator self-reference ("me") grounds to kind "operator". An
    unknown name/gesture is recorded unresolved.
    """
    text = raw.person
    kind = raw.descriptor_kind
    if not text:
        return {}
    if text.strip().lower() in _OPERATOR_REFS:
        return {"descriptor": "you", "kind": "operator"}
    if kind == "name":
        canon = _ground(unresolved, "person", text, world.name)
        return {"descriptor": canon or text, "kind": "name"}
    if kind in ("gesture", "pose"):
        canon = _ground(unresolved, "person", text, world.gesture)
        return {"descriptor": canon or text, "kind": kind}
    if kind == "clothing":
        return {"descriptor": text, "kind": "clothing"}
    # No kind given: try name, then gesture, else treat as an open description.
    canon = world.name(text)
    if canon:
        return {"descriptor": canon, "kind": "name"}
    canon = world.gesture(text)
    if canon:
        return {"descriptor": canon, "kind": "gesture"}
    return {"descriptor": text, "kind": "clothing"}


def _ground_person_where(unresolved, raw: RawStep, world: WorldModel) -> dict:
    """Ground WHERE a person is, for find_person / greet / count-persons.

    Accepts a room OR a beacon/location: ``meetPrsAtBeac`` ("meet Charlie at the
    {beacon}") and the like name a placement/beacon, not a room. A room stores
    under ``room``, a beacon under ``location`` — the skills read ``location or
    room`` (like count/find_object), and go_to_named navigates either. A named
    place that matches neither is a grounding gap. Empty -> {} (no place given).
    """
    text = raw.room
    if not text:
        return {}
    r = world.room(text)
    if r:
        return {"room": r}
    loc = world.location(text)
    if loc:
        return {"location": loc}
    unresolved.append(("room", text))
    return {}


def _ground_which(unresolved, raw: RawStep, allowed: set[str]) -> str | None:
    if not raw.which:
        unresolved.append(("which", ""))
        return None
    key = raw.which.strip().lower()
    if key in allowed:
        return "color" if key == "colour" else key
    unresolved.append(("which", raw.which))
    return None


def ground_step(raw: RawStep, world: WorldModel) -> PlanStep:
    """Map one RawStep onto canonical world args, recording grounding gaps. Pure."""
    try:
        primitive = Primitive(raw.primitive)
    except ValueError:
        return PlanStep(Primitive.SAY, {}, raw.raw, [("primitive", str(raw.primitive))])

    unresolved: list[tuple[str, str]] = []
    args: dict = {}

    def loc(field="location"):
        return _ground(unresolved, field, getattr(raw, field, None), world.location)

    def obj():
        if _is_generic_object(raw.object):
            return None  # placement-scoped query, not a concrete item — no gap
        return _ground(unresolved, "object", raw.object, world.obj)

    if primitive is Primitive.NAVIGATE:
        args["target"] = _ground_target(unresolved, raw, world)

    elif primitive is Primitive.FIND_OBJECT:
        args["object"] = obj()
        args["location"] = world.location(raw.location) if raw.location else None
        args["room"] = world.room(raw.room) if raw.room else None
        if raw.location and args["location"] is None and not (raw.room and args["room"]):
            unresolved.append(("location", raw.location))

    elif primitive is Primitive.FIND_PERSON:
        args.update(_ground_person(unresolved, raw, world))
        args.update(_ground_person_where(unresolved, raw, world))

    elif primitive is Primitive.PICK:
        args["object"] = obj()
        args["location"] = loc()

    elif primitive is Primitive.PLACE:
        args["object"] = obj()
        if not raw.location:
            unresolved.append(("location", ""))
        else:
            args["location"] = loc()

    elif primitive is Primitive.DELIVER:
        args["object"] = obj()
        recipient = raw.recipient
        if recipient and recipient.strip().lower() in ("me", "operator", "you"):
            args["recipient"] = "me"
        elif recipient:
            person = _ground_person(unresolved, RawStep(primitive="find_person", person=recipient, descriptor_kind=raw.descriptor_kind, raw=raw.raw), world)
            args["recipient"] = person.get("descriptor", recipient)
            args["kind"] = person.get("kind")
        args["room"] = world.room(raw.room) if raw.room else None

    elif primitive is Primitive.FOLLOW:
        args.update(_ground_person(unresolved, raw, world))
        args["to"] = world.room(raw.to_location) or world.location(raw.to_location) if raw.to_location else None

    elif primitive is Primitive.GUIDE:
        args.update(_ground_person(unresolved, raw, world))
        args["from"] = world.location(raw.from_location) or world.room(raw.from_location) if raw.from_location else None
        to_text = raw.to_location or raw.location
        if not to_text:
            unresolved.append(("to", ""))
        else:
            to = world.location(to_text) or world.room(to_text)
            if to is None:
                unresolved.append(("to", to_text))
            args["to"] = to

    elif primitive is Primitive.COUNT:
        if raw.person or raw.descriptor_kind or _is_person_noun(raw.object):  # counting people
            args["what"] = "persons"
            args.update(_ground_person(unresolved, raw, world))
            args.update(_ground_person_where(unresolved, raw, world))
        else:  # counting objects
            args["what"] = "objects"
            cat = world.category(raw.object) if raw.object else None
            if cat:
                args["category"] = cat
            else:
                args["object"] = obj()
            args["location"] = loc()
            args["room"] = world.room(raw.room) if raw.room else None

    elif primitive is Primitive.GET_PERSON_INFO:
        args["which"] = _ground_which(unresolved, raw, _PERSON_INFO)

    elif primitive is Primitive.GET_OBJECT_PROPERTY:
        args["object"] = obj()
        args["which"] = _ground_which(unresolved, raw, _OBJECT_PROPERTIES)

    elif primitive is Primitive.SAY:
        if not (raw.info or "").strip():
            unresolved.append(("info", ""))
        else:
            args["info"] = raw.info.strip()

    elif primitive is Primitive.GREET:
        args.update(_ground_person(unresolved, raw, world))
        args.update(_ground_person_where(unresolved, raw, world))

    # Drop None-valued args so the plan/render stay clean.
    args = {k: v for k, v in args.items() if v is not None}
    return PlanStep(primitive=primitive, args=args, raw=raw.raw, unresolved=unresolved)


def ground_plan(raw: RawPlan, world: WorldModel, *, source: str = "") -> Plan:
    """Ground every step of a RawPlan. Pure — the offline-testable core."""
    return Plan(steps=[ground_step(s, world) for s in raw.steps], source=source)


# --- LLM edge ---------------------------------------------------------------
#
# These take a `model` (a langchain ChatOpenAI), NOT a TaskContext, so the parser
# — and its Phase-0 coverage gate — runs with only OpenRouter and no robot/CUDA
# import chain (the dev box has no GPU). `subtasks.py` passes `ctx.model`.


def _extract(model, schema: "type[BaseModel]", instructions: str, text: str):
    """Standalone structured extraction (mirrors TaskContext.extract).

    Tries with_structured_output, then a JSON-mode fallback for models without
    tool-calling. Returns a validated `schema` instance or None.
    """
    from langchain.messages import HumanMessage, SystemMessage
    try:
        structured = model.with_structured_output(schema)
        return structured.invoke([SystemMessage(content=instructions), HumanMessage(content=text)])
    except Exception as exc:
        print(f"[gpsr] structured extract failed ({exc}); trying JSON fallback")
    try:
        prompt = (
            f"{instructions}\n\nRespond ONLY with a JSON object matching this "
            f"schema:\n{json.dumps(schema.model_json_schema())}"
        )
        reply = model.invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        match = re.search(r"\{.*\}", str(reply.content), re.DOTALL)
        if match:
            return schema.model_validate(json.loads(match.group(0)))
    except Exception as exc:
        print(f"[gpsr] JSON-fallback extract failed ({exc})")
    return None


def parse_command(model, command: str, world: WorldModel) -> Plan:
    """Parse ONE command string into a grounded Plan (needs the LLM)."""
    instructions = f"{prompts.PARSE_INSTRUCTIONS}\n\n{world.vocab_prompt()}"
    raw = _extract(model, RawPlan, instructions, command)
    if raw is None:
        print(f"[gpsr] parse failed for command: {command!r}")
        return Plan(steps=[], source=command)
    plan = ground_plan(raw, world, source=command)
    ungrounded = [u for s in plan.steps for u in s.unresolved]
    if ungrounded:
        print(f"[gpsr] command {command!r}: {len(ungrounded)} ungrounded ref(s): {ungrounded}")
    return plan


def parse_commands(model, utterance: str, world: WorldModel) -> list[tuple[str, Plan]]:
    """Split an utterance into commands and parse each into a Plan.

    Returns ``[(command_text, plan), ...]`` capped at GPSR_MAX_COMMANDS. An empty
    list means nothing was understood (the caller re-asks once — §5.2).
    """
    split = _extract(model, prompts.CommandList, prompts.SPLIT_COMMANDS_INSTRUCTIONS, utterance)
    commands = (split.commands if split else None) or ([utterance] if utterance.strip() else [])
    max_n = int(os.getenv("GPSR_MAX_COMMANDS", "3"))
    return [(c, parse_command(model, c, world)) for c in commands[:max_n]]


def _demo() -> None:
    """No-robot dry run: type a command, see the typed plan + spoken plan.

        python -m tasks.GPSR.parse          # needs OPENROUTER_API_KEY, no robot

    The Phase-0 "type a command, read back the spoken plan" check
    (docs/GPSR_DESIGN.md §10) without bringing up the hardware stack.
    """
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    from .plan import render_plan_speech

    load_dotenv()
    model = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("WALKIE_MODEL", "anthropic/claude-sonnet-4.5"),
        temperature=0,
    )
    world = load_world()
    print("GPSR parser dry run — enter a command (blank to quit).")
    while True:
        try:
            line = input("\ncommand> ").strip()
        except EOFError:
            break
        if not line:
            break
        for text, plan in parse_commands(model, line, world):
            steps = [f"{s.primitive.value}({s.args})" + ("" if s.grounded else f" !{s.unresolved}") for s in plan.steps]
            print(f"  [{ 'OK ' if plan.is_complete else 'GAP'}] {text!r}")
            for st in steps:
                print(f"      - {st}")
            print(f"      say: {render_plan_speech(plan)}")


if __name__ == "__main__":
    _demo()
