"""Harder LLM stress battery — 50 parametrized challenges for a local model.

A breadth-and-depth scorecard beyond the must-pass gate in ``test_llm.py``.
These probe the weak spots of small self-hosted models on the work walkie-agent-v2
asks of an LLM: multi-step reasoning, strict output formats, tool-arg extraction,
tool selection among distractors, GPSR-style nested planning, negation/grounding,
and multi-turn coreference.

Unlike ``test_llm.py`` (a strict gate that should be green), this battery is a
*scorecard*: some cases are expected to fail on weaker models, and that pass/fail
spread is the signal. Skipped unless a local server is reachable (see llm_harness).

    LOCAL_MODEL=qwen3.5-9b uv run pytest tests/test_llm_hard.py -v
    LOCAL_MODEL=gemma4     uv run pytest tests/test_llm_hard.py -v
    # one family:
    LOCAL_MODEL=gemma4     uv run pytest tests/test_llm_hard.py -v -k reasoning
"""
from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from llm_harness import SKIP, Plan, _extract, _model, _tool_calls

pytestmark = SKIP


# --------------------------------------------------------------------------- #
# matchers
# --------------------------------------------------------------------------- #
def _nums(s: str) -> list[str]:
    return re.findall(r"-?\d+", s or "")


def _answer_has(out: str, accepted) -> bool:
    """True if any accepted answer appears in `out`. Digit answers match a number
    token; word/time answers match on a loose boundary (case-insensitive)."""
    low = (out or "").lower()
    for a in accepted:
        a = str(a)
        if a.lstrip("-").isdigit():
            if a in _nums(out):
                return True
        elif re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", low):
            return True
    return False


def _ask(prompt: str) -> str:
    return _model().invoke(prompt).content or ""


def _items(out: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,\n]", out or "") if t.strip()]


# =========================================================================== #
# Family 1 — multi-step reasoning & math word problems (12)
# =========================================================================== #
REASONING = [
    ("boxes-trips", "A robot carries 3 boxes per trip and must move 17 boxes. How many trips are needed? Reply with only the number.", ["6"]),
    ("shelf-cups", "A shelf has 5 rows and each row holds 4 cups. How many cups in total? Reply with only the number.", ["20"]),
    ("line-move", "A robot starts at position 2 on a number line, moves +5, then -3. What is its final position? Reply with only the number.", ["4"]),
    ("two-rates", "One robot packs 4 boxes per minute, another packs 6 per minute. Working together, how many minutes to pack 30 boxes? Reply with only the number.", ["3"]),
    ("path-time", "A path is 12 meters long. A robot moves at 0.5 meters per second. How many seconds to traverse it? Reply with only the number.", ["24"]),
    ("empty-chairs", "There are 3 tables, each with 4 chairs, and 2 chairs at each table are occupied. How many chairs are empty in total? Reply with only the number.", ["6"]),
    ("finish-time", "It is 14:45 and a task takes 35 minutes. What time does it finish? Reply in 24-hour HH:MM.", ["15:20"]),
    ("tallest", "Alice is taller than Bob. Bob is taller than Carol. Who is tallest? Reply with only the name.", ["alice"]),
    ("turn-facing", "You are facing north. You turn right twice (90 degrees each). Which direction are you facing now? Reply with one word.", ["south"]),
    ("apples", "You have 5 apples, give away 2, then buy 4 more. How many apples now? Reply with only the number.", ["7"]),
    ("gripper", "Each box weighs 2 kg and the gripper can lift at most 9 kg. What is the maximum number of whole boxes it can carry at once? Reply with only the number.", ["4"]),
    ("dozen", "You have 2 dozen eggs and use 5. How many eggs remain? Reply with only the number.", ["19"]),
]


@pytest.mark.parametrize("cid,prompt,accepted", REASONING, ids=[c[0] for c in REASONING])
def test_reasoning(cid, prompt, accepted):
    out = _ask(prompt)
    assert _answer_has(out, accepted), f"{cid}: expected one of {accepted}, got {out!r}"


# =========================================================================== #
# Family 2 — strict instruction / output-format following (8)
# =========================================================================== #
FORMAT = [
    ("yes-no", "Is 10 greater than 3? Reply with only the word YES or NO.",
     lambda o: re.match(r"\s*yes\b", o, re.I) is not None),
    ("count-letters", "How many letters are in the word 'robot'? Reply with only the number.",
     lambda o: "5" in _nums(o)),
    ("opposite", "Reply with a single uppercase word: the opposite of 'open'.",
     lambda o: "clos" in o.lower()),
    ("third-word", "Reply with only the third word of this sentence: The quick brown fox.",
     lambda o: re.search(r"\bbrown\b", o.lower()) is not None),
    ("json-ok", 'Output only valid JSON with a single key "ok" whose value is the boolean true.',
     lambda o: _json_ok(o)),
    ("five-words", "Describe a robot in exactly five words.",
     lambda o: len(re.findall(r"[A-Za-z']+", o)) == 5),
    ("stove-room", "In one word, which room of a house has a stove? Reply with only the room name.",
     lambda o: "kitchen" in o.lower()),
    ("count-r", "How many times does the letter r appear in the word 'strawberry'? Reply with only the number.",
     lambda o: "3" in _nums(o)),
]


def _json_ok(out: str) -> bool:
    import json
    m = re.search(r"\{.*\}", out or "", re.DOTALL)
    if not m:
        return False
    try:
        return json.loads(m.group(0)).get("ok") is True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("cid,prompt,check", FORMAT, ids=[c[0] for c in FORMAT])
def test_format(cid, prompt, check):
    out = _ask(prompt)
    assert check(out), f"{cid}: format not honored, got {out!r}"


# =========================================================================== #
# Family 3 — tool-argument extraction (8)
# =========================================================================== #
@tool
def navigate_to(location: str) -> str:
    """Drive the robot to a named location."""
    return f"navigating to {location}"


@tool
def pick_object(obj: str) -> str:
    """Pick up an object by name."""
    return f"picking {obj}"


@tool
def set_temperature(celsius: int) -> str:
    """Set the room temperature in degrees Celsius."""
    return f"set to {celsius}"


@tool
def count_objects(obj: str) -> str:
    """Count how many of an object are visible."""
    return f"counting {obj}"


TOOL_ARG = [
    ("nav-living", "Go to the living room.", "navigate_to", "location", "living room"),
    ("pick-mug", "Pick up the red mug.", "pick_object", "obj", "mug"),
    ("set-22", "Set the temperature to 22 degrees.", "set_temperature", "celsius", 22),
    ("count-chairs", "How many chairs are there? Use a tool.", "count_objects", "obj", "chair"),
    ("nav-kitchen", "Drive to the kitchen, please.", "navigate_to", "location", "kitchen"),
    ("pick-bottle", "Grab the bottle from the table.", "pick_object", "obj", "bottle"),
    ("set-18", "Make the temperature 18 degrees.", "set_temperature", "celsius", 18),
    ("nav-bedroom", "Could you head over to the bedroom?", "navigate_to", "location", "bedroom"),
]


@pytest.mark.parametrize("cid,utter,tool_name,key,expected", TOOL_ARG, ids=[c[0] for c in TOOL_ARG])
def test_tool_arg_extraction(cid, utter, tool_name, key, expected):
    m = _model().bind_tools([navigate_to, pick_object, set_temperature, count_objects])
    calls = _tool_calls(m.invoke(utter))
    assert any(c["name"] == tool_name for c in calls), f"{cid}: expected {tool_name}, got {calls}"
    args = next(c["args"] for c in calls if c["name"] == tool_name)
    val = args.get(key)
    if isinstance(expected, int):
        assert str(val).strip().rstrip(".0") == str(expected) or _answer_has(str(val), [expected]), \
            f"{cid}: bad numeric arg {args}"
    else:
        assert expected.lower() in str(val).lower(), f"{cid}: bad arg {args}"


# =========================================================================== #
# Family 4 — tool selection among distractors (8)
# =========================================================================== #
@tool
def speak(text: str) -> str:
    """Say something out loud to people."""
    return "spoke"


@tool
def get_battery() -> str:
    """Report the robot's current battery level."""
    return "battery 80%"


@tool
def take_photo() -> str:
    """Capture a photo with the robot's camera."""
    return "photo taken"


_SELECT_TOOLS = [navigate_to, pick_object, speak, get_battery, take_photo]

TOOL_SELECT = [
    ("say-hello", "Say hello to the guests.", "speak"),
    ("battery", "What is your battery level?", "get_battery"),
    ("photo", "Take a picture of the room.", "take_photo"),
    ("bring-cup", "Bring me the cup.", "pick_object"),
    ("move-entrance", "Move to the entrance.", "navigate_to"),
    ("announce", "Tell everyone the food is ready.", "speak"),
    ("battery2", "How much battery do you have left?", "get_battery"),
    ("photo2", "Photograph the whiteboard.", "take_photo"),
]


@pytest.mark.parametrize("cid,utter,expected_tool", TOOL_SELECT, ids=[c[0] for c in TOOL_SELECT])
def test_tool_selection(cid, utter, expected_tool):
    m = _model().bind_tools(_SELECT_TOOLS)
    names = [c["name"] for c in _tool_calls(m.invoke(utter + " Use a tool."))]
    assert expected_tool in names, f"{cid}: expected {expected_tool}, got {names}"


# =========================================================================== #
# Family 5 — GPSR-style nested plan extraction (6)
# =========================================================================== #
GPSR = [
    ("kitchen-milk", "Go to the kitchen, get the milk, and bring it to the living room.", ["kitchen", "milk", "living room"]),
    ("office-person", "Find a person in the office and tell them the meeting started.", ["office", "person"]),
    ("book-shelf", "Pick up the book from the desk and put it on the shelf.", ["book", "desk", "shelf"]),
    ("count-bottles", "Count the bottles on the counter and report the number.", ["bottle", "counter"]),
    ("trash-bin", "Take the trash to the bin near the door.", ["trash", "bin"]),
    ("lead-guest", "Lead the guest from the entrance to the sofa.", ["entrance", "sofa"]),
]


@pytest.mark.parametrize("cid,command,entities", GPSR, ids=[c[0] for c in GPSR])
def test_gpsr_plan(cid, command, entities):
    out, path = _extract(
        Plan,
        f"Break this robot command into ordered steps: '{command}'",
        '{"steps": [{"action": <verb>, "target": <string>}, ...]}',
        required_keys=("steps",),
        ok=lambda p: len(p.steps) >= 2,
    )
    if out is None:
        pytest.fail(f"{cid}: no nested step plan produced via any path")
    blob = " ".join(f"{s.action} {s.target}" for s in out.steps).lower()
    missing = [e for e in entities if e.lower() not in blob]
    assert not missing, f"{cid}: plan dropped {missing} (via {path}): {[s.model_dump() for s in out.steps]}"


# =========================================================================== #
# Family 6 — negation & grounding (4)
# =========================================================================== #
GROUNDING = [
    ("no-apple", "Name two fruits, but do NOT mention apple. Reply comma-separated, fruits only.",
     lambda o: not re.search(r"\bapple", o.lower()) and len(_items(o)) >= 2),
    ("sky-false", "Reply with only the word true or false: The sky is green.",
     lambda o: "false" in o.lower() and not re.search(r"\btrue\b", o.lower())),
    ("odd-one-out", "Which word does not belong: apple, banana, carrot, mango? Reply with only that word.",
     lambda o: "carrot" in o.lower()),
    ("no-red", "List three colors, but do not include red. Reply comma-separated.",
     lambda o: not re.search(r"\bred\b", o.lower()) and len(_items(o)) >= 3),
]


@pytest.mark.parametrize("cid,prompt,check", GROUNDING, ids=[c[0] for c in GROUNDING])
def test_grounding(cid, prompt, check):
    out = _ask(prompt)
    assert check(out), f"{cid}: grounding/negation failed, got {out!r}"


# =========================================================================== #
# Family 7 — multi-turn coreference / correction / memory (4)
# =========================================================================== #
def _conv(turns):
    msgs = []
    for role, text in turns:
        msgs.append(HumanMessage(text) if role == "h" else AIMessage(text))
    return msgs


MULTITURN = [
    ("coref", [("h", "The target object is the blue box."), ("a", "Understood."),
               ("h", "What color is the target object? Reply with one word.")],
     lambda o: "blue" in o.lower()),
    ("correction", [("h", "My name is Sam."), ("a", "ok"),
                    ("h", "Actually, call me Alex from now on."), ("a", "ok"),
                    ("h", "What is my name? Reply with one word.")],
     lambda o: "alex" in o.lower() and not re.search(r"\bsam\b", o.lower())),
    ("memory-distractor", [("h", "Remember this access code: 7421."), ("a", "ok"),
                           ("h", "What is 2 plus 2?"), ("a", "4"),
                           ("h", "What was the access code I gave you?")],
     lambda o: "7421" in o),
    ("pronoun", [("h", "There is a cat and a dog. The dog is sleeping. "
                  "Which animal is sleeping? Reply with one word.")],
     lambda o: "dog" in o.lower()),
]


@pytest.mark.parametrize("cid,turns,check", MULTITURN, ids=[c[0] for c in MULTITURN])
def test_multiturn(cid, turns, check):
    out = _model().invoke(_conv(turns)).content or ""
    assert check(out), f"{cid}: multi-turn failed, got {out!r}"
