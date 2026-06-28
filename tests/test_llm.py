"""LLM ability-coverage gate for a local OpenAI-compatible model.

Points a standalone ``ChatOpenAI`` (the exact stack the agents use) at a local
vLLM server and verifies the abilities walkie-agent-v2 actually depends on. Used
to qualify a self-hosted model (e.g. the qwen3.5-9b / gemma4 MTP servers) as a
drop-in for the OpenRouter model in ``main.build_model``.

Mirrors ``tests/test_gpsr_coverage.py``: standalone model, SKIPPED unless the
endpoint is reachable AND serves a model, runs on the dev box (no robot). The
shared harness lives in ``tests/llm_harness.py``; the broader 50-case stress
battery is ``tests/test_llm_hard.py``.

Ability tiers (per CLAUDE.md — the agent is "provider-agnostic as long as the
model supports tool calls"):
  * HARD: basic chat, instruction following, single tool call, tool round-trip,
    structured output, multi-turn, math word-problem reasoning, tool selection
    among distractors, multi-step tool chaining.
  * SOFT (reported as xfail, never blocks): parallel tool calls, reasoning
    channel, nested/GPSR-style structured output.

Run against each server (one at a time on the 24 GB GPU):
    LOCAL_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=qwen3.5-9b uv run pytest tests/test_llm.py -s
    LOCAL_BASE_URL=http://localhost:8000/v1 LOCAL_MODEL=gemma4     uv run pytest tests/test_llm.py -s
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm_harness import (
    MODEL,
    SKIP,
    Plan,
    _extract,
    _model,
    _raw_chat,
    _tool_calls,
)

pytestmark = SKIP


# --- tools the agent-style tests bind -------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is 21 degrees Celsius and sunny."


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def set_alarm(time: str) -> str:
    """Set an alarm for a given time."""
    return f"alarm set for {time}"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    return "email sent"


# =========================== HARD abilities ================================= #
def test_basic_chat():
    """Model answers a plain question with non-empty text."""
    resp = _model().invoke("In one short sentence, what is a robot?")
    assert isinstance(resp.content, str) and resp.content.strip(), f"empty content: {resp!r}"


def test_instruction_following():
    """Follows a precise output-format instruction. Unambiguous: a question with
    a strict format, so a 'helpful' reinterpretation still has to honor it."""
    resp = _model().invoke("What is 7 multiplied by 6? Reply with only the number, no words.")
    assert "42" in resp.content, f"did not follow instruction: {resp.content!r}"


def test_single_tool_call():
    """bind_tools → the model emits a structured tool call with the right args.
    The load-bearing capability for the whole agent stack."""
    m = _model().bind_tools([get_weather])
    resp = m.invoke("What's the weather in Paris right now? Use the available tool.")
    calls = _tool_calls(resp)
    assert calls, f"model emitted no tool call: content={resp.content!r}"
    assert any(c["name"] == "get_weather" for c in calls), f"wrong tool: {calls}"
    args = next(c["args"] for c in calls if c["name"] == "get_weather")
    assert "paris" in str(args.get("city", "")).lower(), f"bad args: {args}"


def test_tool_roundtrip():
    """Full ReAct loop: tool call → feed result back → final answer uses it."""
    m = _model().bind_tools([get_weather])
    first = m.invoke([HumanMessage("What's the weather in Paris? Use the tool.")])
    calls = _tool_calls(first)
    assert calls, f"no tool call to round-trip: {first.content!r}"
    call = next(c for c in calls if c["name"] == "get_weather")
    messages = [
        HumanMessage("What's the weather in Paris? Use the tool."),
        first,
        ToolMessage(content=get_weather.invoke(call["args"]), tool_call_id=call["id"]),
    ]
    final = m.invoke(messages)
    text = (final.content or "").lower()
    assert "21" in text or "sunny" in text, f"final answer ignored tool result: {final.content!r}"


class Person(BaseModel):
    """A person extracted from text."""
    name: str = Field(description="the person's full name")
    age: int = Field(description="the person's age in years")


def test_structured_output():
    """The agent must extract a typed object (GPSR parse). Passes if native
    structured output OR a JSON-mode fallback works — the tasks/base.extract
    contract."""
    out, path = _extract(
        Person,
        "Extract the person: 'John Smith just turned 42 last week.'",
        '{"name": <string>, "age": <integer>}',
        required_keys=("name", "age"),
        ok=lambda p: "john" in p.name.lower() and p.age == 42,
    )
    assert out is not None, "structured extraction failed (native + both JSON fallbacks)"
    print(f"[structured_output] ok via {path}")


def test_multiturn_memory():
    """Carries context across turns."""
    msgs = [
        HumanMessage("My name is Walkie. Remember it. Reply 'ok'."),
        AIMessage("ok"),
        HumanMessage("What is my name? Reply with just the name."),
    ]
    resp = _model().invoke(msgs)
    assert "walkie" in resp.content.lower(), f"lost multi-turn context: {resp.content!r}"


# =========================== HARD: harder challenges ======================== #
def test_math_word_problem():
    """Multi-step quantitative reasoning (ceiling division), not a one-step lookup."""
    import re
    resp = _model().invoke(
        "A robot carries 3 boxes per trip and must move 17 boxes. "
        "How many trips are needed? Reply with only the number."
    )
    assert "6" in re.findall(r"\d+", resp.content or ""), f"wrong/unclear answer: {resp.content!r}"


def test_tool_selection_among_distractors():
    """With several tools bound, pick the RIGHT one and don't fire the others."""
    m = _model().bind_tools([get_weather, set_alarm, send_email, add])
    resp = m.invoke("What's the weather like in Tokyo right now? Use a tool.")
    names = [c["name"] for c in _tool_calls(resp)]
    assert "get_weather" in names, f"did not select get_weather: {names} / {resp.content!r}"
    assert set(names) <= {"get_weather"}, f"fired distractor tools too: {names}"


def test_multistep_tool_chaining():
    """Chain tools across turns: add then multiply, feeding each result back.
    Computes (12 + 8) * 3 = 60."""
    m = _model().bind_tools([add, multiply])
    msgs = [HumanMessage("Compute (12 + 8) * 3 using the tools, one operation at a time.")]
    impl = {"add": add, "multiply": multiply}
    saw = []
    for _ in range(6):
        resp = m.invoke(msgs)
        msgs.append(resp)
        calls = _tool_calls(resp)
        if not calls:
            break
        for c in calls:
            result = impl[c["name"]].invoke(c["args"])
            saw.append((c["name"], result))
            msgs.append(ToolMessage(content=str(result), tool_call_id=c["id"]))
    # The ability is the CHAIN: add(12,8)=20 feeds multiply(20,3)=60. We don't
    # require a final text answer — by the "no plain text output" contract a model
    # may legitimately end with empty content, as gemma4 does here.
    assert ("add", 20) in saw, f"did not add 12+8=20: {saw}"
    assert ("multiply", 60) in saw, f"did not chain 20*3=60: {saw}"


# =========================== SOFT abilities ================================= #
def test_parallel_tool_calls():
    """Some agents fan out parallelable tools in one turn. Many models serialize
    instead — reported, never blocks."""
    m = _model().bind_tools([get_weather])
    resp = m.invoke("Get the weather for BOTH Paris and Tokyo. Call the tool once per city, in a single turn.")
    calls = _tool_calls(resp)
    if len(calls) < 2:
        pytest.xfail(f"model did not emit parallel tool calls (got {len(calls)})")
    cities = {str(c["args"].get("city", "")).lower() for c in calls}
    assert any("paris" in c for c in cities) and any("tokyo" in c for c in cities), \
        f"parallel calls but wrong cities: {cities}"


def test_reasoning_channel():
    """With a reasoning parser active and thinking enabled, the server exposes a
    SEPARATE reasoning trace. Checked via the raw API because langchain drops
    vLLM's ``reasoning`` field. Optional — the agent doesn't consume it."""
    msg = _raw_chat("What is 17 * 23?", enable_thinking=True, max_tokens=1024)
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if not reasoning or not str(reasoning).strip():
        pytest.xfail("no separate reasoning channel exposed (parser off or no think block)")
    assert len(str(reasoning)) > 20, f"reasoning trace too thin: {reasoning!r}"
    assert "391" in (msg.get("content") or "") or "391" in str(reasoning), \
        f"reasoning present but wrong answer: content={msg.get('content')!r}"


def test_nested_structured_output():
    """Harder, GPSR-shaped extraction: a command → an ordered list of typed steps.
    Soft: small models often manage flat objects but not nested lists."""
    out, path = _extract(
        Plan,
        "Break this robot command into ordered steps: "
        "'Go to the kitchen, pick up the apple, and bring it to me.'",
        '{"steps": [{"action": <verb>, "target": <string>}, ...]}',
        required_keys=("steps",),
        ok=lambda p: len(p.steps) >= 2,
    )
    if out is None:
        pytest.xfail("model could not produce a nested step plan via any path")
    assert all(s.action.strip() and s.target.strip() for s in out.steps), \
        f"step with empty action/target (via {path}): {[s.model_dump() for s in out.steps]}"
    blob = " ".join(f"{s.action} {s.target}" for s in out.steps).lower()
    assert "kitchen" in blob and "apple" in blob, \
        f"plan dropped key entities (via {path}): {[s.model_dump() for s in out.steps]}"
    print(f"[nested_structured] ok via {path} with {len(out.steps)} steps")
