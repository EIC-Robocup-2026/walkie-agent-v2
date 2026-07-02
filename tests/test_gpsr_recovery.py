"""Offline tests for GPSR command-receipt recovery (ReceiveAndPlanCommands).

The rulebook's escape valve: when a command can't be parsed, re-ask a bounded
number of times (each −30), then request a custom operator (−20/command) rather
than silently forfeiting the run. Drives the REAL subtask with a scripted fake
ctx and a stubbed parse_commands — no LLM, no robot. The bug this guards: the old
flow left ctx.data["commands"] unset on repeated failure, so the robot executed
nothing and went home with 0 points.
"""

from __future__ import annotations

import pytest

from tasks.base import StepResult
from tasks.GPSR import prompts
from tasks.GPSR.plan import Plan, PlanStep, Primitive
from tasks.GPSR.subtasks import ReceiveAndPlanCommands, _is_affirmative, _is_negative


def _ok_plan() -> Plan:
    return Plan(
        steps=[PlanStep(Primitive.NAVIGATE, {"target": "kitchen"}, "go to the kitchen")],
        source="go to the kitchen",
    )


def _fake_parse(model, utterance, world):
    """Stub parser: a 'good'/'kitchen' utterance plans; anything else is garbage."""
    if "good" in utterance.lower() or "kitchen" in utterance.lower():
        return [("go to the kitchen", _ok_plan())]
    return []


class _Ctx:
    """Fake TaskContext: replays scripted ask answers, records speech."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.saids: list[str] = []
        self.asked: list[str] = []
        self.ask_retries: list[int] = []
        self.ask_kwargs: list[dict] = []  # timeout/min_silence_ms pass-throughs
        self.scored: list[tuple[str, int]] = []  # (key, n) for each ctx.score()
        self.data: dict = {}
        self.world = object()  # non-None so run() proceeds (parse_commands is stubbed)
        self.model = object()
        self.disable_listening = False  # answers come via ask() = the STT path

    def ask(self, question, retries=1, **kwargs):
        self.asked.append(question)
        self.ask_retries.append(retries)
        self.ask_kwargs.append(kwargs)
        return self._answers.pop(0) if self._answers else ""  # exhausted -> silence

    def say(self, text):
        self.saids.append(text)

    def score(self, key, n=1):
        self.scored.append((key, n))


@pytest.fixture(autouse=True)
def _stub_parser(monkeypatch):
    monkeypatch.setattr("tasks.GPSR.subtasks.parse_commands", _fake_parse)


def _run(ctx):
    return ReceiveAndPlanCommands().run(ctx)


def test_first_answer_parses_with_no_rephrasing():
    ctx = _Ctx(["go to the kitchen"])
    assert _run(ctx) is StepResult.DONE
    assert len(ctx.asked) == 1                      # no re-ask
    # The loop owns re-prompting: it must call ask with retries=0 so ctx.ask
    # does NOT re-prompt internally (that would inflate the -30 count + clock).
    assert ctx.ask_retries == [0]
    assert len(ctx.data["commands"]) == 1
    assert prompts.CONFIRM_RECEIVED in ctx.saids
    assert prompts.ASK_REPHRASE not in ctx.saids
    assert prompts.REQUEST_CUSTOM_OPERATOR not in ctx.saids


def test_empty_parse_triggers_one_rephrasing_then_succeeds():
    ctx = _Ctx(["", "go to the kitchen"])           # blank, then a good command
    assert _run(ctx) is StepResult.DONE
    assert ctx.saids.count(prompts.ASK_REPHRASE) == 1
    assert prompts.REQUEST_CUSTOM_OPERATOR not in ctx.saids
    assert prompts.CONFIRM_RECEIVED in ctx.saids
    assert len(ctx.data["commands"]) == 1


def test_persistent_failure_requests_custom_operator_then_gives_up(monkeypatch):
    monkeypatch.setenv("GPSR_MAX_REPHRASINGS", "1")
    monkeypatch.setenv("GPSR_CUSTOM_OPERATOR_ATTEMPTS", "1")
    ctx = _Ctx([])                                  # every ask returns silence
    status = _run(ctx)
    assert status is StepResult.DONE                # NOT abort — stays attending
    assert prompts.REQUEST_CUSTOM_OPERATOR in ctx.saids
    assert prompts.GIVE_UP_ON_COMMANDS in ctx.saids
    assert prompts.CONFIRM_RECEIVED not in ctx.saids
    assert ctx.data["commands"] == []               # set (not unset) -> no silent forfeit
    assert len(ctx.asked) == 3                       # 1 initial + 1 rephrase + 1 custom, bounded


def test_custom_operator_recovers_the_command(monkeypatch):
    monkeypatch.setenv("GPSR_MAX_REPHRASINGS", "1")
    monkeypatch.setenv("GPSR_CUSTOM_OPERATOR_ATTEMPTS", "2")
    ctx = _Ctx(["bad", "bad", "go to the kitchen"])  # fail, rephrase-fail, custom-success
    assert _run(ctx) is StepResult.DONE
    assert prompts.ASK_REPHRASE in ctx.saids
    assert prompts.REQUEST_CUSTOM_OPERATOR in ctx.saids
    assert prompts.CONFIRM_RECEIVED in ctx.saids
    assert len(ctx.data["commands"]) == 1


def test_custom_operator_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GPSR_MAX_REPHRASINGS", "1")
    monkeypatch.setenv("GPSR_USE_CUSTOM_OPERATOR", "0")
    ctx = _Ctx([])
    assert _run(ctx) is StepResult.DONE
    assert prompts.REQUEST_CUSTOM_OPERATOR not in ctx.saids   # never requested
    assert prompts.GIVE_UP_ON_COMMANDS in ctx.saids
    assert ctx.data["commands"] == []
    assert len(ctx.asked) == 2                                # 1 initial + 1 rephrase, then give up


# --- plan-confirmation gate (GPSR_CONFIRM_PLAN) -----------------------------

def test_confirm_gate_off_by_default_does_not_ask():
    """Gate off (default): no extra ask, and the command is confirmed implicitly."""
    ctx = _Ctx(["go to the kitchen"])
    _run(ctx)
    assert len(ctx.asked) == 1                       # only the command ask, no confirm
    assert ctx.data["commands"][0].confirmed is True
    assert prompts.ASK_CONFIRM_PLAN.format(n=1) not in ctx.asked


def test_confirm_gate_yes_approves(monkeypatch):
    monkeypatch.setenv("GPSR_CONFIRM_PLAN", "1")
    ctx = _Ctx(["go to the kitchen", "yes please"])
    _run(ctx)
    assert prompts.ASK_CONFIRM_PLAN.format(n=1) in ctx.asked
    assert ctx.data["commands"][0].confirmed is True
    assert prompts.PLAN_CONFIRMED in ctx.saids


def test_confirm_gate_no_declines(monkeypatch):
    monkeypatch.setenv("GPSR_CONFIRM_PLAN", "1")
    ctx = _Ctx(["go to the kitchen", "no, don't do that"])
    _run(ctx)
    assert ctx.data["commands"][0].confirmed is False
    assert prompts.PLAN_REJECTED in ctx.saids


def test_confirm_gate_unclear_proceeds_by_default(monkeypatch):
    monkeypatch.setenv("GPSR_CONFIRM_PLAN", "1")
    ctx = _Ctx(["go to the kitchen", "hmm what was that"])   # neither yes nor no
    _run(ctx)
    assert ctx.data["commands"][0].confirmed is True         # GPSR_CONFIRM_DEFAULT=proceed


def test_confirm_gate_unclear_skips_when_configured(monkeypatch):
    monkeypatch.setenv("GPSR_CONFIRM_PLAN", "1")
    monkeypatch.setenv("GPSR_CONFIRM_DEFAULT", "skip")
    ctx = _Ctx(["go to the kitchen", ""])                    # silence, even after re-ask
    _run(ctx)
    assert ctx.data["commands"][0].confirmed is False


def test_affirmative_negative_word_matching():
    assert _is_affirmative("yes") and _is_affirmative("ok go ahead")
    assert _is_negative("no") and _is_negative("please don't")
    # "no" must not be triggered by substrings like "now"/"know"
    assert not _is_negative("do it now")
    assert not _is_affirmative("") and not _is_negative("")
