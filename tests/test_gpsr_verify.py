"""Offline tests for GPSR per-command verification (GPSR_VERIFY_COMMANDS).

The competition failure this guards: the referee reads all three commands in ONE
continuous, halting stream; STT mishears a command (or the split blurs where one
ends and the next begins) and the robot burns the 7-minute run executing the
wrong errand. With the gate on, the robot reads each command back (yes/no),
re-captures just the rejected one, recovers a merged-away command, and only then
executes — the sure-points strategy. Drives the REAL subtask with a scripted
fake ctx and a stubbed parse_commands — no LLM, no robot.
"""

from __future__ import annotations

import pytest

from tasks.base import StepResult
from tasks.GPSR import prompts
from tasks.GPSR.plan import Plan, PlanStep, Primitive
from tasks.GPSR.subtasks import ReceiveAndPlanCommands


def _plan(source: str) -> Plan:
    return Plan(
        steps=[PlanStep(Primitive.NAVIGATE, {"target": "kitchen"}, source)],
        source=source,
    )


def _fake_parse(model, utterance, world):
    """Stub parser: one command per ';'-separated segment; 'bad' segments fail;
    'vague' segments parse to text with an EMPTY (unusable) plan.

    Mirrors the real contract (list of (text, plan) pairs, [] when nothing was
    understood, empty-steps Plan when the text grounded to nothing) while
    letting a test script multi-command batches and garbage re-captures
    without an LLM.
    """
    out = []
    for part in utterance.split(";"):
        part = part.strip()
        if not part or "bad" in part.lower():
            continue
        plan = Plan(steps=[], source=part) if "vague" in part.lower() else _plan(part)
        out.append((part, plan))
    return out


class _Ctx:
    """Fake TaskContext: replays scripted ask answers, records speech + kwargs."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.saids: list[str] = []
        self.asked: list[str] = []
        self.ask_kwargs: list[dict] = []  # timeout/min_silence_ms pass-throughs
        self.scored: list[tuple[str, int]] = []
        self.data: dict = {}
        self.world = object()  # non-None so run() proceeds (parse_commands is stubbed)
        self.model = object()
        self.disable_listening = False

    def ask(self, question, retries=1, **kwargs):
        self.asked.append(question)
        self.ask_kwargs.append(kwargs)
        return self._answers.pop(0) if self._answers else ""  # exhausted -> silence

    def say(self, text):
        self.saids.append(text)

    def score(self, key, n=1):
        self.scored.append((key, n))


@pytest.fixture(autouse=True)
def _stub_parser(monkeypatch):
    monkeypatch.setattr("tasks.GPSR.subtasks.parse_commands", _fake_parse)


@pytest.fixture
def _verify_on(monkeypatch):
    monkeypatch.setenv("GPSR_VERIFY_COMMANDS", "1")


def _run(ctx):
    return ReceiveAndPlanCommands().run(ctx)


def _pen_count(ctx) -> int:
    return sum(1 for key, _ in ctx.scored if key == "pen_rephrasing")


def _texts(ctx) -> list[str]:
    return [c.utterance for c in ctx.data["commands"]]


# --- gate off (default) ------------------------------------------------------

def test_verify_off_by_default_asks_nothing_extra():
    """In-code default is the pre-change flow: no read-back, no got-all ask."""
    ctx = _Ctx(["go to the kitchen"])
    assert _run(ctx) is StepResult.DONE
    assert len(ctx.asked) == 1                       # only the command ask
    assert prompts.GREET_OPERATOR in ctx.saids
    assert prompts.GREET_OPERATOR_VERIFY not in ctx.saids


def test_batch_ask_uses_wide_listen_window():
    """The 3-command capture must widen timeout + end-of-speech silence — the
    1 s mic default cuts a halting referee's stream at the first stumble."""
    ctx = _Ctx(["go to the kitchen"])
    _run(ctx)
    assert ctx.ask_kwargs[0] == {"timeout": 90.0, "min_silence_ms": 2500}


# --- gate on: confirmation ---------------------------------------------------

def test_all_yes_keeps_commands_verbatim(_verify_on):
    ctx = _Ctx([
        "go to the kitchen; guide charlie to the exit; count the apples",
        "yes", "yes", "yes",
    ])
    assert _run(ctx) is StepResult.DONE
    assert ctx.saids.count(prompts.GREET_OPERATOR_VERIFY) == 1
    assert _texts(ctx) == ["go to the kitchen", "guide charlie to the exit", "count the apples"]
    assert _pen_count(ctx) == 0                      # no re-captures
    # 3 commands heard -> no "did I get all of them?" ask
    assert not any(a.startswith("I heard") for a in ctx.asked)


def test_no_recaptures_just_that_command(_verify_on):
    """A 'no' on command 2 re-captures ONLY command 2; 1 and 3 stand as heard."""
    ctx = _Ctx([
        "go to the kitchen; guide daniel to the exit; count the apples",
        "yes",                        # command 1 confirmed
        "no",                         # command 2 misheard
        "guide charlie to the exit",  # re-speak just command 2
        "yes",                        # corrected command 2 confirmed
        "yes",                        # command 3 confirmed
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen", "guide charlie to the exit", "count the apples"]
    assert _pen_count(ctx) == 1                      # one −30 re-ask, tallied honestly
    assert prompts.ASK_REPEAT_ONE.format(n=2) in ctx.asked
    # The single-command re-capture uses the shorter re-listen window.
    i = ctx.asked.index(prompts.ASK_REPEAT_ONE.format(n=2))
    assert ctx.ask_kwargs[i] == {"timeout": 30.0, "min_silence_ms": 1500}


def test_unclear_verdict_proceeds_by_default(_verify_on):
    """A mumbled non-yes/non-no must not forfeit the command (GPSR_CONFIRM_DEFAULT)."""
    ctx = _Ctx([
        "go to the kitchen; count the apples; guide charlie to the exit",
        "yes", "hmm what", "yes",
    ])
    _run(ctx)
    assert len(ctx.data["commands"]) == 3            # command 2 kept as heard
    assert _pen_count(ctx) == 0


def test_recaptures_bounded_then_best_effort(_verify_on, monkeypatch):
    """Persistent 'no' + garbage re-speaks can't loop forever: after the budget
    the best understanding stands (partial credit beats a forfeit)."""
    monkeypatch.setenv("GPSR_VERIFY_MAX_RECAPTURES", "1")
    ctx = _Ctx([
        "go to the kitchen",
        "no",            # reject
        "bad mumbling",  # re-capture parses to nothing
        "no",            # reject the (unchanged) read-back again -> budget spent
        "yes",           # got-all ask (only 1 of 3 commands heard)
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen"]      # best effort kept
    assert prompts.VERIFY_BEST_EFFORT.format(n=1) in ctx.saids
    assert prompts.VERIFY_RECAPTURE_MISSED in ctx.saids
    assert _pen_count(ctx) == 1                      # bounded by the budget


def test_mixed_answer_treated_as_negative(_verify_on):
    """Negative is checked FIRST (mirrors _confirm_plan): on 'okay, no, that is
    wrong', wrongly keeping a misheard command wastes an errand; wrongly
    re-capturing a correct one costs only −30 — bias toward the cheap mistake."""
    ctx = _Ctx([
        "go to the kitchen",
        "okay, no, that is wrong",     # affirmative token + negative -> negative wins
        "go to the bedroom",           # re-speak
        "yes",                         # confirm the corrected command
        "yes",                         # got-all
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the bedroom"]
    assert _pen_count(ctx) == 1


def test_unplannable_recapture_is_not_read_back(_verify_on):
    """A re-capture that parses to text with an UNUSABLE plan must not be read
    back for confirmation — that round would end in a guaranteed forfeit."""
    ctx = _Ctx([
        "go to the kitchen",
        "no",                      # reject
        "do the vague thing",      # parses, but empty plan -> treated as missed
        "yes",                     # re-confirm of the ORIGINAL text
        "yes",                     # got-all
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen"]   # original kept, vague never adopted
    assert prompts.VERIFY_RECAPTURE_MISSED in ctx.saids
    assert not any("vague" in a for a in ctx.asked)


def test_exhausted_skip_mode_declines_instead_of_executing(_verify_on, monkeypatch):
    """GPSR_VERIFY_EXHAUSTED=skip (the config value): a command the operator
    rejected past the budget is announced, kept unconfirmed (so ExecuteCommands
    skips it via the existing decline machinery), and its plan is not spoken."""
    monkeypatch.setenv("GPSR_VERIFY_MAX_RECAPTURES", "0")
    monkeypatch.setenv("GPSR_VERIFY_EXHAUSTED", "skip")
    ctx = _Ctx([
        "go to the kitchen; count the apples; guide charlie to the exit",
        "no",    # command 1 rejected -> budget (0) already spent -> give up
        "yes",   # command 2
        "yes",   # command 3
    ])
    assert _run(ctx) is StepResult.DONE
    cmds = ctx.data["commands"]
    assert [c.confirmed for c in cmds] == [False, True, True]
    assert cmds[0].result_note == "skipped: operator could not confirm the command"
    assert prompts.VERIFY_GIVE_UP.format(n=1) in ctx.saids
    # No plan speech (and no claimed speak_plan point) for the declined command.
    assert prompts.PLAN_PREAMBLE.format(n=1) not in " ".join(ctx.saids)
    assert sum(1 for k, _ in ctx.scored if k == "speak_plan") == 2
    assert _pen_count(ctx) == 0                   # budget 0: no re-ask was made


def test_exhausted_execute_mode_is_default(monkeypatch, _verify_on):
    """In-code default keeps the pre-knob behaviour: best effort still runs."""
    monkeypatch.setenv("GPSR_VERIFY_MAX_RECAPTURES", "0")
    ctx = _Ctx([
        "go to the kitchen; count the apples; guide charlie to the exit",
        "no", "yes", "yes",
    ])
    _run(ctx)
    assert [c.confirmed for c in ctx.data["commands"]] == [True, True, True]
    assert prompts.VERIFY_BEST_EFFORT.format(n=1) in ctx.saids


# --- gate on: mis-split (merged-command) recovery ----------------------------

def test_missing_command_recovered_and_verified(_verify_on):
    """Halting speech merged two commands into one parse entry: the got-all ask
    captures the missing command, which is verified like the rest."""
    ctx = _Ctx([
        "go to the kitchen; count the apples",
        "yes", "yes",           # both heard commands confirmed
        "no",                   # "did I get all of them?" -> no
        "guide charlie to the exit",  # the missing command
        "yes",                  # its verification (now 3 == max -> loop ends)
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen", "count the apples", "guide charlie to the exit"]
    assert prompts.ASK_GOT_ALL.format(n=2) in ctx.asked
    assert prompts.ASK_VERIFY_COMMAND.format(n=3, command="guide charlie to the exit") in ctx.asked
    assert _pen_count(ctx) == 1                      # the say-missing re-ask


def test_got_all_yes_moves_on(_verify_on):
    """Operators legitimately give fewer than three (practice): 'yes' ends it."""
    ctx = _Ctx([
        "go to the kitchen",
        "yes",   # verify command 1
        "yes",   # got-all: that was everything
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen"]
    assert _pen_count(ctx) == 0


def test_missing_capture_failure_breaks_not_loops(_verify_on):
    """A garbage missing-command capture breaks out (clock > maybe-recoverable)."""
    ctx = _Ctx([
        "go to the kitchen",
        "yes",           # verify command 1
        "no",            # got-all -> claims one is missing
        "bad mumbling",  # capture parses to nothing -> break
    ])
    assert _run(ctx) is StepResult.DONE
    assert _texts(ctx) == ["go to the kitchen"]
    assert prompts.VERIFY_RECAPTURE_MISSED in ctx.saids
    assert ctx.asked.count(prompts.ASK_SAY_MISSING) == 1   # no retry loop
