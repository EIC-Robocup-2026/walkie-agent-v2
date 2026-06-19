"""Real-LLM test for the command splitter (parse_commands).

The splitter must NOT over-split a single multi-clause command into several — that
would push the operator's real later commands off the GPSR_MAX_COMMANDS cap and
forfeit them. (Caught in a dry run: the rulebook's own rephrasing example, "go to
the kitchen, find a coke, and bring it to me", was being split into THREE.) It
must still split genuinely separate commands. Needs OPENROUTER_API_KEY; skipped
offline.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from tasks.GPSR.parse import parse_commands
from tasks.GPSR.world import load_world

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="splitter test needs OPENROUTER_API_KEY (LLM); skipped offline",
)


@pytest.fixture(scope="module")
def world():
    return load_world()


@pytest.fixture(scope="module")
def model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("WALKIE_MODEL", "anthropic/claude-sonnet-4.5"),
        temperature=0,
    )


@pytest.mark.parametrize(
    "utterance",
    [
        "go to the kitchen, find a coke, and bring it to me",
        "go to the office, grab the pringles, and put them on the kitchen table",
    ],
)
def test_chained_single_command_is_not_oversplit(model, world, utterance):
    """A single command chaining several actions toward one goal stays as ONE."""
    cmds = parse_commands(model, utterance, world)
    assert len(cmds) == 1, [c for c, _ in cmds]


def test_three_distinct_commands_split_to_three(model, world):
    """Three genuinely separate tasks must still split into three commands."""
    utterance = (
        "bring me a coke from the kitchen. "
        "then guide Charlie to the bedroom. "
        "finally count the apples on the desk"
    )
    cmds = parse_commands(model, utterance, world)
    assert len(cmds) == 3, [c for c, _ in cmds]
