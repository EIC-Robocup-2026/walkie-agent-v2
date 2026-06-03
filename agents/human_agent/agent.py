from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import HUMAN_AGENT_SYSTEM_PROMPT
from .tools import make_human_tools


def create_human_agent(
    model, walkieAI, walkie: WalkieInterface, *, people_store=None
):
    """Build the Walkie Human (HRI) sub-agent — live-camera people understanding.

    ``people_store`` is reserved for the later face-recognition slice; the
    current tools (describe / count) do not use it.
    """
    tools = make_human_tools(
        walkie, walkieAI, agent_name="human", people_store=people_store
    )
    return create_walkie_agent(
        name="human_agent",
        model=model,
        tools=tools,
        system_prompt=HUMAN_AGENT_SYSTEM_PROMPT,
    )
