from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import VISION_AGENT_SYSTEM_PROMPT
from .tools import make_vision_tools


def create_vision_agent(model, walkieAI, walkie: WalkieInterface):
    tools = make_vision_tools(walkie, walkieAI, agent_name="vision")
    return create_walkie_agent(
        name="vision_agent",
        model=model,
        tools=tools,
        system_prompt=VISION_AGENT_SYSTEM_PROMPT,
    )
