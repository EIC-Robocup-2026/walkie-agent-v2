from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import make_actuator_tools


def create_actuator_agent(model, walkieAI, walkie: WalkieInterface):
    tools = make_actuator_tools(walkie, walkieAI, agent_name="actuator")
    return create_walkie_agent(
        name="actuator_agent",
        model=model,
        tools=tools,
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
