from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import ACTUATOR_AGENT_SYSTEM_PROMPT
from .tools import make_actuator_tools


def create_actuator_agent(model, walkieAI, walkie: WalkieInterface, *, ctx=None):
    """``ctx`` (a TaskContext) unlocks the skill-backed tools (go_to_location,
    pick_up_object, place_object_down, go_through_door); without it the agent keeps
    only the low-level move/arm primitives."""
    tools = make_actuator_tools(walkie, walkieAI, agent_name="actuator", ctx=ctx)
    return create_walkie_agent(
        name="actuator_agent",
        model=model,
        tools=tools,
        system_prompt=ACTUATOR_AGENT_SYSTEM_PROMPT,
    )
