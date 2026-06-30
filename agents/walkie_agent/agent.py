from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import make_walkie_main_tools


def create_walkie_main_agent(
    model,
    walkieAI,
    walkie: WalkieInterface,
    actuator_agent,
    vision_agent,
    database_agent,
    *,
    ctx=None,
):
    """Build the main Walkie agent (orchestrator over actuator + vision + database).

    ``ctx`` (a TaskContext) unlocks `handle_person_request`, which runs the GPSR
    command pipeline (parse → repeat → execute) for a person's spoken request.
    """
    tools = make_walkie_main_tools(
        walkie,
        walkieAI,
        actuator_agent,
        vision_agent,
        database_agent,
        agent_name="walkie",
        ctx=ctx,
    )
    return create_walkie_agent(
        name="walkie_agent",
        model=model,
        tools=tools,
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
    )
