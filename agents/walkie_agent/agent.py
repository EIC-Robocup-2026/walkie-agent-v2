from __future__ import annotations

from agents.core.agent import create_walkie_agent
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface

from .prompts import WALKIE_AGENT_SYSTEM_PROMPT
from .tools import make_walkie_main_tools


def create_walkie_main_agent(
    model,
    walkieAI,
    walkie: WalkieInterface,
    db: WalkieVectorDB,
    actuator_agent,
    vision_agent,
    *,
    scene_store=None,
):
    """Build the main Walkie agent (orchestrator over actuator + vision)."""
    tools = make_walkie_main_tools(
        walkie,
        walkieAI,
        db,
        actuator_agent,
        vision_agent,
        agent_name="walkie",
        scene_store=scene_store,
    )
    return create_walkie_agent(
        name="walkie_agent",
        model=model,
        tools=tools,
        system_prompt=WALKIE_AGENT_SYSTEM_PROMPT,
    )
