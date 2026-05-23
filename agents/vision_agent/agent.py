from __future__ import annotations

from agents.core.agent import create_walkie_agent
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface

from .prompts import VISION_AGENT_SYSTEM_PROMPT
from .tools import make_vision_tools


def create_vision_agent(
    model, walkieAI, walkie: WalkieInterface, db: WalkieVectorDB, *, scene_store=None
):
    tools = make_vision_tools(
        walkie, walkieAI, db, agent_name="vision", scene_store=scene_store
    )
    return create_walkie_agent(
        name="vision_agent",
        model=model,
        tools=tools,
        system_prompt=VISION_AGENT_SYSTEM_PROMPT,
    )
