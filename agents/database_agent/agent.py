from __future__ import annotations

from agents.core.agent import create_walkie_agent
from db.walkie_db import WalkieVectorDB
from interfaces.walkie_interface import WalkieInterface

from .prompts import DATABASE_AGENT_SYSTEM_PROMPT
from .tools import make_database_tools


def create_database_agent(
    model, walkieAI, walkie: WalkieInterface, db: WalkieVectorDB, *, scene_store=None
):
    """Build the Walkie Database sub-agent (long-term spatial memory specialist)."""
    tools = make_database_tools(
        walkie, walkieAI, db, agent_name="database", scene_store=scene_store
    )
    return create_walkie_agent(
        name="database_agent",
        model=model,
        tools=tools,
        system_prompt=DATABASE_AGENT_SYSTEM_PROMPT,
    )
