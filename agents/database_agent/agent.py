from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import DATABASE_AGENT_SYSTEM_PROMPT
from .tools import make_database_tools


def create_database_agent(
    model, walkieAI, walkie: WalkieInterface, *, graphs=None
):
    """Build the Walkie Database sub-agent (long-term 3D spatial-memory specialist).

    ``graphs`` is a :class:`walkie_graphs.WalkieGraphs` instance whose store the
    tools query; when None the tools report that memory is unavailable.
    """
    tools = make_database_tools(
        walkie, walkieAI, agent_name="database", graphs=graphs
    )
    return create_walkie_agent(
        name="database_agent",
        model=model,
        tools=tools,
        system_prompt=DATABASE_AGENT_SYSTEM_PROMPT,
    )
