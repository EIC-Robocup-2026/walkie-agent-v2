from __future__ import annotations

from agents.core.agent import create_walkie_agent
from interfaces.walkie_interface import WalkieInterface

from .prompts import DATABASE_AGENT_SYSTEM_PROMPT
from .tools import make_database_tools


def create_database_agent(
    model, walkieAI, walkie: WalkieInterface, *, world=None, ctx=None
):
    """Build the Walkie Database sub-agent (long-term 3D spatial-memory specialist).

    ``world`` is a :class:`walkie_world.WalkieWorld` whose scene graph + map + people
    the tools query (query_text / default_location_for / find_person_by_caption / ...);
    when None the tools report memory is off. ``ctx`` (a TaskContext) is accepted so the
    shared stack can pass it uniformly — its ``ctx.world`` is used when ``world`` is not
    given explicitly.
    """
    if world is None and ctx is not None:
        world = ctx.world
    tools = make_database_tools(
        walkie, walkieAI, agent_name="database", world=world
    )
    return create_walkie_agent(
        name="database_agent",
        model=model,
        tools=tools,
        system_prompt=DATABASE_AGENT_SYSTEM_PROMPT,
    )
