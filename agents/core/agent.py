from __future__ import annotations

import os
from typing import Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from .middleware import (
    PerceptionContextMiddleware,
    RobotContextMiddleware,
    StubToolMiddleware,
    ToolGroupingMiddleware,
    TraceMiddleware,
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes")


def create_walkie_agent(
    *,
    name: str,
    model,
    tools: Sequence[BaseTool],
    system_prompt: str,
    extra_middleware: Sequence[AgentMiddleware] = (),
    checkpointer=None,
    enable_summarization: bool = True,
    enable_todos: bool = True,
):
    """Factory used by all three agents (Walkie main, Actuator, Vision).

    Built on top of `langchain.agents.create_agent`, with these middlewares:

      - SummarizationMiddleware: compresses old messages once token budget is hit.
      - TodoListMiddleware: gives the agent a `write_todos` task tracker.
      - PerceptionContextMiddleware: appends the latest perception snapshot to
        the system message on every model call.
      - RobotContextMiddleware: appends the cross-agent speech log + stage.
      - ToolGroupingMiddleware: enforces sequential/parallel grouping —
        consecutive `parallelable_tool`s run concurrently, sequential tools
        run alone (see middleware/tool_grouping.py).

    The "no output step" is structural: when the model emits an AIMessage with
    no tool calls, the agent loop ends. The system prompt instructs every
    agent that plain text is internal reasoning — to communicate they must
    call `speak`.
    """
    tools_list = list(tools)

    grouping = ToolGroupingMiddleware()
    grouping.set_tool_registry(tools_list)

    middleware: list[AgentMiddleware] = []
    if enable_summarization:
        middleware.append(
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", int(os.getenv("WALKIE_SUMMARIZE_AT_TOKENS", "18000"))),
                keep=("messages", int(os.getenv("WALKIE_SUMMARIZE_KEEP_MSGS", "12"))),
            )
        )
    # if enable_todos:
    #     middleware.append(TodoListMiddleware())
    middleware.extend(
        [
            # PerceptionContextMiddleware(),
            # RobotContextMiddleware(),
            grouping,
            *extra_middleware,
        ]
    )

    # Offline / dry-run mode (see manual_tests/run_stub_agent.py). Env-gated so
    # production boot is unaffected when the flags are unset.
    if _env_flag("WALKIE_STUB_TOOLS"):
        middleware.append(StubToolMiddleware(agent_name=name))
    if _env_flag("WALKIE_TRACE"):
        middleware.append(TraceMiddleware(agent_name=name))

    agent = create_agent(
        model=model,
        tools=tools_list,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer or InMemorySaver(),
        name=name,
    )
    return agent
