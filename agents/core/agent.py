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

from tasks.runtime import apply_task_prompt

from .middleware import (
    PerceptionContextMiddleware,
    RobotContextMiddleware,
    ToolGroupingMiddleware,
)


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
                trigger=("tokens", int(os.getenv("WALKIE_SUMMARIZE_AT_TOKENS", "6000"))),
                keep=("messages", int(os.getenv("WALKIE_SUMMARIZE_KEEP_MSGS", "12"))),
            )
        )
    if enable_todos:
        middleware.append(TodoListMiddleware())
    middleware.extend(
        [
            PerceptionContextMiddleware(),
            RobotContextMiddleware(),
            grouping,
            *extra_middleware,
        ]
    )

    agent = create_agent(
        model=model,
        tools=tools_list,
        # Layer any tasks/<active>/prompt(s) addendum on top of the base prompt.
        # No-op (returns system_prompt unchanged) when no task is active.
        system_prompt=apply_task_prompt(name, system_prompt),
        middleware=middleware,
        checkpointer=checkpointer or InMemorySaver(),
        name=name,
    )
    return agent
