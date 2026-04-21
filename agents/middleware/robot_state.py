"""Middleware that injects current robot state into the system prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import SystemMessage

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from agents.robot_state import RobotState


class RobotStateMiddleware(AgentMiddleware):
    """Middleware that appends current robot state to the system message.

    Each model call receives an updated snapshot of position, heading,
    vision status, and arm status so the agent can reason about the
    physical world.
    """

    tools: list = []

    def __init__(self, robot_state: RobotState) -> None:
        """Initialize with a robot state provider.

        Args:
            robot_state: Instance used to read and format current state.
        """
        super().__init__()
        self.robot_state = robot_state

    def _build_system_message(self, request: ModelRequest) -> SystemMessage:
        """Append robot state block to the current system message."""
        state_text = self.robot_state.format_for_prompt()
        extra_text = f"\n\n{state_text}"

        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": extra_text},
            ]
        else:
            new_system_content = [{"type": "text", "text": extra_text.lstrip()}]

        return SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: "Callable[[ModelRequest], ModelResponse]",
    ) -> ModelCallResult:
        """Inject robot state into the system message and run the model."""
        new_system_message = self._build_system_message(request)
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: "Callable[[ModelRequest], Awaitable[ModelResponse]]",
    ) -> ModelCallResult:
        """Async: inject robot state into the system message and run the model."""
        new_system_message = self._build_system_message(request)
        return await handler(request.override(system_message=new_system_message))
