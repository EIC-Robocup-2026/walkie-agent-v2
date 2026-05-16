from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from ..robot_context import RobotContext


def _build_section() -> str:
    try:
        ctx = RobotContext.get()
    except RuntimeError:
        return ""
    parts = [f"## Stage: {ctx.stage}"]
    speech = ctx.recent_speech_text()
    if speech:
        parts.append(f"## Recently spoken (any agent)\n{speech}")
    return "\n\n".join(parts)


def _append_to_system(request: ModelRequest, extra: str) -> ModelRequest:
    if not extra.strip():
        return request
    if request.system_message is None:
        new_msg = SystemMessage(content=extra.strip())
    else:
        existing = request.system_message.content
        if isinstance(existing, list):
            new_content = [*existing, {"type": "text", "text": "\n\n" + extra.strip()}]
        else:
            new_content = f"{existing}\n\n{extra.strip()}"
        new_msg = SystemMessage(content=new_content)
    return request.override(system_message=new_msg)


class RobotContextMiddleware(AgentMiddleware):
    """Injects the cross-agent speech log + current stage into the system prompt.

    Lets every agent (Walkie, Actuator, Vision) see what each one has just said
    so they can avoid speaking redundantly.
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        return handler(_append_to_system(request, _build_section()))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        return await handler(_append_to_system(request, _build_section()))
