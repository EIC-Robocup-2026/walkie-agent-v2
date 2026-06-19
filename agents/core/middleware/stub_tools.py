from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..robot_context import RobotContext

# ToolCallRequest is exported from langchain.agents.middleware in langchain 1.2;
# imported lazily to stay robust to older typing layouts (same pattern as
# tool_grouping.py).
try:
    from langchain.agents.middleware import ToolCallRequest  # noqa: F401
except ImportError:  # pragma: no cover
    ToolCallRequest = Any  # type: ignore[assignment,misc]


# Tools whose names start with this prefix are sub-agent delegations — they must
# run for real so the sub-agent graphs reason (their own leaf tools then get
# stubbed by the copy of this middleware attached to each sub-agent).
_DELEGATE_PREFIX = "delegate_to_"
# The task-tracker tool from TodoListMiddleware is internal bookkeeping; let it run.
_DEFAULT_PASSTHROUGH = {"write_todos"}


class StubToolMiddleware(AgentMiddleware):
    """Short-circuits leaf tools to a canned success — no robot, no AI server.

    Lets the full agent reasoning loop run offline so the thinking process can be
    inspected. Every tool call is intercepted in ``wrap_tool_call``: leaf tools
    (movement, detection, captioning, memory lookups, ...) never touch hardware or
    HTTP — they return ``"[stub] <name> succeeded."`` instantly and log the call.

    Two exceptions:
      - ``delegate_to_*`` and ``write_todos`` are passed through to the real
        handler, so sub-agents actually run and the todo tracker still works.
      - ``speak`` skips TTS/audio but still records to the cross-agent speech log
        (``RobotContext.add_speech``) so ``RobotContextMiddleware``'s "Recently
        spoken" context stays faithful.

    Both the sync (``wrap_tool_call``) and async (``awrap_tool_call``) paths are
    implemented; the offline runner drives agents synchronously via ``.invoke`` /
    ``.stream``, so the sync path is the one that executes.
    """

    def __init__(self, *, agent_name: str, passthrough: set[str] | None = None) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.passthrough = set(passthrough) if passthrough is not None else set(_DEFAULT_PASSTHROUGH)

    def _is_passthrough(self, name: str) -> bool:
        return name.startswith(_DELEGATE_PREFIX) or name in self.passthrough

    def _stub_message(self, name: str, args: Any, tool_call_id: str) -> ToolMessage:
        # `speak` is special-cased: surface the would-be utterance and preserve the
        # cross-agent speech log without running TTS.
        if name == "speak":
            text = ""
            if isinstance(args, dict):
                text = str(args.get("text", ""))
            print(f"[SPEAK][{self.agent_name}] {text}")
            try:
                RobotContext.get().add_speech(self.agent_name, text)
            except RuntimeError:
                pass
            return ToolMessage(
                content="[stub] spoken.",
                tool_call_id=tool_call_id,
                name=name,
                status="success",
            )

        print(f"[STUB][{self.agent_name}] {name}({args}) -> success")
        return ToolMessage(
            content=f"[stub] {name} succeeded.",
            tool_call_id=tool_call_id,
            name=name,
            status="success",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        name = request.tool_call["name"]
        if self._is_passthrough(name):
            return handler(request)
        return self._stub_message(name, request.tool_call["args"], request.tool_call["id"])

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        name = request.tool_call["name"]
        if self._is_passthrough(name):
            return await handler(request)
        return self._stub_message(name, request.tool_call["args"], request.tool_call["id"])
