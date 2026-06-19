from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage


def _ai_messages(response: Any) -> list[AIMessage]:
    """Pull the AIMessage(s) out of whatever a model handler returned.

    Handlers may return an ``AIMessage`` directly, a ``ModelResponse`` (``.result``
    list), or an ``ExtendedModelResponse`` (``.model_response.result``). Stay
    defensive — a trace failure must never break the agent loop.
    """
    if isinstance(response, AIMessage):
        return [response]
    result = getattr(response, "result", None)
    if result is None:
        model_response = getattr(response, "model_response", None)
        result = getattr(model_response, "result", None)
    if not result:
        return []
    return [m for m in result if isinstance(m, AIMessage)]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("reasoning") or "")
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _print_trace(agent_name: str, response: Any) -> None:
    try:
        for msg in _ai_messages(response):
            reasoning = msg.additional_kwargs.get("reasoning_content") or msg.additional_kwargs.get(
                "reasoning"
            )
            if reasoning:
                print(f"[THINK][{agent_name}] (reasoning) {str(reasoning).strip()}")
            text = _text(msg.content).strip()
            if text:
                print(f"[THINK][{agent_name}] {text}")
            calls = getattr(msg, "tool_calls", None) or []
            if calls:
                rendered = ", ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
                print(f"[THINK][{agent_name}] tool_calls: [{rendered}]")
            if not text and not calls and not reasoning:
                print(f"[THINK][{agent_name}] (end turn — no tool calls, no speech)")
    except Exception as exc:  # pragma: no cover - tracing must never crash the loop
        print(f"[THINK][{agent_name}] <trace error: {exc}>")


class TraceMiddleware(AgentMiddleware):
    """Prints each agent's reasoning + tool decisions on every model call.

    This is what makes the *thinking process* visible when running offline. It only
    observes — it returns the model response unchanged. Attach it to every agent
    (via the factory) so all four agents' turns are traced, including sub-agents
    invoked inside ``delegate_to_*``.
    """

    def __init__(self, *, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        response = handler(request)
        _print_trace(self.agent_name, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        response = await handler(request)
        _print_trace(self.agent_name, response)
        return response
