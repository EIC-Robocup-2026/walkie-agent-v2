from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware

from ..tool_decorators import is_parallelable


# ToolCallRequest is exported from langchain.agents.middleware in langchain 1.2;
# we import lazily to keep this module robust to older typing layouts.
try:
    from langchain.agents.middleware import ToolCallRequest  # noqa: F401
except ImportError:  # pragma: no cover
    ToolCallRequest = Any  # type: ignore[assignment,misc]


@dataclass
class _Coord:
    """Cross-call coordination state for one AIMessage's tool_calls batch."""

    # call_id -> group index (0..N-1)
    call_to_group: dict[str, int]
    # group_idx -> Event signaling that group has finished.
    group_done: list[asyncio.Event]
    # group_idx -> remaining # of calls in that group not yet done.
    group_remaining: list[int]
    # Number of calls that have entered this batch (for cleanup detection).
    entered: int = 0
    # Lock for atomic remaining decrement + Event set.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _group_consecutive_parallelable(
    tool_calls: list[dict[str, Any]],
    parallelable_names: set[str],
) -> tuple[dict[str, int], list[int]]:
    """Returns (call_id -> group_idx, group_size_per_idx).

    seq, par, par, par, seq -> [[seq], [par,par,par], [seq]]
    Group sizes -> [1, 3, 1].
    """
    call_to_group: dict[str, int] = {}
    group_sizes: list[int] = []
    cur_group_idx: int | None = None
    cur_size = 0
    for c in tool_calls:
        is_par = c["name"] in parallelable_names
        if is_par:
            if cur_group_idx is None:
                cur_group_idx = len(group_sizes)
                group_sizes.append(0)
                cur_size = 0
            call_to_group[c["id"]] = cur_group_idx
            cur_size += 1
            group_sizes[cur_group_idx] = cur_size
        else:
            # Flush current parallel group, then add a singleton group.
            if cur_group_idx is not None:
                cur_group_idx = None
                cur_size = 0
            new_idx = len(group_sizes)
            group_sizes.append(1)
            call_to_group[c["id"]] = new_idx
    return call_to_group, group_sizes


class ToolGroupingMiddleware(AgentMiddleware):
    """Enforces sequential/parallel tool grouping.

    `seq -> par -> par -> par -> seq` becomes `seq, [par‖par‖par], seq`:
    consecutive parallelable tool calls run concurrently; a sequential tool
    runs alone, blocking until the previous group finishes.

    Implementation: ToolNode invokes all tool_calls of an AIMessage concurrently
    via asyncio.gather, calling our `awrap_tool_call` for each. We coordinate
    across these concurrent invocations using per-batch asyncio.Events:
    each call awaits the previous group's done-event, then runs, then signals
    its own group when its remaining counter hits 0.

    Tool parallelism is determined by the `_walkie_parallelable` attribute set
    by `parallelable_tool` / `sequential_tool` decorators.
    """

    def __init__(self) -> None:
        super().__init__()
        # Map from id(AIMessage) -> _Coord. Cleaned up when the last call done.
        self._batches: dict[int, _Coord] = {}
        self._batches_lock = threading.Lock()

    @staticmethod
    def _parallelable_set(state) -> set[str]:
        # Reach into state to find the AgentMiddleware-bound tools.
        # In create_agent, the active tools list isn't in state; instead we
        # examine each call's `tool` reference passed to awrap_tool_call.
        # This helper is kept as a placeholder; the actual parallelable check
        # happens in `_grouping_for_batch` below using the tool registry the
        # middleware was given — see `_compute_for` which uses request.tool.
        return set()

    def _get_or_init_batch(
        self,
        ai_msg,
        parallelable_names: set[str],
    ) -> _Coord:
        key = id(ai_msg)
        with self._batches_lock:
            existing = self._batches.get(key)
            if existing is not None:
                return existing
            tool_calls = list(getattr(ai_msg, "tool_calls", []) or [])
            call_to_group, group_sizes = _group_consecutive_parallelable(
                tool_calls, parallelable_names
            )
            coord = _Coord(
                call_to_group=call_to_group,
                group_done=[asyncio.Event() for _ in group_sizes],
                group_remaining=list(group_sizes),
            )
            self._batches[key] = coord
            return coord

    def _drop_batch_if_done(self, ai_msg, coord: _Coord) -> None:
        # Drop when every group is signaled complete.
        if all(ev.is_set() for ev in coord.group_done):
            with self._batches_lock:
                self._batches.pop(id(ai_msg), None)

    @staticmethod
    def _build_parallelable_names(state, request_tool) -> set[str]:
        """Discover parallelable tools.

        Strategy: look at the tools we were called with via the request, and
        also at `state["messages"][-1].tool_calls` paired with the bound tools
        on the agent. As a robust fallback, we mark just the current call's
        tool as parallelable based on its own attribute, and accumulate as
        more wrap_tool_call invocations come in (one per call). To keep the
        algorithm correct we instead rely on a single source: the tool
        registry passed at construction time is unavailable here, so we
        inspect each call's tool via the AIMessage and `request.tool`.

        Concretely: callers should pass tools through `tools=` to create_agent
        and decorate with parallelable_tool/sequential_tool. We rebuild the
        names set lazily by reading the bound tools off the runtime if
        available, otherwise we fall back to a per-call lookup.
        """
        # Fallback path: build names by reading every tool referenced in
        # the current AIMessage tool_calls and looking up via state.
        # Since we can't see the agent's bound tools from here, we accept
        # a registry override via _set_tool_registry below; if absent we
        # treat only the explicitly-known tool as parallelable.
        return set()

    # Public hook for the agent factory to inject the full tool registry.
    def set_tool_registry(self, tools) -> None:
        self._parallelable_names: set[str] = {
            t.name for t in tools if is_parallelable(t)
        }

    @property
    def parallelable_names(self) -> set[str]:
        return getattr(self, "_parallelable_names", set())

    def wrap_tool_call(self, request, handler):
        # Sync path: no event-loop coordination available. Best we can do is
        # run tools as-is. Sync execution in create_agent already serializes
        # tools through ToolNode in a single thread, so grouping here is a
        # no-op — parallelism only matters in the async path.
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        # Identify this tool call's batch (the AIMessage that emitted it).
        messages = request.state["messages"]
        ai_msg = messages[-1]
        names = self.parallelable_names
        coord = self._get_or_init_batch(ai_msg, names)
        call_id = request.tool_call["id"]
        my_group = coord.call_to_group.get(call_id)

        # If we couldn't classify this call, just run it.
        if my_group is None:
            return await handler(request)

        # Wait for the previous group to be fully done.
        if my_group > 0:
            await coord.group_done[my_group - 1].wait()

        try:
            result = await handler(request)
            return result
        finally:
            async with coord.lock:
                coord.group_remaining[my_group] -= 1
                if coord.group_remaining[my_group] <= 0:
                    coord.group_done[my_group].set()
            self._drop_batch_if_done(ai_msg, coord)
