from __future__ import annotations

import math
import os
import time
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from ..robot_context import RobotContext


def _heading_str(heading) -> str:
    if heading is None:
        return ""
    return f" heading={math.degrees(float(heading)):+.0f}°"


def _frame_str(rec: dict) -> str:
    h = rec.get("frame_h")
    v = rec.get("frame_v")
    if not h and not v:
        return ""
    if h == "center" and v == "center":
        return " frame=center"
    return f" frame=({h or '?'}, {v or '?'})"


def _format_objects(objs: list[dict]) -> str:
    if not objs:
        return "  (none)"
    lines = []
    for o in objs:
        cls = o.get("class") or o.get("class_name") or "?"
        pos = o.get("position_3d") or o.get("position")
        conf = o.get("conf") or o.get("confidence")
        cap = o.get("caption") or ""
        if pos:
            x, y, z = pos
            pos_str = f"({x:+.2f}, {y:+.2f}, {z:+.2f})"
        else:
            pos_str = "(unknown)"
        cap_str = f' "{cap}"' if cap else ""
        conf_str = f" conf={conf:.2f}" if conf is not None else ""
        heading_str = _heading_str(o.get("heading"))
        frame_str = _frame_str(o)
        lines.append(f"  - {cls} @ {pos_str}{conf_str}{heading_str}{frame_str}{cap_str}")
    return "\n".join(lines)


def _build_section() -> str:
    """Compute the dynamic perception section, or '' if not applicable."""
    try:
        ctx = RobotContext.get()
    except RuntimeError:
        return ""
    if ctx.stage != "ready":
        return ""
    snap = ctx.perception_snapshot()
    if not snap:
        return ""
    ts = snap.get("ts")
    age = time.time() - ts if ts else None
    stale_sec = float(os.getenv("PERCEPTION_STALE_SEC", "10"))
    if age is not None and age > stale_sec:
        return ""
    age_str = f" (updated {age:.1f}s ago)" if age is not None else ""
    objects_str = _format_objects(snap.get("objects", []))
    return f"## Current perception{age_str}\nObjects in view:\n{objects_str}"


def _append_to_system(request: ModelRequest, extra: str) -> ModelRequest:
    """Return a new ModelRequest with `extra` appended to the system message."""
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


class PerceptionContextMiddleware(AgentMiddleware):
    """Injects the current perception snapshot into the system prompt on each model call.

    No-op during the explore stage or if the perception JSON is missing/stale.
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
