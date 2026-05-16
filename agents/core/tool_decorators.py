from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, tool as _lc_tool


_PARALLELABLE_ATTR = "_walkie_parallelable"


def _wrap(func: Callable[..., Any] | BaseTool, parallelable: bool) -> BaseTool:
    if isinstance(func, BaseTool):
        t = func
    else:
        # Plain @tool — callers needing Google-style docstring parsing should
        # apply @tool(parse_docstring=True) themselves before this decorator.
        t = _lc_tool(func)
    setattr(t, _PARALLELABLE_ATTR, parallelable)
    return t


def parallelable_tool(func: Callable[..., Any] | BaseTool) -> BaseTool:
    """Mark a tool as safe to run in parallel with neighboring parallelable tools."""
    return _wrap(func, parallelable=True)


def sequential_tool(func: Callable[..., Any] | BaseTool) -> BaseTool:
    """Mark a tool as requiring exclusive execution (no parallel neighbors)."""
    return _wrap(func, parallelable=False)


def is_parallelable(tool_obj: BaseTool) -> bool:
    """Return True if a tool has been marked parallelable. Default: False (safer)."""
    return bool(getattr(tool_obj, _PARALLELABLE_ATTR, False))
