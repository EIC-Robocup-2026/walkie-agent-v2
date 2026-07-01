"""RoboCup@Home robot Inspection task.

A deterministic scaffold that walks the referee's inspection: notice the entry door
open and drive through, stop for a person who steps in front (safety), reach the
inspection points, declare external devices, prove the speaker is loud enough, then
move to the exit on a signal. Language understanding is done with lightweight custom
model calls (``ctx.extract`` / ``ctx.model.invoke``); ``INSPECTION_AGENT_MODE=1``
swaps the referee Q&A for the full Walkie agent. Run with:

    uv run python -m tasks.Inspection.run
"""

from __future__ import annotations

from .subtasks import build_inspection_task

__all__ = ["build_inspection_task"]
