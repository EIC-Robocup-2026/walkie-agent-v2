"""RoboCup@Home 2026 Finals task (rulebook chapter 6).

Hybrid scaffold + agent: deterministic handlers for the fixed high-value problems
(welcome the guest, move the laundry basket, close the dishwasher) plus an
agent-driven room patrol for the open-ended ones (trash, misplaced objects, person
requests). Run with ``uv run python -m tasks.Final.run``.
"""

from __future__ import annotations

from .scoring import FINAL_SHEET
from .subtasks import build_final_task

__all__ = ["build_final_task", "FINAL_SHEET"]
