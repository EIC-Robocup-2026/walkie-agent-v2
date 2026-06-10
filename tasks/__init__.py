"""Per-challenge task launchers for walkie-agent-v2.

See :mod:`tasks.runtime` for the loader API and ``tasks/_template/`` for the
skeleton to copy when adding a RoboCup challenge.
"""

from __future__ import annotations

from .runtime import (
    active_task_dir,
    active_task_name,
    apply_task_prompt,
    load_task_config,
    task_prompt_addendum,
)

__all__ = [
    "active_task_dir",
    "active_task_name",
    "apply_task_prompt",
    "load_task_config",
    "task_prompt_addendum",
]
