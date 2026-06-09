"""Per-challenge task launcher support.

A *task* is a single RoboCup challenge living under ``tasks/<NAME>/``:

    tasks/<NAME>/
        config.toml          # env-var overrides (model, tuning) — loaded ABOVE base config.toml
        prompt.md            # appended to the Walkie main agent's system prompt
        prompts/<agent>.md   # optional per-sub-agent prompt addenda (vision_agent, ...)
        run.sh               # thin launcher that exports WALKIE_TASK_DIR then runs the app
        README.md            # what the challenge is + how this task is wired

The whole mechanism is env-driven — *no monkey-patching*. ``run.sh`` exports
``WALKIE_TASK_DIR``; ``main.py`` calls :func:`load_task_config` right after
``load_dotenv()`` and *before* the base ``load_config()``, so task values win
over base ``config.toml`` (both use ``setdefault``). The shared agent factory
(:func:`agents.core.agent.create_walkie_agent`) calls :func:`apply_task_prompt`
for every agent, so each agent automatically picks up its optional addendum.

Precedence ends up being:

    shell env > .env > task config.toml > base config.toml > code default

To add a challenge: ``cp -r tasks/_template tasks/MyChallenge``, edit its
``config.toml`` / ``prompt.md``, then ``./tasks/MyChallenge/run.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

from walkie_config import load_config

# tasks/ lives directly under the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"


def active_task_dir() -> Path | None:
    """Return the active task directory from ``WALKIE_TASK_DIR``, or ``None``.

    A relative path is resolved against the repo root. Returns ``None`` when
    the var is unset/blank or doesn't point at a real directory, so the normal
    (task-less) boot is the natural fallback.
    """
    raw = os.getenv("WALKIE_TASK_DIR", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p if p.is_dir() else None


def active_task_name() -> str | None:
    """Human-readable name of the active task (``WALKIE_TASK`` or the dir name)."""
    d = active_task_dir()
    if d is None:
        return None
    return os.getenv("WALKIE_TASK", "").strip() or d.name


def load_task_config() -> int:
    """Load ``<task>/config.toml`` as an override layer above base config.toml.

    Must run *after* ``load_dotenv()`` and *before* the base ``load_config()``
    so task values (filled via ``setdefault``) sit above base config but below
    ``.env`` / shell env. Returns the number of keys filled (0 when no task is
    active or the file is absent).
    """
    d = active_task_dir()
    if d is None:
        return 0
    cfg = d / "config.toml"
    if not cfg.is_file():
        return 0
    return load_config(cfg)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def task_prompt_addendum(agent_name: str) -> str:
    """Return the task-specific prompt addendum for one agent, or ``""``.

    Looks for ``tasks/<task>/prompts/<agent_name>.md``. For the main agent
    (``name="walkie_agent"``) the shorthand ``tasks/<task>/prompt.md`` is also
    accepted, so the common "just tweak the orchestrator" case needs a single
    file at the task root.
    """
    d = active_task_dir()
    if d is None:
        return ""
    candidates = [d / "prompts" / f"{agent_name}.md"]
    if agent_name == "walkie_agent":
        candidates.append(d / "prompt.md")
    for c in candidates:
        text = _read(c)
        if text:
            return text
    return ""


def apply_task_prompt(agent_name: str, base_prompt: str) -> str:
    """Append the agent's task addendum (if any) under a ``# Current task`` heading.

    Appending (not replacing) keeps the agent's identity, the no-plain-text
    ``speak`` contract, and its delegation rules intact while layering the
    challenge-specific instructions on top.
    """
    add = task_prompt_addendum(agent_name)
    if not add:
        return base_prompt
    task = active_task_name() or "task"
    return f"{base_prompt.rstrip()}\n\n# Current task: {task}\n\n{add}"
