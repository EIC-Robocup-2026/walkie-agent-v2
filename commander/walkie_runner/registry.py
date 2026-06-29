"""Challenge registry + launch resolution — what to run and how.

The set of runnable challenges is *discovered* (every ``<repo>/tasks/*/run.py``,
the same rule ``run.sh`` uses) and decorated with hand-kept metadata: the title,
the rulebook section, and — per challenge — its scorecard env var and any
``*_SLICE`` bring-up variants. A task with no metadata still shows up generically,
so a new challenge added to the repo appears here for free.

Launch is deliberately decoupled from the agent: we exec ``<python> -m
tasks.<NAME>.run`` from the repo root. ``python_launcher()`` prefers the repo's
own ``.venv/bin/python`` (lock-safe — bare ``uv run`` re-resolves and dirties
uv.lock, which the user forbade); it falls back to ``uv run --no-sync`` only when
that venv is absent.

We force each challenge's ``*_SCORECARD_PATH`` into ``walkie-runner-data/`` so (a)
the live-score panel always knows where to read, and (b) the task's default
``gpsr_scorecard.json`` etc. never lands in — and dirties — the repo root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
COMMANDER_DIR = _HERE.parents[1]                  # .../walkie-agent-v2/commander
DATA_DIR = COMMANDER_DIR / "walkie-runner-data"   # scorecards + (future) run logs


def _detect_repo_root() -> Path:
    """The walkie-agent-v2 checkout: $WALKIE_AGENT_ROOT, else walk up to the dir
    that has both ``tasks/`` and ``run.sh``."""
    env = os.getenv("WALKIE_AGENT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    start = COMMANDER_DIR.parent
    for cand in (start, *start.parents):
        if (cand / "tasks").is_dir() and (cand / "run.sh").is_file():
            return cand
    return start


REPO_ROOT = _detect_repo_root()


def python_launcher() -> list[str]:
    """Argv prefix that runs the repo's Python WITHOUT touching uv.lock."""
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return [str(venv)]
    return ["uv", "run", "--no-sync", "python"]   # fallback: no re-lock, no sync


@dataclass(frozen=True)
class SliceOption:
    """A ``*_SLICE`` bring-up selector — pick an isolated stage instead of the
    whole run (e.g. RESTAURANT_SLICE=pick)."""
    env: str
    choices: tuple[str, ...]
    default: str


@dataclass(frozen=True)
class Challenge:
    name: str                       # task dir name, e.g. "GPSR"
    title: str
    description: str
    scorecard_env: str
    rulebook: str = ""
    slice: SliceOption | None = None

    @property
    def module(self) -> str:
        return f"tasks.{self.name}.run"

    @property
    def scorecard_path(self) -> Path:
        return DATA_DIR / f"{self.name.lower()}_scorecard.json"

    def argv(self) -> list[str]:
        return [*python_launcher(), "-m", self.module]

    def env_overrides(self, *, disable_listening: bool, slice_value: str | None,
                      extra: dict[str, str] | None = None) -> dict[str, str]:
        env: dict[str, str] = {
            self.scorecard_env: str(self.scorecard_path),
            "DISABLE_LISTENING": "1" if disable_listening else "0",
        }
        if self.slice and slice_value:
            env[self.slice.env] = slice_value
        if extra:
            env.update(extra)
        return env


# Hand-kept metadata; anything not here is discovered with sensible generic
# defaults. Keys are the task directory names under tasks/.
_KNOWN: dict[str, dict] = {
    "GPSR": dict(
        title="GPSR — General Purpose Service Robot",
        rulebook="5.3",
        description="Parse a spoken command → speak a plan → execute it. "
                    "Mic-driven on the robot; type into the input box in "
                    "DISABLE_LISTENING mode.",
        scorecard_env="GPSR_SCORECARD_PATH",
    ),
    "Restaurant": dict(
        title="Restaurant",
        rulebook="5.5",
        description="Detect a waving customer, take the order at the table, "
                    "fetch from the bar, and serve. Slices isolate bring-up stages.",
        scorecard_env="RESTAURANT_SCORECARD_PATH",
        slice=SliceOption("RESTAURANT_SLICE",
                          ("full", "phase0", "surfaces", "people", "graspplan",
                           "pick", "place"),
                          "full"),
    ),
    "PickAndPlace": dict(
        title="Pick & Place — Serve Breakfast",
        rulebook="5.2",
        description="Clean the dining table into the dishwasher/cabinet/trash, "
                    "then set breakfast. Slices isolate nav/perceive/sort/breakfast.",
        scorecard_env="PNP_SCORECARD_PATH",
        slice=SliceOption("PNP_SLICE",
                          ("full", "nav", "perceive", "sort", "breakfast"),
                          "full"),
    ),
    "Laundry": dict(
        title="Doing Laundry",
        rulebook="5.4",
        description="Retrieve clothes from the washing machine and fold them on a table.",
        scorecard_env="LAUNDRY_SCORECARD_PATH",
    ),
    "HRI": dict(
        title="HRI / Receptionist",
        rulebook="5.1",
        description="Greet arriving guests, learn each one's name and favourite "
                    "drink, and seat them in the living room. Slices isolate the "
                    "bring-up steps.",
        scorecard_env="HRI_SCORECARD_PATH",
        slice=SliceOption("HRI_SLICE",
                          ("full", "seats", "greet", "follow_host"),
                          "full"),
    ),
}


def discover() -> list[Challenge]:
    """Every ``tasks/*/run.py`` as a :class:`Challenge`, known metadata merged in."""
    tasks_dir = REPO_ROOT / "tasks"
    names = sorted(p.parent.name for p in tasks_dir.glob("*/run.py")) \
        if tasks_dir.is_dir() else []
    out: list[Challenge] = []
    for n in names:
        meta = _KNOWN.get(n)
        if meta:
            out.append(Challenge(name=n, **meta))
        else:
            out.append(Challenge(
                name=n, title=n, description=f"tasks/{n}/run.py",
                scorecard_env=f"{n.upper()}_SCORECARD_PATH",
            ))
    return out


def read_scorecard(ch: Challenge) -> dict | None:
    """The challenge's last-written scorecard JSON, or None if it hasn't run.

    Tasks write this only at run end (``run.py``'s ``finally``), so it reflects the
    *last completed or gracefully-stopped* run — not a live in-flight tally.
    """
    try:
        with ch.scorecard_path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
