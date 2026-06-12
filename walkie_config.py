"""Load tuning config from ``config.toml`` files into the process environment.

The app reads all its tuning via ``os.getenv(NAME, default)``. Rather than
rewrite every call site, this loader reads the root ``config.toml`` — then any
module-local ``services/*/config.toml`` — and, for each leaf value, calls
``os.environ.setdefault(NAME, value)`` — so config files provide defaults but
a real shell variable or a value already loaded from ``.env`` always wins
(``setdefault`` only fills what's missing). The root file loads first, so it
can also override a module file's knob.

Precedence: shell env > .env > root config.toml > services/*/config.toml >
code default.

Call :func:`load_config` once at startup, *after* ``load_dotenv()``, in every
entrypoint (main.py, tool scripts).

Competition tasks may layer a task-local ``tasks/<NAME>/config.toml`` on top by
loading it first (``tasks.common.load_task_config``) — same setdefault
semantics, so the task file overrides these but env/.env still win.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterator


# config.toml lives next to this module (repo root).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"


def _walk(d: dict[str, Any], _prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield (KEY, value) for every scalar leaf, recursing into sub-tables.

    Table names are purely organizational — only the leaf key (which is the
    exact env-var name) and its value matter. Non-string scalars are coerced
    to str so ``os.getenv`` semantics are identical to a value set in .env.
    """
    for key, val in d.items():
        if isinstance(val, dict):
            yield from _walk(val)
        elif isinstance(val, (str, int, float)):
            yield key, ("" if val == "" else str(val))
        elif isinstance(val, bool):  # unreachable (bool handled above via int)
            yield key, "1" if val else "0"


def _load_one(path: Path) -> int:
    """setdefault every leaf of one TOML file; 0 on a missing/unreadable file."""
    if not path.is_file():
        return 0
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"[config] failed to read {path}: {e!r}", file=sys.stderr)
        return 0
    filled = 0
    for key, value in _walk(data):
        if key not in os.environ:
            os.environ[key] = value
            filled += 1
    return filled


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> int:
    """Populate ``os.environ`` defaults from the root + module config files.

    Loads ``path`` (the root ``config.toml``) first, then every
    ``services/*/config.toml`` next to it — first-set wins, so the root can
    override a module knob and env/.env override both. Returns the number of
    keys filled. Missing files are fine — a deployment can run on env vars
    alone.
    """
    p = Path(path)
    filled = _load_one(p)
    for module_cfg in sorted(p.resolve().parent.glob("services/*/config.toml")):
        filled += _load_one(module_cfg)
    return filled
