"""Load tuning config from ``config.toml`` into the process environment.

The app reads all its tuning via ``os.getenv(NAME, default)``. Rather than
rewrite every call site, this loader reads ``config.toml`` and, for each leaf
value, calls ``os.environ.setdefault(NAME, value)`` — so config.toml provides
defaults but a real shell variable or a value already loaded from ``.env``
always wins (``setdefault`` only fills what's missing).

Precedence ends up being: shell env > .env > config.toml > code default.

Call :func:`load_config` once at startup, *after* ``load_dotenv()``, in every
entrypoint (main.py, tools/chroma_viewer.py, tools/scene_explore.py).
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


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> int:
    """Populate ``os.environ`` defaults from ``config.toml``.

    Returns the number of keys filled (i.e. not already set in the
    environment). Missing config file is fine — returns 0 silently so a
    deployment can run on env vars alone.
    """
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"[config] failed to read {p}: {e!r}", file=sys.stderr)
        return 0
    filled = 0
    for key, value in _walk(data):
        if key not in os.environ:
            os.environ[key] = value
            filled += 1
    return filled
