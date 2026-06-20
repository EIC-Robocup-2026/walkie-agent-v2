"""Drive the OFFICIAL RoboCup@Home CommandGenerator over OUR arena vocabulary to
produce a realistic command corpus — the test oracle for parser coverage
(docs/GPSR_DESIGN.md §10.3).

Why this exists: the hand-written ``tests/gpsr_command_corpus.txt`` has one line
per generator category (~56), so "100% coverage" against it proves the parser
handles *each category once*, not the real *distribution* — the rulebook draws
commands from the generator. This module wires the actual generator
(`github.com/RoboCupAtHome/CommandGenerator`) so coverage can be measured over
hundreds of sampled commands instead, which is the strongest offline protection
for the draw-independent 540 (understand + speak-a-plan).

The generator is an EXTERNAL repo, not a dependency of this one (it pulls qrcode /
PIL via its CLI). Only its pure command grammar (``gpsr_commands.CommandGenerator``,
stdlib-only) is imported, located at runtime:

    GPSR_GENERATOR_DIR  — path to the CommandGenerator checkout (or its ``src/``).
                          Default: ~/Documents/GitHub/CommandGenerator.

If it can't be found, :func:`load_generator_cls` raises ImportError with the fix —
callers (the coverage test) skip rather than fail.

Vocabulary comes from OUR :class:`~tasks.GPSR.world.WorldModel` (loaded with
``include_absent=True`` for the full arena), so every generated noun is one the
parser is meant to ground — a coverage miss is a real parser/grounding gap, not a
vocabulary mismatch.

CLI — dump a corpus to stdout (no LLM, no robot)::

    uv run python -m tasks.GPSR.tools.gen_corpus -n 300 --seed 0 > /tmp/corpus.txt
    uv run python -m tasks.GPSR.tools.gen_corpus -n 50 --category people
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from ..world import WorldModel, _singulars, load_world

# The generator's two top-level buckets (gpsr_commands.generate_command_start).
CATEGORIES = ("people", "objects")


def _generator_search_paths() -> list[Path]:
    """Candidate dirs that may contain ``robocupathome_generator`` as an importable
    package (i.e. the dir to put on sys.path)."""
    roots: list[Path] = []
    env = os.getenv("GPSR_GENERATOR_DIR")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path.home() / "Documents" / "GitHub" / "CommandGenerator")
    out: list[Path] = []
    for r in roots:
        # Accept either the repo root (has src/) or the src/ dir itself.
        out.extend([r / "src", r, r.parent])
    return out


def load_generator_cls():
    """Import and return the official ``CommandGenerator`` class, or raise
    ImportError with a remediation hint. Stdlib-only import (no qrcode/PIL)."""
    try:  # already importable?
        from robocupathome_generator.gpsr_commands import CommandGenerator
        return CommandGenerator
    except ImportError:
        pass
    for p in _generator_search_paths():
        pkg = p / "robocupathome_generator" / "gpsr_commands.py"
        if pkg.exists():
            sys.path.insert(0, str(p))
            from robocupathome_generator.gpsr_commands import CommandGenerator
            return CommandGenerator
    raise ImportError(
        "Could not find the RoboCup@Home CommandGenerator. Clone "
        "https://github.com/RoboCupAtHome/CommandGenerator and point "
        "GPSR_GENERATOR_DIR at the checkout (the dir containing src/), e.g. "
        "GPSR_GENERATOR_DIR=~/Documents/GitHub/CommandGenerator."
    )


def _spaced(name: str) -> str:
    """Canonical world key -> the spoken/space form the generator emits."""
    return name.replace("_", " ")


def _singular_category(cat: str) -> str:
    """Best-effort singular of a (normalized, plural) category key: 'drinks'->
    'drink', 'dishes'->'dish', 'cleaning_supplies'->'cleaning_supply'. Falls back
    to the input when it has no plural form ('food')."""
    cands = _singulars(cat)
    return cands[0] if cands else cat


def build_generator(world: WorldModel):
    """Construct a ``CommandGenerator`` whose vocabulary IS the world model's, so
    every generated noun grounds (rooms/locations/objects/names/categories)."""
    cls = load_generator_cls()
    plural_cats = list(world.categories.keys())
    return cls(
        person_names=list(world.names),
        location_names=[_spaced(n) for n in world.locations],
        placement_location_names=[
            _spaced(n) for n, loc in world.locations.items() if loc.placement
        ],
        room_names=[_spaced(n) for n in world.rooms],
        object_names=[_spaced(n) for n in world.objects],
        object_categories_plural=[_spaced(c) for c in plural_cats],
        object_categories_singular=[_spaced(_singular_category(c)) for c in plural_cats],
    )


def generate_corpus(
    world: WorldModel, n: int, *, seed: int = 0, category: str | None = None
) -> list[str]:
    """Sample *n* generator commands over *world*'s vocabulary.

    Deterministic for a given ``seed`` (the generator uses the global ``random``).
    ``category`` pins one bucket ("people"/"objects"); None alternates them so the
    corpus is balanced rather than 50/50-random. Skips the generator's "WARNING"
    sentinel (an uncovered template) and de-dups while preserving order.
    """
    gen = build_generator(world)
    random.seed(seed)
    out: list[str] = []
    seen: set[str] = set()
    for i in range(n):
        cat = category or CATEGORIES[i % len(CATEGORIES)]
        cmd = gen.generate_command_start(cmd_category=cat).strip()
        if not cmd or cmd == "WARNING" or cmd in seen:
            continue
        seen.add(cmd)
        out.append(cmd)
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Dump a GPSR command corpus from the official generator.")
    ap.add_argument("-n", type=int, default=100, help="number of commands to sample")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible)")
    ap.add_argument("--category", choices=CATEGORIES, default=None, help="pin one bucket")
    ap.add_argument("--world", default=None, help="world.toml path (default: GPSR_WORLD_FILE / bundled)")
    args = ap.parse_args()

    world = load_world(args.world, include_absent=True)
    for cmd in generate_corpus(world, args.n, seed=args.seed, category=args.category):
        print(cmd)


if __name__ == "__main__":
    _main()
