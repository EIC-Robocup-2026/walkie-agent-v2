"""GPSR parser coverage gate (the Phase-0 acceptance test, docs/GPSR_DESIGN.md §10).

Feeds a corpus covering every CommandGenerator category through the LLM parser and
measures the fraction that ground to a COMPLETE typed plan with no Tier-2 fallback.
That number is the confidence in the draw-independent 540 (understand + speak-plan)
— there is no on-robot test for parse quality, so this offline-against-OpenRouter
run is the gate.

Needs the model (OpenRouter), NOT the robot — it runs on the dev box. It is
SKIPPED unless `OPENROUTER_API_KEY` is set (so CI / no-key machines don't fail);
it makes ~35 LLM calls. Run it directly to see the per-command breakdown:

    uv run pytest tests/test_gpsr_coverage.py -s
    GPSR_COVERAGE_MIN=0.9 uv run pytest tests/test_gpsr_coverage.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tasks.GPSR.parse import parse_command
from tasks.GPSR.world import load_world

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="GPSR coverage gate needs OPENROUTER_API_KEY (LLM parser); skipped offline",
)

# Default acceptance bar; override with GPSR_COVERAGE_MIN.
COVERAGE_MIN = float(os.getenv("GPSR_COVERAGE_MIN", "0.85"))


def _load_corpus() -> list[str]:
    text = (Path(__file__).with_name("gpsr_command_corpus.txt")).read_text()
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and line.split(":", 1)[0].replace("_", "").isalpha() and " " not in line.split(":", 1)[0]:
            line = line.split(":", 1)[1].strip()  # drop the "<category>: " prefix
        out.append(line)
    return out


def _build_model():
    """A ChatOpenAI on OpenRouter, built standalone (no tasks.common / no hardware)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("WALKIE_MODEL", "anthropic/claude-sonnet-4.5"),
        temperature=0,
    )


def test_parser_coverage():
    world = load_world()
    model = _build_model()
    corpus = _load_corpus()
    assert corpus, "empty corpus"

    complete = 0
    failures: list[str] = []
    for cmd in corpus:
        plan = parse_command(model, cmd, world)
        if plan.is_complete:
            complete += 1
        else:
            gaps = [u for s in plan.steps for u in s.unresolved] or ["no steps"]
            failures.append(f"  ✗ {cmd!r}\n      steps={[s.primitive.value for s in plan.steps]} gaps={gaps}")

    coverage = complete / len(corpus)
    print(f"\nGPSR parser coverage: {complete}/{len(corpus)} = {coverage:.0%} (min {COVERAGE_MIN:.0%})")
    if failures:
        print("Incomplete:\n" + "\n".join(failures))
    assert coverage >= COVERAGE_MIN, f"coverage {coverage:.0%} below gate {COVERAGE_MIN:.0%}"
