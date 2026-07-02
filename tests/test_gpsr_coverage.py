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
from tasks.GPSR.plan import render_plan_speech
from walkie_config import load_config
from walkie_world.map.vocab import load_world

# The frozen, vocab-complete CompetitionTemplate arena (see tests/fixtures/).
# The repo-root world.toml is the LIVE surveyed arena — no grammar vocab in it.
WORLD_FIXTURE = Path(__file__).parent / "fixtures" / "world.competition_template.toml"

# Same env precedence as every runtime entrypoint (.env > config.toml), so the
# gate measures the SAME model the robot runs (config [llm] WALKIE_MODEL) —
# without this the gate silently graded the in-code fallback model instead.
load_dotenv()
load_config()


def _render_defect(cmd: str, plan) -> str | None:
    """A COMPLETE plan must also render a clean spoken plan — that render is the
    scored 300 ("demonstrate a plan"). A degenerate render (empty, the
    can't-understand fallback, or a leaked raw primitive token, which still carries
    an underscore where every healthy phrase spaces names) silently forfeits it
    even when the parse succeeded. Return a defect string, or None when sane."""
    speech = render_plan_speech(plan)
    if not speech or "could not work out a plan" in speech:
        return f"  ✗ {cmd!r} -> empty/degenerate render: {speech!r}"
    if "_" in speech:
        return f"  ✗ {cmd!r} -> render leaks a raw token: {speech!r}"
    return None

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
    """A ChatOpenAI on OpenRouter, built standalone (no tasks.common / no hardware).

    Model resolution mirrors the runtime parser: GPSR_PARSER_MODEL (the dedicated
    parser model), else the shared WALKIE_MODEL — so this gate always measures the
    model the parser actually runs."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("GPSR_PARSER_MODEL")
        or os.getenv("WALKIE_MODEL", "google/gemini-3-flash-preview:nitro"),
        temperature=0,
    )


def test_parser_coverage():
    # Full CompetitionTemplate vocabulary: the corpus references the whole arena
    # (kitchen, cabinet, desk, sofa…), most of which is present=false in the
    # practice world.toml. Coverage measures parser/grounding quality over the
    # grammar, not which places are physically surveyed — so load the full vocab
    # (else every present=false noun reads as an ungrounded "miss"). Mirrors the
    # test_gpsr_parse fixture; the present-flag drop is tested separately there.
    world = load_world(WORLD_FIXTURE, include_absent=True)
    model = _build_model()
    corpus = _load_corpus()
    assert corpus, "empty corpus"

    complete = 0
    failures: list[str] = []
    render_defects: list[str] = []
    for cmd in corpus:
        plan = parse_command(model, cmd, world)
        if plan.is_complete:
            complete += 1
            if defect := _render_defect(cmd, plan):
                render_defects.append(defect)
        else:
            gaps = [u for s in plan.steps for u in s.unresolved] or ["no steps"]
            failures.append(f"  ✗ {cmd!r}\n      steps={[s.primitive.value for s in plan.steps]} gaps={gaps}")

    coverage = complete / len(corpus)
    print(f"\nGPSR parser coverage: {complete}/{len(corpus)} = {coverage:.0%} (min {COVERAGE_MIN:.0%})")
    if failures:
        print("Incomplete:\n" + "\n".join(failures))
    # A complete parse that renders a broken plan loses the 300 silently — gate it
    # independently of the coverage threshold.
    assert not render_defects, "spoken-plan render defects:\n" + "\n".join(render_defects)
    assert coverage >= COVERAGE_MIN, f"coverage {coverage:.0%} below gate {COVERAGE_MIN:.0%}"
