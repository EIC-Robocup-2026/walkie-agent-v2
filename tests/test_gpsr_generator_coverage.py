"""GPSR parser coverage against the OFFICIAL generator (the §10.3 oracle).

Where ``test_gpsr_coverage.py`` runs a hand-written line per category (category
coverage), this samples the real ``CommandGenerator`` over our arena vocabulary
and measures coverage on the actual command *distribution* — the strongest offline
read on the draw-independent 540 (understand + speak-a-plan).

Coverage = fraction of sampled commands that parse to a COMPLETE typed plan (every
step grounded, no Tier-2 fallback). The generator includes genuinely hard tails
(two-person relays, "answer a quiz") that our two-tier design routes to Tier-2 *by
design*, so the bar here is a **regression floor**, not 100% — the real value is
the printed per-command breakdown of what misses. Run it to see that breakdown:

    uv run pytest tests/test_gpsr_generator_coverage.py -s
    GPSR_GENERATOR_CORPUS_N=120 uv run pytest tests/test_gpsr_generator_coverage.py -s

Doubly gated: needs OPENROUTER_API_KEY (the LLM parser) AND the external
CommandGenerator checkout (GPSR_GENERATOR_DIR); skipped if either is missing so CI
/ teammate machines don't fail.
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
# gate measures the SAME model the robot runs (config [llm] WALKIE_MODEL).
load_dotenv()
load_config()


def _render_defect(cmd: str, plan) -> str | None:
    """A COMPLETE plan must also render a clean spoken plan (the scored 300). A
    degenerate render (empty, the can't-understand fallback, or a leaked raw
    primitive token — which still carries an underscore where healthy phrases space
    names) loses it silently even on a good parse. Returns a defect, or None."""
    speech = render_plan_speech(plan)
    if not speech or "could not work out a plan" in speech:
        return f"  ✗ {cmd!r} -> empty/degenerate render: {speech!r}"
    if "_" in speech:
        return f"  ✗ {cmd!r} -> render leaks a raw token: {speech!r}"
    return None

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="generator coverage gate needs OPENROUTER_API_KEY (LLM parser); skipped offline",
)

gen_corpus = pytest.importorskip("tasks.GPSR.tools.gen_corpus")

# How many commands to sample (kept modest for pytest wall-clock; raise via env for
# a deep sweep). The CLI tool dumps an arbitrarily large corpus for manual review.
CORPUS_N = int(os.getenv("GPSR_GENERATOR_CORPUS_N", "40"))
# Regression floor. Measured ~98% (39/40, seed 0, N=40) on claude-sonnet-4.5 — the
# lone miss is a beacon-scoped "meet at the exit" (a known grounding gap, not noise).
# On google/gemini-3-flash-preview:nitro (2026-07) it measures 82% — systematic
# misses: raw person descriptors ("someone waving") left unnormalized in `person`,
# and get_person_info/get_object_property `which` left empty / set to the
# superlative word. claude-haiku-4.5 (the GPSR_PARSER_MODEL this gate now measures
# via load_config) scored 100% (56/56) on the curated corpus (eval_llm_compare,
# 2026-07). The floor sits below the Sonnet number (matching the curated gate's
# 0.85) to absorb sampling / LLM variance and the generator's intentional Tier-2
# tail; tighten once triaged.
COVERAGE_MIN = float(os.getenv("GPSR_GENERATOR_COVERAGE_MIN", "0.85"))


def _build_model():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        # Mirrors the runtime parser: GPSR_PARSER_MODEL, else the shared model.
        model=os.getenv("GPSR_PARSER_MODEL")
        or os.getenv("WALKIE_MODEL", "google/gemini-3-flash-preview:nitro"),
        temperature=0,
    )


def test_generator_parser_coverage():
    try:
        gen_corpus.load_generator_cls()
    except ImportError as exc:
        pytest.skip(f"CommandGenerator not available: {exc}")

    world = load_world(WORLD_FIXTURE, include_absent=True)  # full arena vocabulary
    corpus = gen_corpus.generate_corpus(world, CORPUS_N, seed=0)
    assert corpus, "empty corpus"
    model = _build_model()

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
            failures.append(
                f"  ✗ {cmd!r}\n      steps={[s.primitive.value for s in plan.steps]} gaps={gaps}"
            )

    coverage = complete / len(corpus)
    print(f"\nGPSR generator coverage: {complete}/{len(corpus)} = {coverage:.0%} "
          f"(N={CORPUS_N}, floor {COVERAGE_MIN:.0%})")
    if failures:
        print("Incomplete (Tier-2 tail or real gaps):\n" + "\n".join(failures))
    # A complete parse that renders a broken plan loses the 300 silently — gate it
    # independently of the coverage floor.
    assert not render_defects, "spoken-plan render defects:\n" + "\n".join(render_defects)
    assert coverage >= COVERAGE_MIN, f"coverage {coverage:.0%} below floor {COVERAGE_MIN:.0%}"
