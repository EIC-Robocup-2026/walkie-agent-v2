"""Offline tests for the generator-corpus oracle (tasks/GPSR/tools/gen_corpus.py).

No LLM, no robot — these check that the official CommandGenerator is wired over
OUR world vocabulary correctly, so the (LLM) coverage gate in
``test_gpsr_generator_coverage.py`` measures parser quality, not a vocabulary
mismatch. SKIPPED when the external CommandGenerator checkout isn't present
(point GPSR_GENERATOR_DIR at it); never fails for its absence.
"""

from __future__ import annotations

import pytest

from tasks.GPSR.world import load_world

gen_corpus = pytest.importorskip("tasks.GPSR.tools.gen_corpus")

try:
    gen_corpus.load_generator_cls()
except ImportError as exc:  # external repo not checked out on this machine
    pytest.skip(f"CommandGenerator not available: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def world():
    # Full arena vocabulary (present flag ignored) — what the generator draws from.
    return load_world(include_absent=True)


def test_singular_category_derivation():
    f = gen_corpus._singular_category
    assert f("drinks") == "drink"
    assert f("dishes") == "dish"
    assert f("cleaning_supplies") == "cleaning_supply"
    assert f("fruits") == "fruit"
    assert f("food") == "food"  # no plural form -> unchanged


def test_build_generator_vocab_is_fully_groundable(world):
    """Every noun handed to the generator must ground back through the world model,
    so a generated command references only resolvable nouns (vocabulary alignment)."""
    gen = gen_corpus.build_generator(world)

    for n in gen.person_names:
        assert world.name(n), f"name {n!r} does not ground"
    for r in gen.room_names:
        assert world.room(r), f"room {r!r} does not ground"
    for loc in gen.location_names:
        assert world.location(loc), f"location {loc!r} does not ground"
    for loc in gen.placement_location_names:
        assert world.location(loc), f"placement {loc!r} does not ground"
    for o in gen.object_names:
        assert world.obj(o), f"object {o!r} does not ground"
    for c in gen.object_categories_plural:
        assert world.category(c), f"plural category {c!r} does not ground"
    for c in gen.object_categories_singular:
        assert world.category(c), f"singular category {c!r} does not ground"


def test_placements_are_a_subset_of_locations(world):
    gen = gen_corpus.build_generator(world)
    assert set(gen.placement_location_names) <= set(gen.location_names)


def test_generate_corpus_is_nonempty_and_clean(world):
    corpus = gen_corpus.generate_corpus(world, 60, seed=0)
    assert len(corpus) >= 20  # de-dup may shrink it, but not to nothing
    assert all(isinstance(c, str) and c.strip() for c in corpus)
    assert "WARNING" not in corpus  # the generator's uncovered-template sentinel
    assert len(corpus) == len(set(corpus))  # de-duped


def test_generate_corpus_is_deterministic_for_a_seed(world):
    assert gen_corpus.generate_corpus(world, 30, seed=1) == gen_corpus.generate_corpus(world, 30, seed=1)


def test_category_pin_changes_the_mix(world):
    people = gen_corpus.generate_corpus(world, 30, seed=0, category="people")
    objects = gen_corpus.generate_corpus(world, 30, seed=0, category="objects")
    assert people and objects and people != objects
