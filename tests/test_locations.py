"""Offline tests for the shared named-location ("map") layer (tasks/skills/locations.py).

Covers the LocationBook loader (alias/article/fuzzy lookup, barrier, present-drop +
cascade-drop, missing-file→empty) and the resolve_pose fallback chain
(book → env var → literal default) that every challenge now drives through.
"""

from __future__ import annotations

import pytest

from walkie_world.map import locations as loc
from walkie_world.map.locations import LocationBook, load_location_book, resolve_pose


_MAP_TOML = """
[rooms]
kitchen     = { pose = [1.0, 2.0, 0.5] }
living_room = { pose = [5.0, -1.0, 1.57], aliases = ["lounge"], barrier = true }
office      = { pose = [9.0, 9.0, 0.0], present = false }

[locations]
dining_table   = { room = "kitchen", pose = [1.5, 2.5, 3.14], aliases = ["dinner table"] }
kitchen_bar    = { room = "kitchen", pose = [0.2, 0.3, -1.0] }
absent_shelf   = { room = "office", pose = [9.1, 9.1, 0.0] }

[doors]
entrance   = { pose = [0.0, 0.0, 1.57], radius = 2.0 }
hall_door  = { pose = [4.0, 4.0, 0.0] }
old_door   = { pose = [9.0, 9.0, 0.0], present = false }
"""


@pytest.fixture
def book(tmp_path):
    p = tmp_path / "map.toml"
    p.write_text(_MAP_TOML)
    return load_location_book(p)


# --- LocationBook lookups ---------------------------------------------------

def test_pose_exact_location_and_room(book):
    assert book.pose("dining_table") == (1.5, 2.5, 3.14)
    assert book.pose("kitchen") == (1.0, 2.0, 0.5)        # falls through to rooms


def test_lookup_is_alias_and_article_tolerant(book):
    assert book.pose("the dinner table") == (1.5, 2.5, 3.14)   # alias + article + spaces
    assert book.pose("lounge") == (5.0, -1.0, 1.57)            # room alias


def test_fuzzy_match_recovers_a_near_miss(book):
    # "dining tabel" isn't an exact alias but clears the difflib cutoff.
    assert book.pose("dining tabel") == (1.5, 2.5, 3.14)


def test_barrier_flag(book):
    assert book.is_barrier("living_room") is True
    assert book.is_barrier("lounge") is True       # via alias
    assert book.is_barrier("kitchen") is False


def test_has_and_names(book):
    assert book.has("kitchen_bar") is True
    assert book.has("nonexistent") is False
    # office is present=false -> dropped; absent_shelf cascade-drops with its room.
    assert "office" not in book.names()
    assert "absent_shelf" not in book.names()
    assert {"kitchen", "living_room", "dining_table", "kitchen_bar"} <= set(book.names())


def test_present_false_and_cascade_drop(book):
    assert book.pose("office") is None         # present=false room dropped
    assert book.pose("absent_shelf") is None   # its room was dropped -> cascade


def test_missing_file_returns_empty_book(tmp_path):
    empty = load_location_book(tmp_path / "nope.toml")
    assert isinstance(empty, LocationBook)
    assert empty.names() == []
    assert empty.pose("anything") is None
    assert empty.has_doors() is False
    assert empty.door_near(0.0, 0.0, default_radius=1.5) is None


# --- doors (proximity, not name lookup) -------------------------------------

def test_doors_parsed_with_present_drop(book):
    assert book.has_doors() is True
    assert set(book.doors) == {"entrance", "hall_door"}   # present=false old_door dropped
    assert book.doors["entrance"].pose == (0.0, 0.0, 1.57)
    assert book.doors["entrance"].radius == 2.0
    assert book.doors["hall_door"].radius is None          # no per-door radius


def test_door_near_uses_per_door_then_default_radius(book):
    # entrance has radius=2.0: a point 1.5 m away is inside its own circle...
    assert book.door_near(1.5, 0.0, default_radius=0.5).name == "entrance"
    # ...but hall_door (no radius) only triggers within the default radius (0.9 m
    # away: outside a 0.5 m default, inside a 1.5 m one).
    assert book.door_near(4.9, 4.0, default_radius=0.5) is None
    assert book.door_near(4.9, 4.0, default_radius=1.5).name == "hall_door"
    # far from every door -> nothing.
    assert book.door_near(100.0, 100.0, default_radius=1.5) is None


def test_nearest_door_returns_closest_with_distance(book):
    door, dist = book.nearest_door(4.0, 3.0)
    assert door.name == "hall_door" and dist == pytest.approx(1.0)
    assert book.nearest_door(0.1, 0.0)[0].name == "entrance"


# --- resolve_pose fallback chain --------------------------------------------

@pytest.fixture
def use_map(tmp_path, monkeypatch):
    """Point the cached book at our test map for resolve_pose."""
    p = tmp_path / "map.toml"
    p.write_text(_MAP_TOML)
    monkeypatch.setenv("WALKIE_MAP_FILE", str(p))
    loc._reset_cache()
    yield
    loc._reset_cache()


def test_resolve_prefers_the_book(use_map, monkeypatch):
    # Even with an env var set, the map (book) wins — it's the source of truth.
    monkeypatch.setenv("DINING_POSE", "99,99,9")
    assert resolve_pose("dining_table", env_fallback="DINING_POSE") == (1.5, 2.5, 3.14)


def test_resolve_falls_back_to_env_when_name_absent(use_map, monkeypatch):
    monkeypatch.setenv("MYSTERY_POSE", "7.0, 8.0, 1.0")   # whitespace tolerated
    assert resolve_pose("not_in_map", env_fallback="MYSTERY_POSE") == (7.0, 8.0, 1.0)


def test_resolve_falls_back_to_default_literal(use_map):
    assert resolve_pose("not_in_map", env_fallback="UNSET_POSE", default="3,4,0") == (3.0, 4.0, 0.0)


def test_resolve_with_no_map_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WALKIE_MAP_FILE", str(tmp_path / "nope.toml"))  # empty book
    monkeypatch.setenv("BAR_POSE", "2,2,0")
    loc._reset_cache()
    try:
        assert resolve_pose("kitchen_bar", env_fallback="BAR_POSE") == (2.0, 2.0, 0.0)
    finally:
        loc._reset_cache()
