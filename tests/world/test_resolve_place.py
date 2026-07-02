"""WalkieWorld.resolve_place — room-scoped, scene-aware place resolution for queries
like "go to the table in the kitchen" (offline; walkie_world is import-light)."""

from __future__ import annotations

import textwrap

import pytest

from walkie_world import WalkieWorld


@pytest.fixture()
def world(tmp_path):
    toml = textwrap.dedent(
        """
        [rooms]
        kitchen     = { pose = [3, 0, 0], polygon = [[0,-2],[6,-2],[6,2],[0,2]] }
        living_room = { pose = [3, 8, 0], aliases = ["living room", "lounge"], polygon = [[0,6],[6,6],[6,10],[0,10]] }

        [locations]
        kitchen_table     = { room = "kitchen",     placement = true, category = "dishes", pose = [2, 0, 0], aliases = ["kitchen table"] }
        cabinet           = { room = "kitchen",     placement = true, category = "drinks", pose = [4, 0, 3.14] }
        living_room_table = { room = "living_room", placement = true, pose = [2, 8, 0], aliases = ["living room table"] }
        sofa              = { room = "living_room", placement = true, pose = [4, 8, 0], aliases = ["couch"] }
        bookshelf         = { room = "living_room", placement = true, pose = [0, 0, 0] }  # unsurveyed

        [object_categories]
        dishes = ["cup", "plate"]
        drinks = ["cola", "milk"]
        """
    )
    p = tmp_path / "world.toml"
    p.write_text(toml)
    return WalkieWorld(map_path=p, scene_dir=str(tmp_path / "scene"), enable_people=False)


def test_direct_and_room_grounding(world):
    assert world.resolve_place("the kitchen table").name == "kitchen_table"
    assert world.resolve_place("cabinet").name == "cabinet"
    m = world.resolve_place("the kitchen")
    assert m.kind == "room" and m.name == "kitchen"


def test_room_scoped_disambiguation(world):
    # A bare "table" is ambiguous; scoping by room (arg or "... in the <room>") resolves it.
    assert world.resolve_place("table", room="kitchen").name == "kitchen_table"
    assert world.resolve_place("the table in the kitchen").name == "kitchen_table"
    assert world.resolve_place("table", room="living room").name == "living_room_table"
    assert world.resolve_place("the table in the living room").name == "living_room_table"


def test_alias_and_category_in_room(world):
    assert world.resolve_place("couch", room="living room").name == "sofa"      # alias
    assert world.resolve_place("drinks", room="kitchen").name == "cabinet"      # category


def test_nearest_pick_and_candidates(world):
    # Bare "table" with no room: pick the nearest to `near`, list the rest.
    near_kitchen = world.resolve_place("table", near=(2.0, 0.0))
    assert near_kitchen.name == "kitchen_table"
    assert "living room table" in near_kitchen.candidates
    near_living = world.resolve_place("table", near=(2.0, 8.0))
    assert near_living.name == "living_room_table"


def test_leading_verb_stripped(world):
    assert world.resolve_place("go to the cabinet").name == "cabinet"
    assert world.resolve_place("navigate to the table in the kitchen").name == "kitchen_table"


def test_unknown_place_returns_none(world):
    assert world.resolve_place("the helicopter") is None
    assert world.resolve_place("") is None


def test_scene_fallback(world, monkeypatch):
    # No map location for "plant"; fall back to an observed scene object's point.
    class _Node:
        id = "obj-9"
        class_name = "potted plant"
        best_caption = "a potted plant"
        centroid = (1.5, 8.0, 0.6)

    monkeypatch.setattr(world, "query_text", lambda q, k=5, **kw: [_Node()])
    m = world.resolve_place("plant", room="living room")
    assert m.kind == "object" and m.source == "scene"
    assert m.pose is None and m.point == (1.5, 8.0)


def test_waypoint_for(world):
    # A map-seeded node (id "map:<location>") resolves to the location's surveyed
    # waypoint — NOT its footprint centroid. A perceived object (no "map" source) has none.
    class _MapNode:
        id = "map:cabinet"
        source = "map"
        centroid = (4.5, 0.5, 0.7)  # footprint centre; deliberately != the waypoint

    class _Perceived:
        id = "obj-9"
        source = "perception"
        centroid = (1.5, 8.0, 0.6)

    assert world.waypoint_for(_MapNode()) == (4, 0, 3.14)  # cabinet's [locations] pose
    assert world.waypoint_for(_Perceived()) is None
    assert world.waypoint_for(None) is None


def test_scene_fallback_map_hit_promoted_to_waypoint(world, monkeypatch):
    # A CLIP scene hit that is actually a map-seeded location must navigate to its
    # approach waypoint (kind "location"), never its footprint centroid.
    class _MapNode:
        id = "map:cabinet"
        source = "map"
        class_name = "cabinet"
        best_caption = "a cabinet"
        centroid = (4.5, 0.5, 0.7)

    monkeypatch.setattr(world, "query_text", lambda q, k=5, **kw: [_MapNode()])
    m = world.resolve_place("fridge", room="kitchen")  # "fridge" isn't a map alias -> scene path
    assert m.kind == "location" and m.source == "map" and m.name == "cabinet"
    assert m.pose == (4, 0, 3.14) and m.point is None  # waypoint, not centroid


def test_scene_fallback_disabled(world, monkeypatch):
    monkeypatch.setenv("WALKIE_RESOLVE_SCENE_FALLBACK", "0")
    monkeypatch.setattr(world, "query_text", lambda *a, **k: [object()])
    assert world.resolve_place("plant", room="living room") is None


def test_unsurveyed_location_not_returned(world, monkeypatch):
    # bookshelf has a [0,0,0] pose -> not a usable destination; with no scene hit -> None.
    monkeypatch.setattr(world, "query_text", lambda *a, **k: [])
    assert world.resolve_place("bookshelf", room="living room") is None


def test_query_text_in_room_scopes_by_room(world, monkeypatch):
    seen = {}

    def fake_query(q, k=5, *, near=None, radius=None, room=None, scope_by_room=True):
        seen["room"], seen["q"] = room, q
        return []

    monkeypatch.setattr(world, "query_text", fake_query)
    world.query_text_in_room("cup", "kitchen")
    # Scopes by the stored object→room link (grounded canonical room), not geometry.
    assert seen["room"] == "kitchen" and seen["q"] == "cup"
    assert world.query_text_in_room("cup", "nowhere") == []
