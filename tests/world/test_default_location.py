"""WalkieWorld.default_location_for / objects_in_room — the Finals "return a
misplaced object to its default location" + per-room recall lookups (offline)."""

from __future__ import annotations

import textwrap

import pytest

from walkie_world import WalkieWorld


@pytest.fixture()
def world(tmp_path):
    toml = textwrap.dedent(
        """
        [rooms]
        kitchen = { pose = [1,2,0], polygon = [[0,0],[4,0],[4,4],[0,4]] }
        office  = { pose = [9,9,0] }   # present, but no polygon and no category here

        [locations]
        kitchen_table = { room = "kitchen", placement = true, category = "dishes", pose = [1,1,0] }
        cabinet       = { room = "kitchen", placement = true, category = "drinks", pose = [3,1,0] }

        [object_categories]
        dishes = ["cup", "mug", "bowl"]
        drinks = ["cola", "milk"]
        toys   = ["dice"]
        """
    )
    p = tmp_path / "world.toml"
    p.write_text(toml)
    return WalkieWorld(
        map_path=p, scene_dir=str(tmp_path / "scene"), enable_people=False
    )


def test_default_location_by_object(world):
    assert world.default_location_for("cup") == ("kitchen_table", (1.0, 1.0, 0.0))
    assert world.default_location_for("cola") == ("cabinet", (3.0, 1.0, 0.0))
    assert world.default_location_for("milk") == ("cabinet", (3.0, 1.0, 0.0))


def test_default_location_by_category(world):
    assert world.default_location_for("dishes") == ("kitchen_table", (1.0, 1.0, 0.0))
    assert world.default_location_for("drinks") == ("cabinet", (3.0, 1.0, 0.0))


def test_default_location_misses(world):
    # 'dice' grounds to category 'toys', but no location holds toys -> None.
    assert world.default_location_for("dice") is None
    # Unknown noun grounds to nothing -> None (never raises).
    assert world.default_location_for("teleporter") is None
    assert world.default_location_for(None) is None


def test_objects_in_room_grounding_and_empty(world):
    # Unknown room -> []; a real room with nothing catalogued -> [].
    assert world.objects_in_room("nowhere") == []
    assert world.objects_in_room("kitchen") == []


def test_objects_in_room_filters_by_polygon(world):
    class _Node:
        def __init__(self, xy):
            self.centroid = (xy[0], xy[1], 0.5)

    inside, outside = _Node((2.0, 2.0)), _Node((20.0, 20.0))
    world.all_objects = lambda: [inside, outside]  # shadow for the test
    hits = world.objects_in_room("the kitchen")  # alias-tolerant grounding
    assert hits == [inside]
