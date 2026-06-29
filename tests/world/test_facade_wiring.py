"""WalkieWorld facade wiring: ctx.world exposes map, polygon, scene and vocab."""

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

        [locations]
        kitchen_table = { room = "kitchen", placement = true, category = "dishes", pose = [1,1,0] }

        [object_categories]
        dishes = ["cup", "mug", "bowl"]
        drinks = ["cola", "milk"]

        [doors]
        kitchen_door = { pose = [4,2,0], radius = 1.0 }
        """
    )
    p = tmp_path / "world.toml"
    p.write_text(toml)
    return WalkieWorld(
        map_path=p, scene_dir=str(tmp_path / "scene"), enable_people=False
    )


def test_vocab_grounding(world):
    assert world.room("the kitchen") == "kitchen"
    assert world.location("kitchen table") == "kitchen_table"
    assert world.obj("cups") == "cup"          # plural tolerated
    assert world.obj("a drink") == "cola"      # category -> first member
    assert world.category("dishes") == "dishes"
    assert "cup" in world.categories["dishes"]
    assert "Rooms:" in world.vocab_prompt()


def test_map_pose_and_room_at(world):
    assert world.location_pose("kitchen_table") == (1.0, 1.0, 0.0)
    assert world.room_at(2, 2) == "kitchen"
    assert world.room_at(20, 20) is None
    assert world.pose("kitchen") == (1.0, 2.0, 0.0)


def test_is_near_door(world):
    assert world.has_doors() is True
    assert world.is_near_door(4.0, 2.0) is True       # at the door
    assert world.is_near_door(4.5, 2.0, radius=1.0) is True
    assert world.is_near_door(20.0, 20.0) is False


def test_scene_queries_empty(world):
    assert world.count() == 0
    assert world.all_objects() == []
    assert world.query_text("anything") == []


def test_people_disabled_raises(world):
    assert world.people_count() == 0
    with pytest.raises(RuntimeError):
        _ = world.people
