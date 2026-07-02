"""Object→room linking + room-aware query_text (offline; walkie_world is import-light).

Objects (map-seeded and perception-upserted) get a persistent ``node.room`` stamped
from room-polygon containment (nearest-room fallback when a room has no polygon), and
``query_text("kitchen table")`` hard-filters to that room.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from walkie_world import WalkieWorld
from walkie_world.scene.ingest import ObjectObservation
from walkie_world.scene.store import SceneStore


def _write(tmp_path, toml: str) -> str:
    p = tmp_path / "world.toml"
    p.write_text(textwrap.dedent(toml))
    return str(p)


# Two rooms with boundary polygons: kitchen (x in 0..6, y in -2..2), living_room (y 6..10).
_POLY_MAP = """
    [rooms]
    kitchen     = { pose = [3, 0, 0], polygon = [[0,-2],[6,-2],[6,2],[0,2]] }
    living_room = { pose = [3, 8, 0], aliases = ["living room"], polygon = [[0,6],[6,6],[6,10],[0,10]] }
    [object_categories]
    dishes = ["cup", "mug"]
"""

# Pose-only rooms (no polygon): kitchen waypoint at (2,0), living_room 10 m further.
# (Non-zero poses: [0,0,0] is the "unsurveyed" sentinel and is skipped for linking.)
_POSE_MAP = """
    [rooms]
    kitchen     = { pose = [2, 0, 0] }
    living_room = { pose = [12, 0, 0], aliases = ["living room"] }
"""


def _obs(cls: str, xy, *, n_obs: int = 5) -> ObjectObservation:
    x, y = xy
    pts = np.array([[x, y, 0.5], [x + 0.02, y, 0.5], [x, y + 0.02, 0.5]], dtype=np.float32)
    return ObjectObservation(
        class_name=cls, class_id=None, conf=0.9, captions=[cls], clip_emb=[],
        ts_first=1.0, ts_last=1.0, n_obs=n_obs, points=pts,
        centroid=(x, y, 0.5), extent=(0.1, 0.1, 0.1),
        aabb_min=(x - 0.05, y - 0.05, 0.45), aabb_max=(x + 0.05, y + 0.05, 0.55),
    )


def _world(tmp_path, toml, scene_dir=None):
    return WalkieWorld(
        map_path=_write(tmp_path, toml), enable_people=False,
        scene_dir=scene_dir or str(tmp_path / "scene"),
    )


def _one(world, cls):
    return next(n for n in world.all_objects() if n.class_name == cls)


def test_perception_object_stamped_by_polygon(tmp_path):
    w = _world(tmp_path, _POLY_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0)), _obs("mug", (3.0, 8.0))])
    assert _one(w, "cup").room == "kitchen"       # inside kitchen polygon
    assert _one(w, "mug").room == "living_room"   # inside living_room polygon


def test_nearest_room_fallback_without_polygon(tmp_path, monkeypatch):
    monkeypatch.setenv("WALKIE_ROOM_LINK_NEAREST", "1")
    monkeypatch.setenv("WALKIE_ROOM_LINK_MAX_M", "6.0")
    w = _world(tmp_path, _POSE_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0))])   # 1 m from kitchen pose (2,0), 9 m from living (12,0)
    assert _one(w, "cup").room == "kitchen"


def test_nearest_room_fallback_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("WALKIE_ROOM_LINK_NEAREST", "1")
    monkeypatch.setenv("WALKIE_ROOM_LINK_MAX_M", "0.5")   # object is 1 m away -> beyond cap
    w = _world(tmp_path, _POSE_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0))])
    assert _one(w, "cup").room is None


def test_nearest_room_fallback_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WALKIE_ROOM_LINK_NEAREST", "0")
    w = _world(tmp_path, _POSE_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0))])
    assert _one(w, "cup").room is None


def test_query_text_hard_filters_by_room(tmp_path):
    w = _world(tmp_path, _POLY_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0)), _obs("cup", (3.0, 8.0))])  # kitchen + living
    kitchen = w.query_text("kitchen cup")   # room parsed out of the query
    assert [n.room for n in kitchen] == ["kitchen"]
    living = w.query_text("cup in the living room")
    assert [n.room for n in living] == ["living_room"]
    assert len(w.query_text("cup")) == 2    # no room word -> unscoped


def test_objects_in_room_uses_stored_link(tmp_path):
    w = _world(tmp_path, _POLY_MAP)
    w.observe_objects([_obs("cup", (3.0, 0.0)), _obs("mug", (3.0, 8.0))])
    assert {n.class_name for n in w.objects_in_room("kitchen")} == {"cup"}
    assert {n.class_name for n in w.objects_in_room("living room")} == {"mug"}


def test_room_persists_and_reloads(tmp_path):
    scene_dir = str(tmp_path / "scene")
    w = _world(tmp_path, _POLY_MAP, scene_dir=scene_dir)
    w.observe_objects([_obs("cup", (3.0, 0.0))])
    w2 = _world(tmp_path, _POLY_MAP, scene_dir=scene_dir)  # fresh instance, same store
    assert _one(w2, "cup").room == "kitchen"


def test_node_from_dict_backward_compatible():
    # A pre-room nodes.json entry (no "room" key) loads with room=None, no error.
    d = {
        "id": "cup-1", "class_name": "cup", "class_id": None,
        "centroid": [1, 2, 0.5], "extent": [0.1, 0.1, 0.1],
        "aabb_min": [0.9, 1.9, 0.4], "aabb_max": [1.1, 2.1, 0.6],
        "clip_emb": [], "captions": ["cup"], "best_caption": "cup",
        "n_obs": 5, "conf": 0.9, "first_seen_ts": 1.0, "last_seen_ts": 1.0,
        "source": "perception",
    }
    node = SceneStore._node_from_dict(d, None)
    assert node.room is None and node.class_name == "cup"


def test_kitchen_room_suffix_alias(tmp_path):
    # A room keyed "kitchen_room" is reachable as "kitchen" (redundant-suffix alias).
    w = _world(tmp_path, """
        [rooms]
        kitchen_room = { pose = [3, 0, 0], polygon = [[0,-2],[6,-2],[6,2],[0,2]] }
    """)
    assert w.room("kitchen") == "kitchen_room"
    assert w.room("the kitchen") == "kitchen_room"
    w.observe_objects([_obs("cup", (3.0, 0.0))])
    assert [n.room for n in w.query_text("kitchen cup")] == ["kitchen_room"]
