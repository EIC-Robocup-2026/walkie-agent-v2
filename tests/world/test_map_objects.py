"""Map-object seeding + perception promotion in the scene store."""

from __future__ import annotations

import textwrap

import numpy as np

from walkie_world import WalkieWorld
from walkie_world.map.locations import MapObject
from walkie_world.scene.ingest import ObjectObservation
from walkie_world.scene.store import SceneStore, aabb_of


def _obs(class_name, center, *, n_obs=2):
    c = np.asarray(center, dtype=float)
    offs = (np.arange(60).reshape(20, 3) % 3 - 1) * 0.02
    pts = (c + offs).astype(np.float32)
    centroid, mn, mx, ext = aabb_of(pts)
    return ObjectObservation(
        class_name=class_name, class_id=None, conf=0.9, captions=[], clip_emb=[],
        ts_first=1.0, ts_last=1.0, n_obs=n_obs, points=pts,
        centroid=centroid, extent=ext, aabb_min=mn, aabb_max=mx,
    )


_WORLD = """
[rooms]
kitchen = { pose = [0,0,0], polygon = [[0,0],[4,0],[4,4],[0,4]] }

[object_instances]
"pringles.0" = { polygon = [[1.0,1.0],[1.2,1.0],[1.2,1.2],[1.0,1.2]], z = [0.7, 0.95], on = "kitchen_table" }
"""


def _world(tmp_path, name="scene"):
    p = tmp_path / "world.toml"
    p.write_text(textwrap.dedent(_WORLD))
    return WalkieWorld(map_path=p, scene_dir=str(tmp_path / name), enable_people=False), p


def test_map_object_seeded_as_placeholder(tmp_path):
    w, _ = _world(tmp_path)
    objs = w.all_objects()
    assert len(objs) == 1  # visible immediately despite n_obs == 0 (authoritative)
    node = objs[0]
    assert node.id == "map:pringles.0"
    assert node.source == "map"
    assert node.n_obs == 0
    assert node.points is None
    assert node.footprint_polygon  # the surveyed footprint is kept
    # Geometry from the XY bbox + Z extent.
    assert node.aabb_min == (1.0, 1.0, 0.7)
    assert node.aabb_max == (1.2, 1.2, 0.95)
    assert node.centroid[2] == 0.825


def test_seeding_is_idempotent(tmp_path):
    w, p = _world(tmp_path)
    # A second WalkieWorld over the SAME scene dir + map re-seeds without duplicating.
    w2 = WalkieWorld(map_path=p, scene_dir=str(tmp_path / "scene"), enable_people=False)
    map_nodes = [n for n in w2.all_objects() if n.source == "map"]
    assert len(map_nodes) == 1


def test_perception_promotes_placeholder(tmp_path):
    w, _ = _world(tmp_path)
    # A pringles detection inside the placeholder box -> promotion to a real cloud.
    w.observe_objects([_obs("pringles", (1.1, 1.1, 0.85), n_obs=2)])
    node = next(n for n in w.all_objects() if n.id == "map:pringles.0")
    assert node.source == "map"          # still authoritative
    assert node.n_obs > 0                # now perceived
    assert node.points is not None       # carries the real cloud
    # geometry now reflects the cloud (centroid near the detection, not the box centre)
    assert abs(node.centroid[0] - 1.1) < 0.1


def test_box_containment_promotes_outside_merge_dist():
    # A large placeholder box; a detection inside it but > merge_dist from its centre.
    store = SceneStore(store_dir=None, min_obs_confirm=2, merge_dist=0.5)
    big = MapObject(
        name="table_thing.0", class_name="bowl",
        footprint_polygon=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        z_min=0.0, z_max=0.5,
    )
    nodes = store.merge_map_objects([big], now=1.0)
    store.install(nodes, [])
    # centre is (1,1); detection at (1.8,1.8) is ~1.13 m away (> 0.5) but inside the box.
    nodes = store.merge([_obs("bowl", (1.8, 1.8, 0.25))], now=2.0)
    store.install(nodes, [])
    node = store.get("map:table_thing.0")
    assert node.n_obs > 0 and node.points is not None  # promoted via box containment


def test_map_node_never_pruned():
    store = SceneStore(store_dir=None, min_obs_confirm=1, prune_max_records=1)
    mo = MapObject(
        name="anchor.0", class_name="sofa",
        footprint_polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        z_min=0.0, z_max=0.8,
    )
    nodes = store.merge_map_objects([mo], now=1.0)
    store.install(nodes, [])
    # Add a far perception node of a different class; cap=1 must still keep the map node.
    nodes = store.merge([_obs("cup", (9.0, 9.0, 0.8))], now=2.0)
    store.install(nodes, [])
    ids = {n.id for n in store.all_objects()}
    assert "map:anchor.0" in ids
