"""Auto-seeding the map's [locations] as query_text-able scene nodes (flag-gated).

The default-off ``seed_locations`` knob promotes surveyed furniture ([locations] in
world.toml) into ``source="map"`` scene nodes, so ``query_text`` / ``query_near`` can
return world.toml entries — not just perception-observed objects. Pure numpy (no
server): a small word-sensitive ``embed_text`` (``_embed``) stands in for CLIP so the
cosine path is exercised. Map nodes now carry a text embedding of their de-numbered
name, so ``query_text`` ranks them by meaning (``table`` over the nearer ``sink``),
not only the keyword union.
"""

from __future__ import annotations

import textwrap

from walkie_world import WalkieWorld

# table near (6.4, 2.0); sink near (2.0, 2.0); a third location marked present=false.
_WORLD = """
[rooms]
kitchen = { pose = [0,0,0], polygon = [[0,0],[8,0],[8,8],[0,8]] }

[locations]
table = { room = "kitchen", pose = [6.4,2.0,0.0], polygon = [[6.0,1.7],[6.8,1.7],[6.8,2.3],[6.0,2.3]] }
sink  = { room = "kitchen", pose = [2.0,2.0,0.0], polygon = [[1.6,1.7],[2.4,1.7],[2.4,2.3],[1.6,2.3]] }
oven  = { room = "kitchen", pose = [4.0,2.0,0.0], polygon = [[3.6,1.7],[4.4,1.7],[4.4,2.3],[3.6,2.3]], present = false }
"""


def _world(tmp_path, *, seed_locations, embed_text=None, name="scene"):
    p = tmp_path / "world.toml"
    p.write_text(textwrap.dedent(_WORLD))
    return WalkieWorld(
        map_path=p,
        scene_dir=str(tmp_path / name),
        enable_people=False,
        seed_locations=seed_locations,
        embed_text=embed_text,
    )


# Word-sensitive fake CLIP text embedder: a bag-of-words vector over the location
# vocab, so 'table' embeds near map:table and away from map:sink (a constant vector
# can't tell them apart). One shared space for the query text and the seeded names.
_VOCAB = ("table", "sink", "oven")


def _embed(text):
    t = text.lower().split()
    return [1.0 if w in t else 0.0 for w in _VOCAB]


def test_locations_not_seeded_by_default(tmp_path):
    # Flag off + no [object_instances] → the scene is empty.
    w = _world(tmp_path, seed_locations=False)
    assert w.all_objects() == []


def test_locations_seeded_when_enabled(tmp_path):
    w = _world(tmp_path, seed_locations=True)
    ids = {n.id for n in w.all_objects()}
    assert ids == {"map:table", "map:sink"}  # 'oven' is present=false → dropped
    table = w.get("map:table")
    assert table.source == "map"
    assert table.class_name == "table"
    assert table.n_obs == 0  # never perceived, still authoritative/visible
    # centroid from the XY footprint; flat at z=0 (a Location has no Z extent).
    assert abs(table.centroid[0] - 6.4) < 1e-6
    assert abs(table.centroid[1] - 2.0) < 1e-6
    assert table.centroid[2] == 0.0


def test_present_false_location_not_seeded(tmp_path):
    w = _world(tmp_path, seed_locations=True)
    assert w.get("map:oven") is None  # present=false never enters the scene


def test_query_text_returns_seeded_location(tmp_path):
    # Map nodes embed their (de-numbered) name, so the cosine path returns the seeded
    # table directly; the tight radius keeps the far sink out regardless.
    w = _world(tmp_path, seed_locations=True, embed_text=_embed)
    hits = w.query_text("table", near=[6.4, 2.0], radius=1.0)
    assert [h.id for h in hits] == ["map:table"]


def test_query_text_respects_text_over_proximity(tmp_path):
    # Querying 'table' from RIGHT AT the sink must rank the (farther) table FIRST, not the
    # nearer sink — proves the seeded query honours the word (name embedding), not radius.
    w = _world(tmp_path, seed_locations=True, embed_text=_embed)
    hits = w.query_text("table", near=[2.0, 2.0], radius=10.0)
    assert hits[0].id == "map:table"  # text beats proximity: named table outranks nearer sink


def test_seeded_location_promoted_by_perception(tmp_path):
    # A real detection at the table's footprint promotes the placeholder to a cloud,
    # exactly like an [object_instances] seed (source stays "map", n_obs grows).
    import numpy as np

    from walkie_world.scene.ingest import ObjectObservation
    from walkie_world.scene.store import aabb_of

    w = _world(tmp_path, seed_locations=True, embed_text=lambda q: [1.0] * 512)
    c = np.array([6.4, 2.0, 0.4])
    pts = (c + (np.arange(60).reshape(20, 3) % 3 - 1) * 0.02).astype(np.float32)
    centroid, mn, mx, ext = aabb_of(pts)
    w.observe_objects([
        ObjectObservation(
            class_name="table", class_id=None, conf=0.9, captions=[], clip_emb=[],
            ts_first=1.0, ts_last=1.0, n_obs=2, points=pts,
            centroid=centroid, extent=ext, aabb_min=mn, aabb_max=mx,
        )
    ])
    node = w.get("map:table")
    assert node.source == "map" and node.n_obs > 0 and node.points is not None
