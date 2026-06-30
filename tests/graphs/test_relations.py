"""Geometric relation derivation: near / on / above / inside golden cases.

Pure numpy — synthetic ObjectNodes with explicit AABBs/centroids. No cv2/open3d/PIL.
"""

from __future__ import annotations

import numpy as np

from walkie_world.scene.relations import derive_relations
from walkie_world.scene.store import ObjectNode, Relation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _node(
    nid: str,
    *,
    aabb_min,
    aabb_max,
    class_name="thing",
) -> ObjectNode:
    """Build an ObjectNode from an explicit AABB (centroid/extent derived)."""
    mn = np.asarray(aabb_min, dtype=float)
    mx = np.asarray(aabb_max, dtype=float)
    c = (mn + mx) / 2.0
    ext = mx - mn
    return ObjectNode(
        id=nid,
        class_name=class_name,
        class_id=1,
        centroid=tuple(float(x) for x in c),
        extent=tuple(float(x) for x in ext),
        aabb_min=tuple(float(x) for x in mn),
        aabb_max=tuple(float(x) for x in mx),
        clip_emb=[],
        captions=[],
        best_caption="",
        n_obs=1,
        conf=0.9,
        first_seen_ts=0.0,
        last_seen_ts=0.0,
    )


def _rels_between(rels, src, dst):
    """All predicates of edges src->dst (directed)."""
    return {r.predicate for r in rels if r.src_id == src and r.dst_id == dst}


def _find(rels, src, dst, predicate):
    for r in rels:
        if r.src_id == src and r.dst_id == dst and r.predicate == predicate:
            return r
    return None


# ---------------------------------------------------------------------------
# Vertical: on / above
# ---------------------------------------------------------------------------
def test_mug_on_table_is_on():
    # Table top at z=0.75; mug base sits right on it (zero gap), small footprint
    # fully within the table footprint -> overlap ratio ~1.0.
    table = _node("table", aabb_min=(-0.5, -0.5, 0.0), aabb_max=(0.5, 0.5, 0.75))
    mug = _node("mug", aabb_min=(0.0, 0.0, 0.75), aabb_max=(0.08, 0.08, 0.85))

    rels = derive_relations([table, mug])

    assert "on" in _rels_between(rels, "mug", "table")
    assert "above" not in _rels_between(rels, "mug", "table")


def test_lamp_above_table_is_above():
    # Lamp hangs well above the table top (gap 0.3 > on_gap default 0.08) with an
    # overlapping footprint -> "above", not "on".
    table = _node("table", aabb_min=(-0.5, -0.5, 0.0), aabb_max=(0.5, 0.5, 0.75))
    lamp = _node("lamp", aabb_min=(0.0, 0.0, 1.05), aabb_max=(0.12, 0.12, 1.25))

    rels = derive_relations([table, lamp])

    preds = _rels_between(rels, "lamp", "table")
    assert "above" in preds
    assert "on" not in preds


def test_small_object_offset_footprint_has_no_vertical():
    # Mug is above the table's top z-range but its XY footprint does not overlap
    # the table at all -> no on/above edge.
    table = _node("table", aabb_min=(-0.5, -0.5, 0.0), aabb_max=(0.5, 0.5, 0.75))
    mug = _node("mug", aabb_min=(2.0, 2.0, 0.75), aabb_max=(2.08, 2.08, 0.85))

    rels = derive_relations([table, mug], relation_max_dist=100.0)

    preds = _rels_between(rels, "mug", "table")
    assert "on" not in preds
    assert "above" not in preds


# ---------------------------------------------------------------------------
# Containment: inside
# ---------------------------------------------------------------------------
def test_object_inside_box_is_inside():
    box = _node("box", aabb_min=(0.0, 0.0, 0.0), aabb_max=(0.5, 0.5, 0.5))
    item = _node("item", aabb_min=(0.1, 0.1, 0.1), aabb_max=(0.3, 0.3, 0.3))

    rels = derive_relations([box, item])

    assert "inside" in _rels_between(rels, "item", "box")
    # Box is larger and not contained in the item -> no reverse containment.
    assert "inside" not in _rels_between(rels, "box", "item")


def test_equal_size_not_inside():
    # Identical AABBs: containment holds geometrically but neither is strictly
    # smaller, so _volume(x) < _volume(y) is False both ways -> no inside edge.
    a = _node("a", aabb_min=(0.0, 0.0, 0.0), aabb_max=(0.2, 0.2, 0.2))
    b = _node("b", aabb_min=(0.0, 0.0, 0.0), aabb_max=(0.2, 0.2, 0.2))

    rels = derive_relations([a, b])

    assert "inside" not in _rels_between(rels, "a", "b")
    assert "inside" not in _rels_between(rels, "b", "a")


# ---------------------------------------------------------------------------
# Proximity: near (with weight) and the max-distance gate
# ---------------------------------------------------------------------------
def test_nearby_objects_emit_near_with_rounded_weight():
    # Centroids 0.3 m apart, default near_m=0.6 -> weight = 1 - 0.3/0.6 = 0.5.
    a = _node("a", aabb_min=(-0.05, -0.05, -0.05), aabb_max=(0.05, 0.05, 0.05))
    b = _node("b", aabb_min=(0.25, -0.05, -0.05), aabb_max=(0.35, 0.05, 0.05))

    rels = derive_relations([a, b])

    edge = _find(rels, "a", "b", "near")
    assert edge is not None
    assert edge.weight == 0.5


def test_near_weight_rounded_to_three_dp():
    # d = 0.1, near_m = 0.6 -> 1 - 0.1/0.6 = 0.8333... -> rounded to 0.833.
    a = _node("a", aabb_min=(-0.01, -0.01, -0.01), aabb_max=(0.01, 0.01, 0.01))
    b = _node("b", aabb_min=(0.09, -0.01, -0.01), aabb_max=(0.11, 0.01, 0.01))

    rels = derive_relations([a, b])

    edge = _find(rels, "a", "b", "near")
    assert edge is not None
    assert edge.weight == 0.833


def test_objects_beyond_relation_max_dist_have_no_relation():
    a = _node("a", aabb_min=(-0.05, -0.05, -0.05), aabb_max=(0.05, 0.05, 0.05))
    b = _node("b", aabb_min=(4.95, -0.05, -0.05), aabb_max=(5.05, 0.05, 0.05))

    # Default relation_max_dist=1.0, centroids ~5 m apart -> nothing.
    rels = derive_relations([a, b])

    assert rels == []


def test_between_near_and_max_dist_no_near_but_pair_considered():
    # 0.8 m apart: beyond near_m (0.6) but within relation_max_dist (1.0).
    # No "near" edge, and no vertical/containment either (disjoint stacked? no).
    a = _node("a", aabb_min=(-0.05, -0.05, -0.05), aabb_max=(0.05, 0.05, 0.05))
    b = _node("b", aabb_min=(0.75, -0.05, -0.05), aabb_max=(0.85, 0.05, 0.05))

    rels = derive_relations([a, b])

    assert _find(rels, "a", "b", "near") is None
    assert _find(rels, "b", "a", "near") is None


# ---------------------------------------------------------------------------
# Sanity: empty / singleton inputs
# ---------------------------------------------------------------------------
def test_empty_and_singleton_inputs():
    assert derive_relations([]) == []
    only = _node("a", aabb_min=(0.0, 0.0, 0.0), aabb_max=(0.1, 0.1, 0.1))
    assert derive_relations([only]) == []


def test_returns_relation_instances():
    table = _node("table", aabb_min=(-0.5, -0.5, 0.0), aabb_max=(0.5, 0.5, 0.75))
    mug = _node("mug", aabb_min=(0.0, 0.0, 0.75), aabb_max=(0.08, 0.08, 0.85))

    rels = derive_relations([table, mug])

    assert rels
    assert all(isinstance(r, Relation) for r in rels)
