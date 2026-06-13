"""Geometric (distance-based) relation predicate tests."""

from __future__ import annotations

from tests.graphs.conftest import put_box


def _preds(rels):
    return {(r.src_id, r.dst_id, r.predicate) for r in rels}


def test_mug_on_table(mem):
    put_box(mem, "table", "table", [0.0, 0.0, 0.0], [1.0, 1.0, 0.4])
    put_box(mem, "mug", "mug", [0.4, 0.4, 0.4], [0.5, 0.5, 0.5])
    preds = _preds(mem.derive_relations())
    assert ("mug", "table", "on") in preds
    assert ("table", "mug", "on") not in preds  # directed


def test_object_above_not_on(mem):
    put_box(mem, "table", "table", [0.0, 0.0, 0.0], [1.0, 1.0, 0.4])
    put_box(mem, "lamp", "lamp", [0.4, 0.4, 0.6], [0.5, 0.5, 0.7])  # 0.2 m gap
    preds = _preds(mem.derive_relations())
    assert ("lamp", "table", "above") in preds
    assert ("lamp", "table", "on") not in preds


def test_fork_inside_drawer(mem):
    put_box(mem, "drawer", "drawer", [0.0, 0.0, 0.0], [0.5, 0.4, 0.3])
    put_box(mem, "fork", "fork", [0.1, 0.1, 0.05], [0.3, 0.15, 0.08])
    preds = _preds(mem.derive_relations())
    assert ("fork", "drawer", "inside") in preds
    assert ("fork", "drawer", "on") not in preds


def test_near_pair(mem):
    put_box(mem, "a", "a", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    put_box(mem, "b", "b", [0.5, 0.0, 0.0], [0.6, 0.1, 0.1])  # 0.5 m apart, no overlap
    preds = _preds(mem.derive_relations())
    assert ("a", "b", "near") in preds
    assert not any(p in preds for p in [("a", "b", "on"), ("a", "b", "inside")])


def test_no_edge_when_beyond_max_dist(mem):
    put_box(mem, "a", "a", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    put_box(mem, "b", "b", [5.0, 0.0, 0.0], [5.1, 0.1, 0.1])
    assert mem.derive_relations() == []


def test_relations_of_sees_both_directions(mem):
    put_box(mem, "table", "table", [0.0, 0.0, 0.0], [1.0, 1.0, 0.4])
    put_box(mem, "mug", "mug", [0.4, 0.4, 0.4], [0.5, 0.5, 0.5])
    mem.derive_relations()
    mug_rels = {(r.src_id, r.dst_id, r.predicate) for r in mem.relations_of("mug")}
    table_rels = {(r.src_id, r.dst_id, r.predicate) for r in mem.relations_of("table")}
    assert ("mug", "table", "on") in mug_rels
    assert ("mug", "table", "on") in table_rels  # visible from the table side too
