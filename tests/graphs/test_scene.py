"""SceneStore: v1 query contract, merge-into-persisted, persistence, thread safety.

Pure numpy/scipy/sklearn — synthetic point clouds and a fake embed_text mapping a
few known strings to known unit vectors. No cv2/open3d/PIL/chromadb.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from services.walkie_graphs.scene import (
    BuiltScene,
    ObjectNode,
    Relation,
    SceneStore,
    aabb_of,
    cosine,
    l2,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------
@dataclass
class FakeObs:
    """Duck-typed ObjectObservation for SceneStore.merge."""

    class_name: str
    class_id: Optional[int]
    conf: float
    captions: list
    clip_emb: list
    ts_first: float
    ts_last: float
    n_obs: int
    points: Optional[np.ndarray]
    centroid: tuple
    extent: tuple
    aabb_min: tuple
    aabb_max: tuple


def _obs(
    class_name="mug",
    *,
    class_id=1,
    conf=0.9,
    captions=None,
    clip_emb=None,
    ts_first=100.0,
    ts_last=100.0,
    n_obs=1,
    points=None,
    centroid=(0.0, 0.0, 0.0),
    extent=(0.1, 0.1, 0.1),
    aabb_min=None,
    aabb_max=None,
):
    c = np.asarray(centroid, dtype=float)
    e = np.asarray(extent, dtype=float)
    aabb_min = aabb_min if aabb_min is not None else tuple(c - e / 2)
    aabb_max = aabb_max if aabb_max is not None else tuple(c + e / 2)
    return FakeObs(
        class_name=class_name,
        class_id=class_id,
        conf=conf,
        captions=captions if captions is not None else [],
        clip_emb=clip_emb if clip_emb is not None else [],
        ts_first=ts_first,
        ts_last=ts_last,
        n_obs=n_obs,
        points=points,
        centroid=tuple(float(x) for x in c),
        extent=tuple(float(x) for x in e),
        aabb_min=tuple(float(x) for x in aabb_min),
        aabb_max=tuple(float(x) for x in aabb_max),
    )


def _unit(*vals, dim=8):
    v = np.zeros(dim, dtype=float)
    for i, x in vals:
        v[i] = x
    n = np.linalg.norm(v)
    return (v / n).tolist()


# Known unit vectors: "red mug" points at axis 0, "blue chair" at axis 1.
_RED = _unit((0, 1.0))
_BLUE = _unit((1, 1.0))


def _fake_embed(mapping):
    def embed(q: str):
        return mapping.get(q)

    return embed


def _node(
    nid,
    class_name="mug",
    *,
    centroid=(0.0, 0.0, 0.0),
    clip_emb=None,
    captions=None,
    best_caption="",
    n_obs=5,
    last_seen_ts=100.0,
):
    c = np.asarray(centroid, dtype=float)
    ext = np.array([0.1, 0.1, 0.1])
    return ObjectNode(
        id=nid,
        class_name=class_name,
        class_id=1,
        centroid=tuple(float(x) for x in c),
        extent=tuple(float(x) for x in ext),
        aabb_min=tuple(float(x) for x in (c - ext / 2)),
        aabb_max=tuple(float(x) for x in (c + ext / 2)),
        clip_emb=clip_emb if clip_emb is not None else [],
        captions=captions if captions is not None else [],
        best_caption=best_caption,
        n_obs=n_obs,
        conf=0.9,
        first_seen_ts=last_seen_ts,
        last_seen_ts=last_seen_ts,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_aabb_of():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    c, mn, mx, ext = aabb_of(pts)
    assert c == (1.0, 2.0, 3.0)
    assert mn == (0.0, 0.0, 0.0)
    assert mx == (2.0, 4.0, 6.0)
    assert ext == (2.0, 4.0, 6.0)


def test_l2_shorter_common_length():
    # 2D center vs 3D centroid → compares only XY.
    assert l2((1.0, 0.0), (1.0, 0.0, 99.0)) == pytest.approx(0.0)
    assert l2((0.0, 0.0), (3.0, 4.0, 5.0)) == pytest.approx(5.0)


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([], [1, 0]) == 0.0
    assert cosine([0, 0], [1, 0]) == 0.0


# ---------------------------------------------------------------------------
# Cosine ranking
# ---------------------------------------------------------------------------
def test_cosine_ranking_returns_right_node():
    store = SceneStore(embed_dim=8, embed_text=_fake_embed({"red mug": _RED}))
    red = _node("red-1", clip_emb=_RED, centroid=(0, 0, 0))
    blue = _node("blue-1", "chair", clip_emb=_BLUE, centroid=(5, 0, 0))
    store.install([blue, red], [])
    hits = store.query_text("red mug", k=1)
    assert [n.id for n in hits] == ["red-1"]


def test_cosine_ranking_orders_by_similarity():
    store = SceneStore(embed_dim=8, embed_text=_fake_embed({"red mug": _RED}))
    # mid is a 45-degree blend; should rank between red and blue.
    mid = _unit((0, 1.0), (1, 1.0))
    red = _node("red-1", clip_emb=_RED, centroid=(0, 0, 0))
    blend = _node("mid-1", clip_emb=mid, centroid=(1, 0, 0))
    blue = _node("blue-1", clip_emb=_BLUE, centroid=(2, 0, 0))
    store.install([blue, blend, red], [])
    hits = store.query_text("red mug", k=3)
    assert [n.id for n in hits] == ["red-1", "mid-1", "blue-1"]


# ---------------------------------------------------------------------------
# Keyword fallback — all four triggers
# ---------------------------------------------------------------------------
def _kw_store(embed_text):
    store = SceneStore(embed_dim=8, embed_text=embed_text)
    n = _node("mug-1", "mug", clip_emb=_RED, captions=["a shiny red mug"], best_caption="a shiny red mug")
    store.install([n], [])
    return store


def test_keyword_fallback_embed_none():
    store = _kw_store(None)  # trigger 1: embed_text is None
    hits = store.query_text("red mug", k=3)
    assert [h.id for h in hits] == ["mug-1"]


def test_keyword_fallback_embed_raises():
    def boom(q):
        raise RuntimeError("embed server down")

    store = _kw_store(boom)  # trigger 2: embed_text raises
    hits = store.query_text("shiny mug", k=3)
    assert [h.id for h in hits] == ["mug-1"]


def test_keyword_fallback_embed_empty():
    store = _kw_store(_fake_embed({}))  # returns None → falsy; trigger 3
    hits = store.query_text("red", k=3)
    assert [h.id for h in hits] == ["mug-1"]
    store2 = _kw_store(lambda q: [])  # returns [] → falsy; trigger 3 variant
    assert [h.id for h in store2.query_text("red", k=3)] == ["mug-1"]


def test_keyword_fallback_vector_search_fails(monkeypatch):
    # trigger 4: embed returns a usable vector but the matmul path raises.
    store = _kw_store(_fake_embed({"red mug": _RED}))

    def explode(self, scene, query, k, *, near=None, radius=None):
        # Confirm the keyword path is actually reached, then return a sentinel.
        return [scene.nodes[0]]

    # Force the vector path to raise by stubbing the snapshot's emb to a bad shape.
    bad_scene = BuiltScene(
        nodes=store._snapshot().nodes,
        emb=np.zeros((1, 999), dtype=np.float32),  # width mismatch vs 8-dim query → raises
        id_index=store._snapshot().id_index,
        relations=[],
    )
    monkeypatch.setattr(store, "_snapshot", lambda: bad_scene)
    hits = store.query_text("red mug", k=3)
    assert [h.id for h in hits] == ["mug-1"]  # fell back to keyword and still found it


def test_keyword_no_match_returns_empty():
    store = _kw_store(None)
    assert store.query_text("zebra spaceship", k=3) == []


# ---------------------------------------------------------------------------
# Confirmation gate on every query method
# ---------------------------------------------------------------------------
def _store_with_provisional():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=3, require_confirmation=True)
    confirmed = _node("ok-1", "mug", n_obs=3, captions=["mug here"], best_caption="mug here",
                      centroid=(0, 0, 0), last_seen_ts=200.0)
    provisional = _node("ghost-1", "mug", n_obs=1, captions=["mug here too"],
                        best_caption="mug here too", centroid=(0.2, 0, 0), last_seen_ts=300.0)
    store.install([confirmed, provisional], [Relation("ok-1", "ghost-1", "near", 0.5)])
    return store


def test_confirmation_gate_query_text():
    store = _store_with_provisional()
    hits = store.query_text("mug", k=5)
    assert [h.id for h in hits] == ["ok-1"]


def test_confirmation_gate_query_near():
    store = _store_with_provisional()
    hits = store.query_near((0, 0), 5.0)
    assert [h.id for h in hits] == ["ok-1"]


def test_confirmation_gate_recently_seen():
    store = _store_with_provisional()
    # ghost-1 is newer but provisional → must be hidden.
    hits = store.recently_seen(5)
    assert [h.id for h in hits] == ["ok-1"]


def test_confirmation_gate_all_objects():
    store = _store_with_provisional()
    assert [h.id for h in store.all_objects()] == ["ok-1"]


def test_confirmation_gate_to_text_description():
    store = _store_with_provisional()
    txt = store.to_text_description()
    assert "Objects (1):" in txt
    assert "ok-1" in txt
    assert "ghost-1" not in txt
    # The relation references the hidden node, so it must not be shown.
    assert "Relations:" not in txt


def test_get_is_not_gated():
    store = _store_with_provisional()
    assert store.get("ghost-1") is not None  # get() bypasses the confirmation gate
    assert store.get("missing") is None


def test_require_confirmation_off_shows_all():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=3, require_confirmation=False)
    store.install([_node("a", n_obs=1, captions=["x"], best_caption="x")], [])
    assert len(store.all_objects()) == 1


def test_count_ignores_confirmation():
    store = _store_with_provisional()
    assert store.count() == 2  # count() sees everything, gated views see fewer


# ---------------------------------------------------------------------------
# query_near distance + sort
# ---------------------------------------------------------------------------
def test_query_near_distance_and_sort():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    near = _node("near", centroid=(0.5, 0.0, 0.0))
    far = _node("far", centroid=(0.9, 0.0, 0.0))
    outside = _node("outside", centroid=(3.0, 0.0, 0.0))
    store.install([far, near, outside], [])
    hits = store.query_near((0.0, 0.0), 1.0)
    assert [h.id for h in hits] == ["near", "far"]  # within radius, nearest first


# ---------------------------------------------------------------------------
# to_text_description format + relations
# ---------------------------------------------------------------------------
def test_to_text_description_format_and_relations():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    a = _node("table-1", "table", centroid=(0.0, 0.0, 0.0), best_caption="wooden table",
              n_obs=4, last_seen_ts=100.0)
    b = _node("mug-1", "mug", centroid=(0.1, 0.0, 0.5), best_caption="", n_obs=2, last_seen_ts=200.0)
    store.install([a, b], [Relation("mug-1", "table-1", "on", 1.0)])
    txt = store.to_text_description()
    lines = txt.splitlines()
    assert lines[0] == "Objects (2):"
    # newest first → mug-1 then table-1
    assert lines[1] == " [mug-1] mug at (0.10, 0.00, 0.50) seen 2x"  # no caption → no quotes
    assert lines[2] == ' [table-1] table "wooden table" at (0.00, 0.00, 0.00) seen 4x'
    assert "Relations:" in txt
    assert " mug [mug-1] on table [table-1]" in txt


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------
def test_persist_then_load_new_store(tmp_path):
    embed = _fake_embed({"red mug": _RED})
    store = SceneStore(store_dir=tmp_path, embed_dim=8, embed_text=embed, min_obs_confirm=1)
    n = _node("red-1", "mug", clip_emb=_RED, captions=["red mug"], best_caption="red mug",
              centroid=(1.0, 2.0, 0.0), n_obs=4)
    n.points = np.random.rand(50, 3).astype(np.float32)  # RAM-only, must NOT persist
    store.install([n], [Relation("red-1", "red-1", "near", 1.0)])

    assert (tmp_path / "nodes.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "edges.json").exists()

    # nodes.json must not contain a 'points' field.
    import json

    raw = json.loads((tmp_path / "nodes.json").read_text())
    assert "points" not in raw[0]

    fresh = SceneStore(store_dir=tmp_path, embed_dim=8, embed_text=embed, min_obs_confirm=1)
    assert fresh.count() == 1
    loaded = fresh.get("red-1")
    assert loaded is not None
    assert loaded.points is None  # clouds are not reloaded
    assert loaded.best_caption == "red mug"
    # Cosine query still works on the freshly loaded store (no clouds needed).
    hits = fresh.query_text("red mug", k=1)
    assert [h.id for h in hits] == ["red-1"]


def test_load_empty_dir_is_noop(tmp_path):
    store = SceneStore(store_dir=tmp_path, embed_dim=8)
    assert store.count() == 0
    assert store.all_objects() == []


# ---------------------------------------------------------------------------
# merge: never shrink, update on match, insert on miss
# ---------------------------------------------------------------------------
def test_merge_updates_on_match():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    base = _node("mug-1", "mug", centroid=(0.0, 0.0, 0.0), n_obs=2, captions=["mug"],
                 best_caption="mug", clip_emb=_RED, last_seen_ts=100.0)
    base.first_seen_ts = 100.0
    store.install([base], [])

    obs = _obs(
        "mug",
        centroid=(0.1, 0.0, 0.0),  # within 0.5 m, same class → match
        captions=["a small ceramic mug"],
        clip_emb=_RED,
        ts_first=50.0,
        ts_last=300.0,
        n_obs=2,
    )
    merged = store.merge([obs], now=300.0)
    assert len(merged) == 1  # never grew the count on a match
    node = merged[0]
    assert node.id == "mug-1"
    assert node.n_obs == 4  # 2 + 2
    assert node.last_seen_ts == 300.0
    assert node.first_seen_ts == 50.0  # min
    assert "a small ceramic mug" in node.captions
    assert "mug" in node.captions
    assert node.best_caption == "a small ceramic mug"  # longest non-empty


def test_merge_inserts_on_non_match():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    base = _node("mug-1", "mug", centroid=(0.0, 0.0, 0.0), n_obs=2)
    store.install([base], [])

    far = _obs("mug", centroid=(5.0, 0.0, 0.0))  # same class but too far → insert
    diff = _obs("chair", centroid=(0.0, 0.0, 0.0))  # near but different class → insert
    merged = store.merge([far, diff], now=300.0)
    assert len(merged) == 3  # never shrank, added two
    ids = {n.id for n in merged}
    assert "mug-1" in ids


def test_merge_never_shrinks_with_empty_observations():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    store.install([_node("a"), _node("b", centroid=(5, 0, 0))], [])
    merged = store.merge([], now=300.0)
    assert len(merged) == 2


def test_merge_then_install_roundtrip():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    store.install([_node("mug-1", centroid=(0, 0, 0), n_obs=1)], [])
    obs = _obs("mug", centroid=(0.05, 0, 0), n_obs=1)
    merged = store.merge([obs], now=300.0)
    store.install(merged, [])
    assert store.count() == 1
    assert store.get("mug-1").n_obs == 2


def test_merge_points_union_recomputes_geometry():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    base = _node("mug-1", "mug", centroid=(0.0, 0.0, 0.0), n_obs=1)
    base.points = np.zeros((20, 3), dtype=np.float32)  # tight cluster at origin
    store.install([base], [])
    # A second partial view, within merge_dist (0.5 m) so it matches, that extends
    # the cloud out to (0.2, 0.0, 0.0) — union should fill the box and shift geometry.
    pts = np.zeros((20, 3), dtype=np.float32)
    pts[:, 0] = 0.2
    obs = _obs("mug", centroid=(0.2, 0.0, 0.0), points=pts, n_obs=1)
    merged = store.merge([obs], now=300.0)
    assert len(merged) == 1  # matched & merged, not inserted
    node = merged[0]
    assert node.points is not None
    # Recomputed centroid lies between the two clusters (voxel grid-mean of the union).
    assert 0.0 < node.centroid[0] < 0.2
    assert node.aabb_min == pytest.approx((0.0, 0.0, 0.0))
    assert node.aabb_max == pytest.approx((0.2, 0.0, 0.0))


def test_merge_prunes_to_cap_by_recency():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1, prune_max_records=2)
    store.install(
        [
            _node("old", "mug", centroid=(0, 0, 0), last_seen_ts=1.0),
            _node("mid", "mug", centroid=(5, 0, 0), last_seen_ts=2.0),
        ],
        [],
    )
    fresh = _obs("newcls", centroid=(10, 0, 0), ts_last=99.0)  # new class → inserts
    merged = store.merge([fresh], now=99.0)
    ids = {n.id for n in merged}
    assert len(merged) == 2  # capped
    assert "old" not in ids  # oldest dropped
    assert "mid" in ids


# ---------------------------------------------------------------------------
# Embedding normalization after merge
# ---------------------------------------------------------------------------
def test_merge_renormalizes_embedding():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    base = _node("mug-1", "mug", centroid=(0, 0, 0), clip_emb=_RED, n_obs=3)
    store.install([base], [])
    obs = _obs("mug", centroid=(0.1, 0, 0), clip_emb=_BLUE, n_obs=1)
    merged = store.merge([obs], now=300.0)
    emb = np.asarray(merged[0].clip_emb, dtype=float)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-6)  # re-normalized
    # weighted toward red (weight 3 vs 1)
    assert emb[0] > emb[1]


# ---------------------------------------------------------------------------
# Concurrency smoke: install() while a query iterates must not crash
# ---------------------------------------------------------------------------
def test_concurrent_install_during_query_does_not_crash():
    store = SceneStore(embed_dim=8, embed_text=None, min_obs_confirm=1)
    nodes = [_node(f"n{i}", centroid=(i * 0.01, 0, 0)) for i in range(50)]
    store.install(nodes, [])

    stop = threading.Event()
    errors: list = []

    def rebuilder():
        i = 0
        while not stop.is_set():
            try:
                ns = [_node(f"r{i}-{j}", centroid=(j * 0.01, 0, 0)) for j in range(50)]
                store.install(ns, [Relation(f"r{i}-0", f"r{i}-1", "near", 0.5)])
                i += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def reader():
        for _ in range(400):
            try:
                _ = store.all_objects()
                _ = store.recently_seen(10)
                _ = store.query_near((0, 0), 1.0)
                _ = store.to_text_description()
                _ = store.query_text("n0", k=5)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    t = threading.Thread(target=rebuilder)
    t.start()
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    t.join()
    assert not errors, errors
