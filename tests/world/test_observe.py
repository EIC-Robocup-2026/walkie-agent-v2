"""The producer->model ingest contract: observe_objects merges, derives, installs;
queries see the result; concurrent reads during installs stay consistent."""

from __future__ import annotations

import threading

import numpy as np

from walkie_world import WalkieWorld
from walkie_world.scene.ingest import ObjectObservation
from walkie_world.scene.store import aabb_of


def _obs(class_name, center, *, n_obs=2, captions=None, emb=None, ts=1.0):
    c = np.asarray(center, dtype=float)
    # A small deterministic cluster around the centre (no RNG).
    offs = (np.arange(60).reshape(20, 3) % 3 - 1) * 0.02
    pts = (c + offs).astype(np.float32)
    centroid, mn, mx, ext = aabb_of(pts)
    return ObjectObservation(
        class_name=class_name, class_id=None, conf=0.9,
        captions=list(captions or []), clip_emb=list(emb or []),
        ts_first=ts, ts_last=ts, n_obs=n_obs, points=pts,
        centroid=centroid, extent=ext, aabb_min=mn, aabb_max=mx,
    )


def test_observe_merges_and_queries(tmp_path):
    w = WalkieWorld(scene_dir=str(tmp_path / "scene"), enable_people=False)
    w.observe_objects([
        _obs("cup", (1.0, 1.0, 0.8), captions=["a red cup"]),
        _obs("bowl", (1.3, 1.0, 0.8), captions=["a blue bowl"]),
    ])
    assert w.count() == 2
    assert {n.class_name for n in w.all_objects()} == {"cup", "bowl"}

    # No embed_text -> keyword fallback finds by caption text.
    hits = w.query_text("red cup")
    assert hits and hits[0].best_caption == "a red cup"

    # 0.3 m apart -> a "near" relation is derived (near_m default 0.6).
    cup = next(n for n in w.all_objects() if n.class_name == "cup")
    assert any(r.predicate == "near" for r in w.relations_of(cup.id))


def test_observe_accretes_across_calls(tmp_path):
    w = WalkieWorld(scene_dir=str(tmp_path / "scene"), enable_people=False)
    w.observe_objects([_obs("cup", (1.0, 1.0, 0.8))])
    w.observe_objects([_obs("bowl", (3.0, 1.0, 0.8))])  # far apart -> distinct node
    assert w.count() == 2


def test_concurrent_query_during_installs(tmp_path):
    w = WalkieWorld(scene_dir=str(tmp_path / "scene"), enable_people=False)
    errors: list[Exception] = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                w.query_text("cup")
                w.all_objects()
                w.recently_seen(3)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(25):
            w.observe_objects([_obs("cup", (1.0, 1.0, 0.8), ts=float(i))])
    finally:
        stop.set()
        t.join(timeout=5)

    assert not errors, f"reader hit: {errors}"
    assert w.count() == 1  # all merged into one node (same class, same place)
