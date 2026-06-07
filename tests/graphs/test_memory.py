"""GraphMemory fusion/dedup + storage + query tests (fake embeddings, no server)."""

from __future__ import annotations

import numpy as np
import pytest

from walkie_graphs.memory import GraphMemory
from tests.graphs.conftest import emb_with_cosine, make_det, unit


# ---------------------------------------------------------------------------
# Fusion / dedup
# ---------------------------------------------------------------------------
def test_insert_creates_node(mem):
    node = mem.upsert(make_det(center=(1.0, 0.0, 0.5), caption="a white mug"))
    assert mem.count() == 1
    assert node.class_name == "mug"
    assert node.n_obs == 1
    assert node.best_caption == "a white mug"
    assert node.centroid[0] == pytest.approx(1.0, abs=0.05)
    assert len(mem.load_pcd(node.id)) > 0


def test_merge_high_cosine(mem):
    a = mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    b = mem.upsert(make_det(center=(1.05, 0.0, 0.5), emb=unit(1, 0, 0), ts=2.0))
    assert mem.count() == 1
    assert a.id == b.id
    assert b.n_obs == 2
    assert b.last_seen_ts == 2.0


def test_insert_when_low_cosine_and_far(mem):
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    # Orthogonal embedding + far away → a distinct object.
    mem.upsert(make_det(center=(3.0, 2.0, 0.5), emb=unit(0, 1, 0)))
    assert mem.count() == 2


def test_merge_mid_cosine_when_tight(mem):
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    mid = emb_with_cosine(0.7)  # between SIM_LOW(0.65) and SIM_HIGH(0.85)
    mem.upsert(make_det(center=(1.1, 0.0, 0.5), emb=mid))  # within TIGHT (0.2 m)
    assert mem.count() == 1


def test_insert_mid_cosine_when_not_tight(mem):
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    mid = emb_with_cosine(0.7)
    # cos 0.7 < HIGH and distance 0.35 m > TIGHT(0.2) → not a merge.
    mem.upsert(make_det(center=(1.35, 0.0, 0.5), emb=mid))
    assert mem.count() == 2


def test_cross_class_never_merges(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    mem.upsert(make_det(class_name="cup", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    assert mem.count() == 2


def test_running_mean_confidence_on_merge(mem):
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), conf=0.8))
    node = mem.upsert(make_det(center=(1.02, 0.0, 0.5), emb=unit(1, 0, 0), conf=1.0, ts=2.0))
    assert node.conf == pytest.approx(0.9, abs=1e-6)


def test_merge_unions_cloud_and_grows_aabb(mem):
    a = mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.005))
    ext_before = a.extent
    b = mem.upsert(make_det(center=(1.2, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.005, ts=2.0))
    # merged cloud spans both centers → x-extent clearly larger, centroid moved.
    assert b.extent[0] > ext_before[0] + 0.1
    assert 1.0 < b.centroid[0] < 1.2


def test_far_visual_merge_keeps_higher_conf_geometry(mem):
    # Identical embedding (cos 1.0 → merge) but far apart: must NOT average into the
    # midpoint; the higher-confidence detection's position wins.
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), conf=0.6))
    node = mem.upsert(make_det(center=(4.0, 0.0, 0.5), emb=unit(1, 0, 0), conf=0.95, ts=2.0))
    assert mem.count() == 1
    assert node.centroid[0] == pytest.approx(4.0, abs=0.1)  # not ~2.5


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def test_query_text_cross_modal(mem):
    mem.embed_text = lambda q: unit(1, 0, 0)  # query embeds onto the first node
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), caption="mug"))
    mem.upsert(make_det(class_name="ball", center=(3.0, 0.0, 0.5), emb=unit(0, 1, 0), caption="ball"))
    hits = mem.query_text("the mug", k=1)
    assert hits and hits[0].class_name == "mug"


def test_query_text_keyword_fallback_when_no_embedder(mem):
    assert mem.embed_text is None
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a red mug"))
    mem.upsert(make_det(class_name="ball", center=(3.0, 0.0, 0.5), caption="a blue ball"))
    hits = mem.query_text("red mug", k=1)
    assert hits and hits[0].class_name == "mug"


def test_query_text_falls_back_when_embed_raises(mem):
    def boom(_q):
        raise RuntimeError("embed route down")

    mem.embed_text = boom
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a mug"))
    hits = mem.query_text("mug", k=1)
    assert hits and hits[0].class_name == "mug"


def test_query_near(mem):
    mem.upsert(make_det(class_name="a", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    mem.upsert(make_det(class_name="b", center=(1.3, 0.0, 0.5), emb=unit(0, 1, 0)))
    mem.upsert(make_det(class_name="c", center=(5.0, 0.0, 0.5), emb=unit(0, 0, 1)))
    near = mem.query_near((1.0, 0.0, 0.5), radius=0.6)
    names = {n.class_name for n in near}
    assert names == {"a", "b"}


def test_recently_seen_orders_by_last_seen(mem):
    mem.upsert(make_det(class_name="a", center=(0, 0, 0), emb=unit(1, 0, 0), ts=1.0))
    mem.upsert(make_det(class_name="b", center=(2, 0, 0), emb=unit(0, 1, 0), ts=3.0))
    mem.upsert(make_det(class_name="c", center=(4, 0, 0), emb=unit(0, 0, 1), ts=2.0))
    recent = mem.recently_seen(2)
    assert [n.class_name for n in recent] == ["b", "c"]


# ---------------------------------------------------------------------------
# Maintenance / persistence
# ---------------------------------------------------------------------------
def test_prune_keeps_newest(mem):
    mem.prune_max_records = 2
    mem.upsert(make_det(class_name="a", center=(0, 0, 0), emb=unit(1, 0, 0), ts=1.0))
    mem.upsert(make_det(class_name="b", center=(2, 0, 0), emb=unit(0, 1, 0), ts=2.0))
    mem.upsert(make_det(class_name="c", center=(4, 0, 0), emb=unit(0, 0, 1), ts=3.0))
    removed = mem.prune()
    assert removed == 1
    assert mem.count() == 2
    assert {n.class_name for n in mem.all_objects()} == {"b", "c"}


def test_clear(mem):
    mem.upsert(make_det(center=(1.0, 0.0, 0.5)))
    mem.clear()
    assert mem.count() == 0


def test_persistence_reload(tmp_path):
    chroma = str(tmp_path / "chroma")
    pcds = str(tmp_path / "pcds")
    thumbs = str(tmp_path / "thumbs")
    edges = str(tmp_path / "edges.json")
    m1 = GraphMemory(chroma_dir=chroma, pcds_dir=pcds, thumbs_dir=thumbs, edges_path=edges)
    m1.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), caption="mug"))
    m1.upsert(make_det(class_name="ball", center=(3.0, 0.0, 0.5), emb=unit(0, 1, 0), caption="ball"))

    m2 = GraphMemory(chroma_dir=chroma, pcds_dir=pcds, thumbs_dir=thumbs, edges_path=edges)
    assert m2.count() == 2
    node = next(n for n in m2.all_objects() if n.class_name == "mug")
    assert node.best_caption == "mug"
    assert len(node.clip_emb) == 3  # embedding round-tripped through Chroma
