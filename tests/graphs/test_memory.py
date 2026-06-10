"""GraphMemory fusion/dedup + storage + query tests (fake embeddings, no server)."""

from __future__ import annotations

from dataclasses import replace

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
# ConceptGraphs additive-greedy association (_associate path, distinct from cascade)
# ---------------------------------------------------------------------------
def test_associate_merges_on_overlap_even_with_mid_cosine(mem):
    # Heavily overlapping clouds + mid cosine: nn_ratio is high, so the additive
    # score clears sim_threshold via the _associate path (the cascade alone would
    # need cos>=sim_high). spread small so the two clouds physically coincide.
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01))
    mid = emb_with_cosine(0.7)
    mem.upsert(make_det(center=(1.005, 0.0, 0.5), emb=mid, spread=0.01, ts=2.0))
    assert mem.count() == 1


def test_associate_is_geometry_gated_on_pure_visual(mem):
    # _associate never fires on visual alone: even identical embeddings (cos 1.0)
    # return None when the clouds don't overlap (additive tops out at 1.0 < 1.1).
    # Probed directly so the _classify cascade's own visual-dedup path (which DOES
    # merge identical embeddings — drift recovery) doesn't mask the geometric gate.
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01))
    far = make_det(center=(1.3, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01, ts=2.0)
    far = replace(far, points_world=mem._denoise(far.points_world))
    assert mem._associate(far) is None


def test_associate_geometry_only_merge_without_embedding(mem):
    # No embeddings at all (embed route down): a re-sighting whose cloud overlaps
    # still merges geometrically — nn_ratio≈1, phi_sem(0)=0.5 → ~1.5 ≥ 1.1.
    mem.upsert(make_det(center=(1.0, 0.0, 0.5), emb=[], spread=0.01))
    node = mem.upsert(make_det(center=(1.005, 0.0, 0.5), emb=[], spread=0.01, ts=2.0))
    assert mem.count() == 1
    assert node.n_obs == 2


def test_cross_class_assoc_merges_detector_label_flipflop(mem):
    # The detector calls one object "cup" then "mug": same place (clouds overlap),
    # same appearance. With cross-class association on (strict threshold), the second
    # sighting folds into the first node instead of duplicating it.
    mem.cross_class_sim_threshold = 1.5
    mem.upsert(make_det(class_name="cup", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01))
    node = mem.upsert(
        make_det(class_name="mug", center=(1.005, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01, ts=2.0)
    )
    assert mem.count() == 1
    assert node.n_obs == 2
    assert node.class_name == "cup"  # the original node's identity is kept


def test_cross_class_assoc_respects_strict_threshold(mem):
    # Cross-class needs ~full overlap AND agreeing CLIP. Overlapping clouds but a
    # mid cosine (phi ≈ 1.0 + 0.85 = 1.85 ≥ 1.5 would merge — so use orthogonal
    # embeddings: phi ≈ 1.0 + 0.5 = 1.5... boundary). Use clearly-different emb so
    # phi ≈ 1.0 + 0.25 < 1.5 → stays two objects.
    mem.cross_class_sim_threshold = 1.5
    mem.upsert(make_det(class_name="cup", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), spread=0.01))
    mem.upsert(
        make_det(
            class_name="bottle",
            center=(1.005, 0.0, 0.5),
            emb=[-0.5, 0.866, 0.0],  # cos -0.5 vs unit(1,0,0) → phi_sem 0.25
            spread=0.01,
            ts=2.0,
        )
    )
    assert mem.count() == 2


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
    # Point cloud round-trips through the .npz sidecar (cold cache on the new store).
    assert len(m2.load_pcd(node.id)) > 0


def test_pcd_cache_write_through_and_invalidation(mem):
    node = mem.upsert(make_det(center=(1.0, 0.0, 0.5)))
    cached = mem.load_pcd(node.id)
    assert node.id in mem._pcd_cache
    assert np.array_equal(cached, mem._pcd_cache[node.id])
    # Disk and cache agree after a merge rewrites the cloud.
    mem.upsert(make_det(center=(1.01, 0.0, 0.5), ts=2.0))
    assert np.array_equal(mem.load_pcd(node.id), np.load(mem._pcd_path(node.id))["points"])
    # Deleting the node evicts the cache entry.
    mem._delete(node.id)
    assert node.id not in mem._pcd_cache
    assert len(mem.load_pcd(node.id)) == 0
