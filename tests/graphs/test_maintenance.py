"""Tier-2 periodic maintenance: re-merge, denoise, confirmation, ghost eviction."""

from __future__ import annotations

import numpy as np

from tests.graphs.conftest import make_cloud, make_det, put_object, unit


# ---------------------------------------------------------------------------
# merge_overlapping_nodes (ConceptGraphs merge_overlap_objects analog)
# ---------------------------------------------------------------------------
def test_merge_fuses_overlapping_split_nodes(mem):
    # Same class, coincident clouds, agreeing embeddings → one object.
    put_object(mem, "chair-a", "chair", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0), n_obs=4)
    put_object(mem, "chair-b", "chair", make_cloud((0, 0, 0), seed=2), emb=unit(1, 0, 0), n_obs=2)
    assert mem.count() == 2
    merged = mem.merge_overlapping_nodes()
    assert merged == 1
    assert mem.count() == 1
    kept = mem.all_objects()[0]
    assert kept.id == "chair-a"  # higher n_obs kept
    assert kept.n_obs == 6  # 4 + 2


def test_merge_refuses_when_visual_gate_fails(mem):
    # Clouds overlap but the embeddings disagree → distinct objects, no merge.
    put_object(mem, "a", "chair", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    put_object(mem, "b", "chair", make_cloud((0, 0, 0), seed=2), emb=unit(0, 1, 0))
    assert mem.merge_overlapping_nodes() == 0
    assert mem.count() == 2


def test_merge_refuses_when_not_overlapping(mem):
    put_object(mem, "a", "chair", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    put_object(mem, "b", "chair", make_cloud((2, 0, 0), seed=2), emb=unit(1, 0, 0))
    assert mem.merge_overlapping_nodes() == 0
    assert mem.count() == 2


def test_merge_only_same_class(mem):
    put_object(mem, "a", "chair", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    put_object(mem, "b", "table", make_cloud((0, 0, 0), seed=2), emb=unit(1, 0, 0))
    assert mem.merge_overlapping_nodes() == 0
    assert mem.count() == 2


# ---------------------------------------------------------------------------
# denoise_nodes (ConceptGraphs denoise_objects analog)
# ---------------------------------------------------------------------------
def test_denoise_strips_outlier_cluster(mem):
    blob = make_cloud((0, 0, 0), n=40, spread=0.005, seed=1)
    outliers = np.array(
        [[3, 0, 0], [0, 3, 0], [-3, 0, 0], [0, -3, 0], [3, 3, 3]], dtype=np.float32
    )
    node = put_object(mem, "x", "box", np.vstack([blob, outliers]), emb=unit(1, 0, 0))
    ext_before = node.extent
    mem._dirty.add("x")
    assert mem.denoise_nodes() == 1
    pts = mem.load_pcd("x")
    assert len(pts) == 40  # the 5 scattered outliers are gone
    assert mem.get("x").extent[0] < ext_before[0]  # AABB shrank


def test_denoise_skips_fragmented_spread_object(mem):
    # Three equal sub-blobs: the largest cluster is < keep_min_frac of the cloud, so a
    # blind largest-cluster keep would gut a legitimately spread object → left intact.
    pts = np.vstack(
        [
            make_cloud((0, 0, 0), n=20, spread=0.005, seed=1),
            make_cloud((1, 0, 0), n=20, spread=0.005, seed=2),
            make_cloud((2, 0, 0), n=20, spread=0.005, seed=3),
        ]
    )
    put_object(mem, "y", "shelf", pts, emb=unit(1, 0, 0))
    mem._dirty.add("y")
    assert mem.denoise_nodes() == 0  # skipped by the keep-min-frac guard
    assert len(mem.load_pcd("y")) == 60


def test_denoise_only_touches_dirty(mem):
    put_object(mem, "z", "box", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    # not marked dirty → not processed
    assert mem.denoise_nodes() == 0


# ---------------------------------------------------------------------------
# Confirmation gate (node precision)
# ---------------------------------------------------------------------------
def test_require_confirmation_hides_provisional(mem):
    mem.require_confirmation = True
    mem.min_obs_confirm = 3
    # One sighting → provisional → hidden from every query surface, but counted.
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), caption="mug"))
    assert mem.count() == 1
    assert mem.all_objects() == []
    assert mem.recently_seen() == []
    assert mem.query_near((1.0, 0.0, 0.5), radius=1.0) == []
    # Two more overlapping sightings promote it to confirmed.
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), caption="mug", ts=2.0))
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0), caption="mug", ts=3.0))
    objs = mem.all_objects()
    assert len(objs) == 1 and objs[0].n_obs == 3


def test_confirmation_off_by_default(mem):
    assert mem.require_confirmation is False
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), emb=unit(1, 0, 0)))
    assert len(mem.all_objects()) == 1  # visible immediately


# ---------------------------------------------------------------------------
# evict_stale_provisional (ghost cleanup)
# ---------------------------------------------------------------------------
def test_evict_stale_provisional(mem):
    mem.min_obs_confirm = 3
    mem.ghost_grace_sec = 100.0
    put_object(mem, "ghost", "x", make_cloud((0, 0, 0), seed=1), n_obs=1, ts=0.0)
    put_object(mem, "real", "y", make_cloud((2, 0, 0), seed=2), n_obs=3, ts=0.0)
    put_object(mem, "fresh", "z", make_cloud((4, 0, 0), seed=3), n_obs=1, ts=150.0)
    removed = mem.evict_stale_provisional(now_ts=200.0)
    assert removed == 1
    ids = {n.id for n in mem.all_objects()}
    assert ids == {"real", "fresh"}  # confirmed kept, recent provisional kept


def test_evict_disabled_by_default(mem):
    # default ghost_grace_sec=0 and min_obs_confirm=1 → never evicts
    put_object(mem, "ghost", "x", make_cloud((0, 0, 0), seed=1), n_obs=1, ts=0.0)
    assert mem.evict_stale_provisional(now_ts=1e9) == 0
    assert mem.count() == 1
