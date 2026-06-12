"""Tier-2 periodic maintenance: re-merge, denoise, confirmation, ghost eviction."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_merge_only_same_class_by_default(mem):
    # Constructor default: cross_class_sim_threshold = 0 → cross-class never merges.
    put_object(mem, "a", "chair", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    put_object(mem, "b", "table", make_cloud((0, 0, 0), seed=2), emb=unit(1, 0, 0))
    assert mem.merge_overlapping_nodes() == 0
    assert mem.count() == 2


def test_merge_cross_class_when_enabled(mem):
    # One physical object stored under two detector labels: with cross-class on,
    # coincident clouds + agreeing embeddings fuse despite the class mismatch.
    mem.cross_class_sim_threshold = 1.5
    put_object(mem, "a", "cup", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0), n_obs=3)
    put_object(mem, "b", "mug", make_cloud((0, 0, 0), seed=2), emb=unit(1, 0, 0), n_obs=1)
    assert mem.merge_overlapping_nodes() == 1
    assert mem.count() == 1
    assert mem.all_objects()[0].class_name == "cup"  # higher-n_obs node kept


def test_merge_cross_class_requires_embeddings(mem):
    # Without CLIP evidence, geometry alone must not override the class labels.
    mem.cross_class_sim_threshold = 1.5
    put_object(mem, "a", "cup", make_cloud((0, 0, 0), seed=1), emb=[])
    put_object(mem, "b", "mug", make_cloud((0, 0, 0), seed=2), emb=[])
    assert mem.merge_overlapping_nodes() == 0
    assert mem.count() == 2


def _o3d_available() -> bool:
    from services.walkie_graphs.dbscan import _open3d

    return bool(_open3d())


def test_big_object_fills_across_overlapping_sweep(tmp_path):
    """A large object scanned as overlapping partial strips accretes into ONE full node.

    Each strip overlaps the previous by ~50% and extends past it; the union must
    preserve every extension so the object fills in. (This is the case the old
    per-object ICP mishandled — flat surfaces are translation-degenerate, so ICP
    slid extensions back onto the stored cloud. With registration done per capture
    against the whole map, no per-object alignment ever touches the strips.)
    """
    from services.walkie_graphs.memory import Detection3D, GraphMemory, aabb_of

    mem = GraphMemory(
        chroma_dir=None, pcds_dir=str(tmp_path / "p"), thumbs_dir=str(tmp_path / "t"),
        edges_path=str(tmp_path / "e.json"),
        dedup_radius_m=0.3, dedup_visual_k=0,
        visual_merge_max_dist_m=0.4, dbscan_enabled=False, sor_k=0, voxel_m=0.02,
        max_points_per_obj=20000,
    )
    rng = np.random.default_rng(0)
    emb = unit(1, 0, 0)

    def strip(x0, x1):
        n = int(4000 * (x1 - x0))
        pts = np.stack(
            [rng.uniform(x0, x1, n), rng.uniform(0, 1.6, n), rng.normal(0, 0.01, n)], axis=1
        ).astype(np.float32)
        return Detection3D("bed", 0, 0.9, (0, 0, 10, 10), pts, emb, "a bed", 1.0)

    for x0, x1 in [(0.0, 0.8), (0.4, 1.2), (0.8, 1.6), (1.2, 2.0)]:
        mem.upsert(strip(x0, x1))

    assert mem.count() == 1
    x_extent = aabb_of(mem.load_pcd(mem.all_objects()[0].id))[3][0]
    assert x_extent > 1.8  # filled the whole ~2 m object, not stuck near one 0.8 m strip


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
    # Three equal sub-blobs: all are real clusters, so the noise-only denoise removes
    # nothing — a legitimately spread/multi-view object is preserved by design.
    pts = np.vstack(
        [
            make_cloud((0, 0, 0), n=20, spread=0.005, seed=1),
            make_cloud((1, 0, 0), n=20, spread=0.005, seed=2),
            make_cloud((2, 0, 0), n=20, spread=0.005, seed=3),
        ]
    )
    put_object(mem, "y", "shelf", pts, emb=unit(1, 0, 0))
    mem._dirty.add("y")
    assert mem.denoise_nodes() == 0  # nothing removed
    assert len(mem.load_pcd("y")) == 60


def test_denoise_keeps_all_view_clusters_drops_only_strays(mem):
    # An accumulated two-view cloud (two ends of a bed, middle never seen) + isolated
    # stray points: the periodic denoise must remove ONLY the strays — both view
    # clusters survive and the AABB still spans them.
    blob_a = make_cloud((0, 0, 0), n=40, spread=0.005, seed=1)
    blob_b = make_cloud((1.5, 0, 0), n=40, spread=0.005, seed=2)
    strays = np.array(
        [[5, 5, 5], [-4, 0, 2], [0, -6, 1], [7, 1, 0], [3, 3, -3]], dtype=np.float32
    )
    node = put_object(mem, "bed", "bed", np.vstack([blob_a, blob_b, strays]), emb=unit(1, 0, 0))
    assert node.aabb_max[0] > 6  # strays inflate the AABB before denoise
    mem._dirty.add("bed")
    assert mem.denoise_nodes() == 1
    pts = mem.load_pcd("bed")
    assert len(pts) == 80  # only the 5 strays removed; BOTH clusters kept
    n = mem.get("bed")
    assert n.aabb_min[0] == pytest.approx(0.0, abs=0.05)
    assert n.aabb_max[0] == pytest.approx(1.5, abs=0.05)  # spans both views, no strays


def test_denoise_only_touches_dirty(mem):
    put_object(mem, "z", "box", make_cloud((0, 0, 0), seed=1), emb=unit(1, 0, 0))
    # not marked dirty → not processed
    assert mem.denoise_nodes() == 0


def test_denoise_sor_erases_accumulated_halo(mem):
    # Simulate fuzz accumulation: a dense object cloud plus a sparse halo built up from
    # many sightings' leftover edge artifacts. SOR in the periodic pass must strip the
    # halo (which inflates the AABB until neighbours falsely overlap) while keeping the
    # dense structure — including a second disjoint view cluster.
    mem.sor_k = 16
    mem.sor_std_ratio = 1.5
    rng = np.random.default_rng(7)
    view_a = make_cloud((0, 0, 0), n=150, spread=0.02, seed=1)
    view_b = make_cloud((1.0, 0, 0), n=150, spread=0.02, seed=2)  # other end, disjoint
    # halo: 20 points scattered well off the surfaces (accumulated flying pixels)
    halo = np.vstack(
        [
            rng.normal((0.5, 0.0, 0.4), 0.3, (10, 3)),
            rng.normal((0.5, 0.4, 0.0), 0.3, (10, 3)),
        ]
    ).astype(np.float32)
    node = put_object(mem, "bed", "bed", np.vstack([view_a, view_b, halo]), emb=unit(1, 0, 0))
    ext_before = node.extent
    mem._dirty.add("bed")
    assert mem.denoise_nodes() == 1
    pts = mem.load_pcd("bed")
    n = mem.get("bed")
    # both dense view clusters survive...
    assert (pts[:, 0] < 0.5).sum() > 100 and (pts[:, 0] > 0.5).sum() > 100
    # ...the halo is (mostly) gone and the AABB tightened back around the object
    assert len(pts) < 300 + 10
    assert n.extent[1] < ext_before[1] and n.extent[2] < ext_before[2]


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
