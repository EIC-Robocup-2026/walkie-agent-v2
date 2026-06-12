"""Segment-backed object clouds: refs, baking, rebuild, GC across the node lifecycle."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from services.walkie_graphs.capture import Capture, CaptureStore, Segment
from services.walkie_graphs.memory import GraphMemory
from tests.graphs.conftest import make_det, unit


@pytest.fixture
def store(tmp_path):
    return CaptureStore(str(tmp_path / "captures"))


@pytest.fixture
def smem(tmp_path, store):
    """GraphMemory wired to a CaptureStore, with a tiny bake threshold."""
    return GraphMemory(
        chroma_dir=None,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
        capture_store=store,
        segments_per_node=3,
    )


def seg_det(store, cid, center, **kw):
    """A Detection3D whose points are also persisted as capture ``cid``'s segment 0."""
    det = make_det(center=center, emb=unit(1, 0, 0), **kw)
    cap = Capture(
        id=cid, ts=det.ts, cam=None,
        segments=[Segment(cid, 0, det.points_world)],
    )
    store.save(cap)
    return replace(det, segment_ref=f"{cid}:0")


def test_insert_attaches_and_retains_ref(smem, store, tmp_path):
    node = smem.upsert(seg_det(store, "c1-a", (1.0, 0.0, 0.5)))
    assert node.segments == ["c1-a:0"]
    store.flush()
    assert store.gc() == 0  # retained by the node → not collectable
    assert (tmp_path / "captures" / "c1-a.npz").exists()


def test_merge_appends_refs_then_bakes_oldest(smem, store, tmp_path):
    # 5 sightings of one object with segments_per_node=3: the node keeps the 3
    # newest refs, the 2 oldest fold into the baked base, and their capture
    # files become collectable.
    for i in range(5):
        node = smem.upsert(seg_det(store, f"c{i}-x", (1.0, 0.0, 0.5), ts=float(i + 1)))
    assert smem.count() == 1
    assert node.n_obs == 5
    assert node.segments == ["c2-x:0", "c3-x:0", "c4-x:0"]
    assert len(smem.load_base(node.id)) > 0
    store.flush()
    assert store.gc() == 2  # c0, c1 released by the bake
    assert not (tmp_path / "captures" / "c0-x.npz").exists()
    assert (tmp_path / "captures" / "c4-x.npz").exists()


def test_base_survives_flush_roundtrip(smem, store):
    for i in range(5):
        node = smem.upsert(seg_det(store, f"c{i}-y", (1.0, 0.0, 0.5), ts=float(i + 1)))
    base_before = smem.load_base(node.id)
    assert len(base_before) > 0
    smem.flush_pcds()
    # A cold reader (fresh caches) sees the same base from the npz.
    smem._base_cache.clear()
    smem._pcd_cache.clear()
    assert np.array_equal(smem.load_base(node.id), base_before)


def test_rebuild_pcd_re_derives_from_segments(smem, store):
    # Two unioned sightings of a long object's two ends: the rebuilt cloud must
    # span both, matching the incrementally-fused extents.
    def end_det(cid, x0, x1, ts):
        rng = np.random.default_rng(int(ts))
        pts = np.stack(
            [rng.uniform(x0, x1, 300), rng.normal(0, 0.01, 300), rng.normal(0.5, 0.01, 300)],
            axis=1,
        ).astype(np.float32)
        det = make_det(class_name="bed", emb=unit(1, 0, 0), ts=ts)
        det = replace(det, points_world=pts)
        cap = Capture(id=cid, ts=ts, cam=None, segments=[Segment(cid, 0, pts)])
        store.save(cap)
        return replace(det, segment_ref=f"{cid}:0")

    smem.upsert(end_det("c1-b", 0.0, 1.2, 1.0))
    node = smem.upsert(end_det("c2-b", 0.8, 2.0, 2.0))
    assert smem.count() == 1
    fused = smem.load_pcd(node.id)
    rebuilt = smem.rebuild_pcd(node.id)
    assert rebuilt[:, 0].min() == pytest.approx(fused[:, 0].min(), abs=0.05)
    assert rebuilt[:, 0].max() == pytest.approx(fused[:, 0].max(), abs=0.05)
    assert node.aabb_max[0] == pytest.approx(2.0, abs=0.1)


def test_rebuild_pcd_keeps_legacy_cloud(smem):
    # A node with no segments and no base (pre-segment store) is left untouched.
    node = smem.upsert(make_det(center=(1.0, 0.0, 0.5)))  # no segment_ref
    before = smem.load_pcd(node.id)
    assert np.array_equal(smem.rebuild_pcd(node.id), before)


def test_merge_nodes_unions_segment_refs(smem, store):
    a = smem.upsert(seg_det(store, "c1-m", (1.0, 0.0, 0.5), ts=1.0))
    b = smem.upsert(
        seg_det(store, "c2-m", (5.0, 0.0, 0.5), class_name="lamp", ts=2.0)
    )
    assert smem.count() == 2
    with smem._lock:
        smem._merge_nodes(smem.get(a.id), smem.get(b.id))
    assert smem.count() == 1
    keep = smem.get(a.id)
    assert keep.segments == ["c1-m:0", "c2-m:0"]
    store.flush()
    assert store.gc() == 0  # both captures still referenced by the kept node


def test_replace_branch_resets_history(smem, store, tmp_path):
    # A far, non-overlapping re-sighting (visual merge, higher conf) swaps the
    # geometry: old segments + base are dropped with the old location.
    for i in range(5):  # build up segments AND a baked base at the old spot
        smem.upsert(seg_det(store, f"c{i}-r", (1.0, 0.0, 0.5), ts=float(i + 1), conf=0.6))
    node = smem.upsert(seg_det(store, "c9-r", (4.0, 0.0, 0.5), ts=9.0, conf=0.95))
    assert smem.count() == 1
    assert node.segments == ["c9-r:0"]
    assert len(smem.load_base(node.id)) == 0
    store.flush()
    store.gc()
    assert not (tmp_path / "captures" / "c4-r.npz").exists()
    assert (tmp_path / "captures" / "c9-r.npz").exists()


def test_delete_releases_refs(smem, store, tmp_path):
    node = smem.upsert(seg_det(store, "c1-d", (1.0, 0.0, 0.5)))
    store.flush()
    with smem._lock:
        smem._delete(node.id)
    assert store.gc() == 1
    assert not (tmp_path / "captures" / "c1-d.npz").exists()


def test_segments_roundtrip_chroma_metadata(smem, store):
    node = smem.upsert(seg_det(store, "c1-s", (1.0, 0.0, 0.5)))
    md = smem._metadata(node)
    again = GraphMemory._node_from_chroma(node.id, node.clip_emb, md)
    assert again.segments == ["c1-s:0"]
    # Legacy metadata (no segments_json) loads as a segment-less node.
    md.pop("segments_json")
    legacy = GraphMemory._node_from_chroma(node.id, node.clip_emb, md)
    assert legacy.segments == []


def test_load_nodes_re_retains_refs(tmp_path, store):
    # Across a restart the loaded graph must re-retain its capture refs, so the
    # startup gc sweep removes only true orphans.
    chroma_dir = str(tmp_path / "chroma")
    mem = GraphMemory(
        chroma_dir=chroma_dir,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
        capture_store=store,
    )
    mem.upsert(seg_det(store, "c1-z", (1.0, 0.0, 0.5)))
    store.flush()
    # Orphan: a capture no node references (e.g. its node was lost in a crash).
    store.save(
        Capture(id="c9-orphan", ts=1.0, cam=None,
                segments=[Segment("c9-orphan", 0, np.ones((5, 3), np.float32))])
    )
    store.flush()

    fresh_store = CaptureStore(str(tmp_path / "captures"))
    mem2 = GraphMemory(
        chroma_dir=chroma_dir,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
        capture_store=fresh_store,
    )
    assert [n.segments for n in mem2.all_objects()] == [["c1-z:0"]]
    assert fresh_store.gc() == 1  # the orphan only
    assert (tmp_path / "captures" / "c1-z.npz").exists()
    assert not (tmp_path / "captures" / "c9-orphan.npz").exists()
