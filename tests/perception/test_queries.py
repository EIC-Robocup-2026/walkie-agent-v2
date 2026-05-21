"""Phase 3 unit tests for the SceneStore query API.

Seed a fresh store with ~10 records spanning multiple classes, positions,
captions, and timestamps. Each test then exercises one query path and
asserts on the result. The FakeEmbedder produces deterministic vectors
so text/visual knn ranking is repeatable.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from perception import Detection, SceneStore
from perception.mocks import FakeEmbedder, make_tiny_image


# Two text queries we pin to specific override vectors so we can predict
# which scene entries rank highest. Each override is a 16-dim unit vector
# whose first coordinate biases toward one "class" of caption embedding.

def _unit_vec(dim: int, axis: int) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


@pytest.fixture
def embedder():
    # The store will record dim=16. We pin the text embedding for
    # specific query strings, and we pin caption embeddings via the
    # override_text knob so semantic_query is deterministic.
    return FakeEmbedder(
        dim=16,
        override_text={
            "coffee mug": _unit_vec(16, 0),
            "chair": _unit_vec(16, 1),
            "table": _unit_vec(16, 2),
        },
    )


@pytest.fixture
def seeded_store(tmp_path, embedder):
    """Seed 10 records with known classes, positions, and timestamps."""
    store = SceneStore(persist_dir=tmp_path / "chroma", embedder=embedder)

    # Build detections whose embeddings line up with the override text
    # vectors so semantic queries return predictable matches.
    base_ts = 1_000_000.0
    records = [
        # Three mugs at varying positions and ages. Spaced > 0.5m apart
        # so they don't merge into each other on insert.
        ("mug",   (0.0, 0.0, 0.5), _unit_vec(16, 0), base_ts + 0),
        ("mug",   (1.0, 0.0, 0.5), _unit_vec(16, 0), base_ts + 100),
        ("mug",   (5.0, 5.0, 0.5), _unit_vec(16, 0), base_ts + 200),
        # Four chairs
        ("chair", (1.0, 0.0, 0.0), _unit_vec(16, 1), base_ts + 50),
        ("chair", (2.0, 0.0, 0.0), _unit_vec(16, 1), base_ts + 60),
        ("chair", (-1.0, -1.0, 0.0), _unit_vec(16, 1), base_ts + 70),
        ("chair", (5.0, -5.0, 0.0), _unit_vec(16, 1), base_ts + 300),
        # Two tables
        ("table", (0.0, 1.0, 0.7), _unit_vec(16, 2), base_ts + 80),
        ("table", (4.0, 4.0, 0.7), _unit_vec(16, 2), base_ts + 400),
        # One lamp (no override; gets a hashed embedding)
        ("lamp",  (3.0, 3.0, 1.2), None,             base_ts + 500),
    ]
    for i, (cls, pos, emb, ts) in enumerate(records):
        if emb is None:
            emb_list = embedder.embed_text(f"a {cls}")
        else:
            emb_list = emb
        det = Detection(
            class_name=cls,
            class_id=i,
            confidence=0.9,
            bbox_xyxy=(10 * i, 10 * i, 10 * i + 50, 10 * i + 50),
            position=pos,
            embedding=tuple(emb_list),
            caption=f"a {cls}, item {i}",
            ts=ts,
        )
        store.upsert(det)
    assert store.count == 10
    return store, base_ts


# ---------------------------------------------------------------------------
# Semantic query
# ---------------------------------------------------------------------------


def test_01_semantic_query_returns_class_match(seeded_store):
    store, _ = seeded_store
    # "coffee mug" override → unit_vec(16, 0) → matches the three mugs.
    hits = store.semantic_query("coffee mug", n_results=5)
    assert len(hits) >= 3
    # Top three should all be mugs.
    top = hits[:3]
    assert all(e.class_name == "mug" for e in top)


def test_02_semantic_query_with_spatial_filter(seeded_store):
    store, _ = seeded_store
    # Exclude the far mug at (5, 5, 0.5) by limiting search to a 1m ball.
    hits = store.semantic_query(
        "coffee mug",
        n_results=5,
        within_radius_of=(0.0, 0.0, 0.5),
        max_distance_m=1.0,
    )
    assert all(e.class_name == "mug" for e in hits)
    for e in hits:
        d = math.sqrt(
            (e.position[0]) ** 2 + (e.position[1]) ** 2 + (e.position[2] - 0.5) ** 2
        )
        assert d <= 1.0


def test_03_semantic_query_with_recency_filter(seeded_store):
    store, base_ts = seeded_store
    # min_last_seen_ts cuts off the early mugs (ts=base_ts, base_ts+100).
    # Only the mug at ts=base_ts+200 survives the recency filter.
    hits = store.semantic_query(
        "coffee mug", n_results=5, min_last_seen_ts=base_ts + 150
    )
    mugs = [h for h in hits if h.class_name == "mug"]
    assert len(mugs) == 1
    assert mugs[0].last_seen_ts == base_ts + 200


# ---------------------------------------------------------------------------
# Visual query
# ---------------------------------------------------------------------------


def test_04_visual_query_returns_results(seeded_store):
    store, _ = seeded_store
    # Just exercise the path — the FakeEmbedder hashes image bytes so we
    # don't pin a winning class. Assert we get n_results back.
    img = make_tiny_image(seed=42)
    hits = store.visual_query(img, n_results=3)
    assert len(hits) == 3
    # Every result should have a populated distance field (from KNN).
    assert all(h.distance is not None for h in hits)


# ---------------------------------------------------------------------------
# Spatial query
# ---------------------------------------------------------------------------


def test_05_spatial_query_returns_all_within_ball(seeded_store):
    store, _ = seeded_store
    # Center at origin, radius 1.5m. Should pick up: mug@(0,0,0.5),
    # mug@(0.5,0,0.5), chair@(1,0,0), chair@(-1,-1,0), table@(0,1,0.7).
    hits = store.spatial_query(center=(0.0, 0.0, 0.0), radius_m=1.5)
    classes = sorted(h.class_name for h in hits)
    assert classes == sorted(["mug", "mug", "chair", "chair", "table"])


def test_06_spatial_query_with_class_filter(seeded_store):
    store, _ = seeded_store
    hits = store.spatial_query(
        center=(0.0, 0.0, 0.0), radius_m=10.0, class_name="chair"
    )
    assert all(h.class_name == "chair" for h in hits)
    assert len(hits) == 4


# ---------------------------------------------------------------------------
# Recency query
# ---------------------------------------------------------------------------


def test_07_recency_query_filters_by_last_seen(seeded_store):
    store, base_ts = seeded_store
    # Cut at base_ts + 250 — survivors: mug@200 is OUT, chair@300, table@400, lamp@500.
    hits = store.recency_query(since_ts=base_ts + 250)
    classes = sorted(h.class_name for h in hits)
    assert classes == sorted(["chair", "table", "lamp"])


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_08_diff_partitions_correctly(seeded_store):
    store, base_ts = seeded_store
    # Cut at base_ts + 250. With a fresh seed, every record has
    # first_seen_ts == last_seen_ts. So:
    #   appeared:    first_seen_ts > 250 → chair@300, table@400, lamp@500
    #   refreshed:   first_seen_ts <= 250 AND last_seen_ts > 250 → none
    #   disappeared: last_seen_ts <= 250 → everything else (7 records)
    diff = store.diff(since_ts=base_ts + 250)
    appeared_classes = sorted(e.class_name for e in diff.appeared)
    assert appeared_classes == sorted(["chair", "table", "lamp"])
    assert len(diff.refreshed) == 0
    assert len(diff.disappeared) == 7


def test_09_diff_with_refresh_detects_existing_object_resighting(
    tmp_path, embedder
):
    """A record that gets re-sighted lands in `refreshed`, not `appeared`."""
    store = SceneStore(persist_dir=tmp_path / "chroma", embedder=embedder)
    base_ts = 1_000_000.0
    # Initial sighting at ts = base_ts
    det1 = Detection(
        class_name="chair", class_id=0, confidence=0.9,
        bbox_xyxy=(0, 0, 50, 50), position=(0.0, 0.0, 0.0),
        embedding=tuple(_unit_vec(16, 1)), caption="a chair", ts=base_ts,
    )
    store.upsert(det1)
    # Same chair re-sighted at ts = base_ts + 500
    det2 = replace(det1, ts=base_ts + 500, position=(0.05, 0.0, 0.0))
    store.upsert(det2)
    assert store.count == 1

    diff = store.diff(since_ts=base_ts + 100)
    assert len(diff.appeared) == 0
    assert len(diff.refreshed) == 1
    assert diff.refreshed[0].sightings == 2
    assert len(diff.disappeared) == 0


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def test_10_prune_by_ttl(seeded_store):
    store, base_ts = seeded_store
    # "now" = base_ts + 1000. TTL = 800 → cutoff = base_ts + 200.
    # Records with last_seen_ts < cutoff are pruned.
    removed = store.prune(ttl_sec=800.0, now=base_ts + 1000)
    # ts < 200: mug@0, chair@50, chair@60, chair@70, table@80, mug@100 → 6 removed.
    assert removed == 6
    assert store.count == 4


def test_11_prune_by_max_records_keeps_freshest(seeded_store):
    store, _ = seeded_store
    removed = store.prune(max_records=3)
    assert removed == 7
    assert store.count == 3
    # The three survivors should be the records with the largest last_seen_ts.
    remaining = store.recency_query(since_ts=0.0)
    timestamps = sorted(e.last_seen_ts for e in remaining)
    assert timestamps == sorted(timestamps, reverse=False)  # trivially true
    assert min(e.last_seen_ts for e in remaining) >= 1_000_000.0 + 300


def test_12_upsert_after_prune_is_clean(seeded_store):
    """Inserting a fresh record after a prune doesn't trip on a stale id."""
    store, base_ts = seeded_store
    store.prune(max_records=3)
    assert store.count == 3

    # Brand new object at a brand new position.
    new = Detection(
        class_name="bottle", class_id=42, confidence=0.9,
        bbox_xyxy=(0, 0, 50, 50), position=(10.0, 10.0, 0.0),
        embedding=tuple(_unit_vec(16, 3)),
        caption="a new bottle", ts=base_ts + 999,
    )
    cid, decision = store.upsert(new)
    assert decision.action == "insert"
    assert store.count == 4
    assert store.get_by_id(cid).class_name == "bottle"
