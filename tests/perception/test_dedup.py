"""Phase 3 unit tests for the dedup decision logic.

Covers every branch in ``perception/dedup.py::classify`` plus the
upsert-side integration in ``SceneStore.upsert`` (so the test is end-to-end
through the store, not just the pure function). Tests share an in-memory
ChromaDB (one fresh store per test) and a FakeEmbedder seeded with
override vectors so we can pin cosine-similarity values precisely.
"""

from __future__ import annotations

import math
import os
import time

import pytest

from perception import Detection, SceneStore
from perception.dedup import classify
from perception.mocks import FakeEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return tuple(x / norm for x in vec) if norm > 0 else tuple(vec)


def _detection(
    *,
    class_name="chair",
    position=(0.0, 0.0, 0.0),
    embedding=(1.0, 0.0, 0.0, 0.0),
    confidence=0.9,
    ts=1000.0,
    bbox=(0, 0, 100, 100),
    caption="a chair",
    class_id=56,
) -> Detection:
    return Detection(
        class_name=class_name,
        class_id=class_id,
        confidence=confidence,
        bbox_xyxy=bbox,
        position=position,
        embedding=_unit(embedding),
        caption=caption,
        ts=ts,
    )


@pytest.fixture
def store(tmp_path):
    # Per-test temp dir — chromadb's EphemeralClient shares process-wide
    # state, so a fresh PersistentClient under tmp_path is the cleanest
    # way to get a truly isolated store per test.
    return SceneStore(persist_dir=tmp_path / "chroma")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_01_empty_store_inserts(store):
    det = _detection()
    cid, decision = store.upsert(det)
    assert decision.action == "insert"
    assert store.count == 1
    assert store.get_by_id(cid).sightings == 1


def test_02_same_object_slight_drift_updates(store):
    a = _detection(position=(1.0, 1.0, 0.0))
    b = _detection(position=(1.1, 1.0, 0.0))  # 0.1m drift, same embedding
    id_a, _ = store.upsert(a)
    id_b, decision = store.upsert(b)
    assert decision.action == "update"
    assert id_b == id_a
    assert store.count == 1
    entry = store.get_by_id(id_a)
    assert entry.sightings == 2
    # Running mean position is between a and b.
    assert 1.0 < entry.position[0] < 1.1


def test_03_two_similar_objects_far_apart_both_insert(store):
    a = _detection(position=(0.0, 0.0, 0.0))
    b = _detection(position=(2.0, 2.0, 0.0))  # > SPATIAL_RADIUS (0.5m)
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "insert"
    assert store.count == 2


def test_04_same_position_different_class_inserts(store):
    a = _detection(class_name="chair", position=(0.5, 0.5, 0.0))
    b = _detection(class_name="table", position=(0.5, 0.5, 0.0))
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "insert"
    assert store.count == 2


def test_05_spatial_near_miss_embedding_identical_updates(store):
    # 0.4m apart (< SPATIAL_RADIUS 0.5m), cosine ≈ 1.0 → should merge via HIGH gate.
    a = _detection(position=(0.0, 0.0, 0.0), embedding=(1.0, 0.0, 0.0, 0.0))
    b = _detection(position=(0.4, 0.0, 0.0), embedding=(1.0, 0.0, 0.0, 0.0))
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "update"
    assert "HIGH" in decision.reason or "≥ EMB_SIM_HIGH" in decision.reason
    assert store.count == 1


def test_06_spatial_near_miss_embedding_far_inserts(store):
    # 0.4m apart, orthogonal embeddings → cosine 0.0 → both gates fail.
    a = _detection(position=(0.0, 0.0, 0.0), embedding=(1.0, 0.0, 0.0, 0.0))
    b = _detection(position=(0.4, 0.0, 0.0), embedding=(0.0, 1.0, 0.0, 0.0))
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "insert"
    assert store.count == 2


def test_07_tight_radius_failsafe_merges_on_mid_similarity(store):
    # Within TIGHT_RADIUS (0.2m), cosine between LOW (0.65) and HIGH (0.85)
    # → should merge via the failsafe.
    e1 = (1.0, 0.0, 0.0, 0.0)
    # Construct e2 with cosine ≈ 0.70 against e1.
    cos = 0.70
    s = math.sqrt(1 - cos * cos)
    e2 = (cos, s, 0.0, 0.0)
    a = _detection(position=(0.0, 0.0, 0.0), embedding=e1)
    b = _detection(position=(0.15, 0.0, 0.0), embedding=e2)
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "update"
    assert "TIGHT_RADIUS" in decision.reason
    assert store.count == 1


def test_08_disappear_then_reappear_merges(store):
    a = _detection(position=(0.0, 0.0, 0.0), ts=1000.0)
    id_a, _ = store.upsert(a)
    # Pretend an hour passes — same object reappears.
    b = _detection(position=(0.05, 0.05, 0.0), ts=1000.0 + 3600)
    id_b, decision = store.upsert(b)
    assert decision.action == "update"
    assert id_b == id_a
    entry = store.get_by_id(id_a)
    assert entry.sightings == 2
    assert entry.last_seen_ts == 1000.0 + 3600
    assert entry.first_seen_ts == 1000.0


def test_09_multiple_candidates_closest_wins(store):
    # Two existing chairs at orthogonal embeddings so they don't merge into
    # each other. The new detection looks like A (same embedding, very close)
    # and is also within radius of B (further away, orthogonal). Closest-
    # wins means A gets the merge and B is untouched.
    a = _detection(position=(0.0, 0.0, 0.0), embedding=(1.0, 0.0, 0.0, 0.0))
    b = _detection(position=(0.45, 0.0, 0.0), embedding=(0.0, 1.0, 0.0, 0.0))
    id_a, _ = store.upsert(a)
    id_b, _ = store.upsert(b)
    assert store.count == 2  # sanity: orthogonal embeddings kept them apart

    # Probe at (0.05, 0.0, 0.0) — distance to A is 0.05m, to B is 0.40m,
    # both within the 0.5m radius. Embedding matches A.
    c = _detection(position=(0.05, 0.0, 0.0), embedding=(1.0, 0.0, 0.0, 0.0))
    target_id, decision = store.upsert(c)
    assert decision.action == "update"
    assert target_id == id_a
    # B is untouched.
    assert store.get_by_id(id_b).sightings == 1


def test_10_threshold_env_override(monkeypatch, store):
    # With the default 0.5m radius this would be an UPDATE (per test 02).
    # Shrink the radius to 0.05m → the candidate is now outside the search
    # ball, so we fall through to INSERT.
    monkeypatch.setenv("SCENE_DEDUP_RADIUS_M", "0.05")
    a = _detection(position=(1.0, 1.0, 0.0))
    b = _detection(position=(1.1, 1.0, 0.0))
    store.upsert(a)
    _, decision = store.upsert(b)
    assert decision.action == "insert"
    assert store.count == 2


def test_11_classify_rejects_cross_class_candidates():
    # Pure-function test of the precondition guard in classify().
    from perception.types import SceneEntry

    fake_entry = SceneEntry(
        id="chair:0:0:0:abc",
        class_name="chair",
        class_id=56,
        position=(0.0, 0.0, 0.0),
        position_frame="map",
        position_conf=0.9,
        caption="a chair",
        bbox_last_xyxy=(0, 0, 10, 10),
        frame_ref=None,
        first_seen_ts=1.0,
        last_seen_ts=1.0,
        sightings=1,
        embedding=(1.0, 0.0, 0.0, 0.0),
        embedding_model="fake",
        embedding_dim=4,
    )
    new = _detection(class_name="table")
    with pytest.raises(ValueError, match="cross-class"):
        classify(new, [fake_entry])
