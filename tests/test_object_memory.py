"""Unit tests for the shared long-term lookup helper.

Covers the quality filters added so "where is X?" only returns navigable
matches: the confidence floor and the spatial ("near me") restriction.
"""

from __future__ import annotations

from agents.core.object_memory import lookup_object_in_memory
from perception import Detection, SceneStore
from perception.mocks import FakeEmbedder


def _unit(dim: int, axis: int) -> list[float]:
    v = [0.0] * dim
    v[axis] = 1.0
    return v


def _mug(store, *, conf, pos, caption):
    store.upsert(
        Detection(
            class_name="mug",
            class_id=0,
            confidence=conf,
            bbox_xyxy=(0, 0, 9, 9),
            position=pos,
            embedding=tuple(_unit(16, 0)),
            caption=caption,
            ts=1000.0,
        )
    )


def _store(tmp_path):
    emb = FakeEmbedder(
        dim=16,
        override_text={
            # Documents are caption-led (no class prefix).
            "a white mug": _unit(16, 0),
            "a blurry mug": _unit(16, 0),
            "coffee mug": _unit(16, 0),
        },
    )
    store = SceneStore(persist_dir=tmp_path / "chroma", embedder=emb)
    # Two mugs >0.4m apart (no spatial merge), differing in confidence.
    _mug(store, conf=0.9, pos=(0.0, 0.0, 0.0), caption="a white mug")
    _mug(store, conf=0.1, pos=(5.0, 5.0, 0.0), caption="a blurry mug")
    assert store.count == 2
    return store


def test_lookup_confidence_floor_drops_weak_positions(tmp_path):
    store = _store(tmp_path)
    # No floor → both mugs come back.
    both = lookup_object_in_memory("coffee mug", scene_store=store, min_position_conf=0.0)
    assert "a white mug" in both and "a blurry mug" in both
    # Floor at 0.5 → only the confident one survives.
    filtered = lookup_object_in_memory(
        "coffee mug", scene_store=store, min_position_conf=0.5
    )
    assert "a white mug" in filtered
    assert "a blurry mug" not in filtered


def test_lookup_near_me_radius_restricts_to_vicinity(tmp_path):
    store = _store(tmp_path)
    # A 1m ball around the origin only contains the mug at (0,0,0).
    near = lookup_object_in_memory(
        "coffee mug",
        scene_store=store,
        within_radius_of=(0.0, 0.0, 0.0),
        max_distance_m=1.0,
    )
    assert "a white mug" in near
    assert "a blurry mug" not in near


def test_lookup_all_filtered_reports_confidence_reason(tmp_path):
    store = _store(tmp_path)
    # Floor above every record → explicit "below the floor" message, not a bare
    # "no record" (which would wrongly imply the object was never seen).
    msg = lookup_object_in_memory("coffee mug", scene_store=store, min_position_conf=0.99)
    assert "confidence floor" in msg


def test_lookup_falls_back_to_keyword_when_embed_down(tmp_path, monkeypatch):
    store = _store(tmp_path)

    # Simulate the embedding server being down: every text embed raises.
    def boom(_text):
        raise RuntimeError("image-embed 503")

    monkeypatch.setattr(store._embedder, "embed_text", boom)

    msg = lookup_object_in_memory("white mug", scene_store=store)
    # The local keyword fallback still finds the white mug by word overlap, and
    # flags that the semantic path was unavailable.
    assert "a white mug" in msg
    assert "semantic search unavailable" in msg
