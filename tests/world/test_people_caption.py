"""Unified people store: attire-caption semantic + lexical re-ID, and the new
PersonRecord fields. Uses an ephemeral (in-memory) chromadb store."""

from __future__ import annotations

import pytest

from walkie_world.people.store import PeopleStore


@pytest.fixture()
def store():
    return PeopleStore(persist_dir=None)  # ephemeral, unique collections per instance


def test_caption_embedding_match(store):
    store.enroll(
        "alice", "cola", [1.0, 0.0, 0.0],
        appearance_caption="a person in a red shirt",
        appearance_caption_embedding=[1.0, 0.0],
    )
    store.enroll(
        "bob", "milk", [0.0, 1.0, 0.0],
        appearance_caption="a person in a blue jacket",
        appearance_caption_embedding=[0.0, 1.0],
    )
    hit = store.find_by_caption_embedding([0.95, 0.05])
    assert hit is not None and hit.id == "alice"
    assert hit.matched_by == "appearance_caption"
    other = store.find_by_caption_embedding([0.05, 0.95])
    assert other is not None and other.id == "bob"


def test_caption_lexical_fallback(store):
    store.enroll("alice", "cola", [1.0, 0.0],
                 appearance_caption="a person in a red shirt and glasses")
    store.enroll("bob", "milk", [0.0, 1.0],
                 appearance_caption="a person in a blue jacket")
    assert store.find_by_caption_lexical("red shirt").id == "alice"
    assert store.find_by_caption_lexical("blue jacket").id == "bob"
    assert store.find_by_caption_lexical("green hat") is None  # no token overlap


def test_new_record_fields_roundtrip(store):
    store.enroll(
        "alice", "cola", [1.0, 0.0],
        appearance_caption="a person in a red shirt",
        last_seen_pose=(1.5, 2.5, 0.5),
        last_seen_room="kitchen",
        pose_label="waving",
        seat="seat_1",
    )
    rec = store.get("alice")
    assert rec.appearance_caption == "a person in a red shirt"
    assert rec.last_seen_pose == (1.5, 2.5, 0.5)
    assert rec.last_seen_room == "kitchen"
    assert rec.pose_label == "waving"
    assert rec.seat == "seat_1"


def test_fields_carried_forward_on_reenroll(store):
    store.enroll("alice", "cola", [1.0, 0.0],
                 appearance_caption="a person in a red shirt",
                 last_seen_room="kitchen")
    # Re-enroll without re-supplying the caption/room: they must persist.
    store.enroll("alice", "cola", [0.9, 0.1])
    rec = store.get("alice")
    assert rec.appearance_caption == "a person in a red shirt"
    assert rec.last_seen_room == "kitchen"
    assert rec.enrollments == 2


def test_attire_only_enrollment_no_face(store):
    """Restaurant has no face: enroll by attire alone (empty face embedding) and still
    re-identify by caption. A zero placeholder face is stored, so recognize() won't
    false-match it."""
    rec = store.enroll(
        "", "", [], person_id="customer-1",
        appearance_caption="a person in a red shirt",
        appearance_caption_embedding=[1.0, 0.0],
        last_seen_pose=(2.0, 3.0, 0.0),
    )
    assert rec.id == "customer-1"
    assert store.count() == 1
    hit = store.find_by_caption_embedding([0.95, 0.05])
    assert hit is not None and hit.id == "customer-1"
    assert hit.last_seen_pose == (2.0, 3.0, 0.0)
    # A zero-norm placeholder face must not be returned by face recognition.
    assert store.recognize([0.0, 1.0]) is None


def test_attire_only_requires_some_vector(store):
    with pytest.raises(ValueError):
        store.enroll("", "", [], person_id="nobody")  # no face, no appearance vectors


def test_clear_drops_all_three_collections(store):
    store.enroll("alice", "cola", [1.0, 0.0],
                 app_embedding=[0.5, 0.5],
                 appearance_caption="a person in a red shirt",
                 appearance_caption_embedding=[1.0, 0.0])
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    assert store.find_by_caption_embedding([1.0, 0.0]) is None
    assert store.appearance_vectors() == {}
