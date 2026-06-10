"""Unit tests for PeopleStore — synthetic unit vectors, in-memory ChromaDB.

No server, no faces — just the enroll / recognize / centroid logic that the
face re-ID slice depends on.
"""

import math
from pathlib import Path

import pytest
from PIL import Image

from perception import PeopleStore, PersonRecord


def _unit(*v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# Three well-separated unit vectors standing in for three people's faces.
ALICE = _unit(1.0, 0.05, 0.0)
ALICE_2 = _unit(0.97, 0.10, 0.05)   # a second shot of Alice (close to ALICE)
BOB = _unit(0.0, 1.0, 0.05)
CAROL = _unit(0.0, 0.05, 1.0)


@pytest.fixture
def store(tmp_path):
    # A unique on-disk dir per test → true isolation. (chromadb caches in-memory
    # EphemeralClients by settings, so persist_dir=None would leak state between
    # tests; a fresh tmp_path each test avoids that.)
    return PeopleStore(persist_dir=tmp_path / "people", embedding_model="test-model")


def test_enroll_then_recognize(store):
    store.enroll("Alice", "cola", ALICE)
    store.enroll("Bob", "milk", BOB)

    rec = store.recognize(ALICE_2)
    assert rec is not None
    assert rec.name == "Alice"
    assert rec.drink == "cola"
    assert rec.distance is not None and rec.distance < 0.4
    assert rec.similarity == pytest.approx(1.0 - rec.distance)


def test_recognize_returns_none_for_stranger(store):
    store.enroll("Alice", "cola", ALICE)
    store.enroll("Bob", "milk", BOB)
    # Carol was never enrolled and is far from both
    assert store.recognize(CAROL, max_distance=0.4) is None


def test_recognize_empty_store_is_none(store):
    assert store.recognize(ALICE) is None


def test_threshold_is_respected(store):
    store.enroll("Alice", "cola", ALICE)
    # A strict threshold rejects even a real-but-imperfect match
    assert store.recognize(ALICE_2, max_distance=0.0) is None
    assert store.recognize(ALICE_2, max_distance=0.4) is not None


def test_reenroll_same_name_updates_not_duplicates(store):
    store.enroll("Alice", "cola", ALICE)
    store.enroll("Alice", "orange juice", ALICE_2)  # corrected drink + 2nd frame
    assert store.count() == 1
    alice = store.get("Alice")
    assert alice.drink == "orange juice"
    assert alice.enrollments == 2
    # the stored vector is the renormalized centroid (still ~unit length)
    assert math.isclose(sum(x * x for x in alice.embedding) ** 0.5, 1.0, abs_tol=1e-6)


def test_get_is_case_insensitive(store):
    store.enroll("John Smith", "water", BOB)
    assert store.get("john smith") is not None
    assert store.get("JOHN SMITH").name == "John Smith"


def test_list_people_orders_recent_first(store):
    store.enroll("Alice", "cola", ALICE, ts=100.0)
    store.enroll("Bob", "milk", BOB, ts=200.0)
    names = [p.name for p in store.list_people()]
    assert names == ["Bob", "Alice"]


def test_attributes_preserved_and_provenance_stamped(store):
    rec = store.enroll("Alice", "cola", ALICE, attributes="blue shirt, glasses")
    assert rec.attributes == "blue shirt, glasses"
    assert rec.embedding_model == "test-model"


def test_clear_empties_the_store(store):
    store.enroll("Alice", "cola", ALICE)
    store.clear()
    assert store.count() == 0
    assert store.get("Alice") is None


def test_enroll_validates_inputs(store):
    with pytest.raises(ValueError):
        store.enroll("", "cola", ALICE)
    with pytest.raises(ValueError):
        store.enroll("Alice", "cola", [])


def test_enroll_archives_face_crop_when_frames_dir_set(tmp_path):
    store = PeopleStore(
        persist_dir=tmp_path / "p", embedding_model="m", frames_dir=tmp_path / "pf"
    )
    img = Image.new("RGB", (200, 200), (10, 20, 30))
    rec = store.enroll("Alice", "cola", ALICE, frame=img, face_bbox_xyxy=(50, 50, 150, 150))
    assert rec.frame_ref is not None
    assert Path(rec.frame_ref).exists()
    # the crop survives a fresh read and a re-enrollment keeps a frame
    assert store.get("Alice").frame_ref == rec.frame_ref


def test_enroll_without_frames_dir_has_no_frame_ref(store):
    rec = store.enroll("Bob", "milk", BOB, frame=Image.new("RGB", (10, 10)), face_bbox_xyxy=(0, 0, 5, 5))
    assert rec.frame_ref is None
