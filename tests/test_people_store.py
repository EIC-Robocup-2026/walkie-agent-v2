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


# ---------------------------------------------------------------------------
# Two-modality fused recognition (face + appearance) — design by Chalk (EIC).
# Appearance vectors live in a second collection keyed by the same ids.
# ---------------------------------------------------------------------------

# Attire vectors: Alice wears red, Bob wears blue. A second sighting of
# Alice's outfit is close to hers and far from Bob's.
ALICE_ATTIRE = _unit(1.0, 0.0, 0.1)
ALICE_ATTIRE_2 = _unit(0.95, 0.05, 0.15)
BOB_ATTIRE = _unit(0.0, 1.0, 0.1)
STRANGER_ATTIRE = _unit(0.1, 0.1, 1.0)


def test_fused_face_and_appearance_match(store):
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    store.enroll("Bob", "milk", BOB, app_embedding=BOB_ATTIRE)
    rec = store.recognize_fused(ALICE_2, ALICE_ATTIRE_2, face_confidence=0.95)
    assert rec is not None and rec.name == "Alice"
    assert rec.matched_by == "face+appearance"
    assert rec.similarity is not None and rec.similarity > 0.9


def test_appearance_only_match_when_no_face(store):
    """A guest facing away (no face embedding) is still found by attire."""
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    store.enroll("Bob", "milk", BOB, app_embedding=BOB_ATTIRE)
    rec = store.recognize_fused(None, ALICE_ATTIRE_2)
    assert rec is not None and rec.name == "Alice"
    assert rec.matched_by == "appearance"


def test_face_only_fallback_when_person_has_no_attire(store):
    """Enrolled without an appearance vector → face alone still matches."""
    store.enroll("Alice", "cola", ALICE)  # no app_embedding stored
    rec = store.recognize_fused(ALICE_2, ALICE_ATTIRE_2, face_confidence=0.95)
    assert rec is not None and rec.name == "Alice"
    assert rec.matched_by == "face"


def test_fused_rejects_stranger(store):
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    assert store.recognize_fused(CAROL, STRANGER_ATTIRE, face_confidence=0.95) is None
    assert store.recognize_fused(None, STRANGER_ATTIRE) is None


def test_low_face_confidence_leans_on_appearance(store):
    """With an unreliable face detection, attire dominates (w_face = 0)."""
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    # face vector says "not Alice" but the detection is junk; attire says Alice
    rec = store.recognize_fused(CAROL, ALICE_ATTIRE_2, face_confidence=0.1)
    assert rec is not None and rec.name == "Alice"
    assert rec.matched_by == "appearance"


def test_fused_min_score_is_respected(store):
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    assert store.recognize_fused(ALICE_2, ALICE_ATTIRE_2, face_confidence=0.95, min_score=0.999) is None


def test_fused_empty_store_and_empty_query(store):
    assert store.recognize_fused(ALICE, ALICE_ATTIRE) is None  # empty store
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    assert store.recognize_fused(None, None) is None  # nothing to match on


def test_reenroll_replaces_attire_latest_wins(store):
    """Clothing changes between sessions — the newest attire vector wins."""
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    store.enroll("Alice", "cola", ALICE_2, app_embedding=BOB_ATTIRE)  # changed outfit
    rec = store.recognize_fused(None, BOB_ATTIRE)
    assert rec is not None and rec.name == "Alice"
    # the old outfit no longer matches her strongly enough on its own
    old = store.recognize_fused(None, STRANGER_ATTIRE)
    assert old is None


# ---------------------------------------------------------------------------
# Conversation notes — what the guest told the robot (shown in the DB viewer).
# ---------------------------------------------------------------------------


def test_add_note_appends_and_survives_reenroll(store):
    store.enroll("Alice", "cola", ALICE)
    store.add_note("Alice", "from Bangkok")
    rec = store.add_note("Alice", "likes football")
    assert rec.notes == "from Bangkok\nlikes football"
    # notes survive a re-enrollment (face refresh must not wipe the chat memory)
    store.enroll("Alice", "cola", ALICE_2)
    assert store.get("Alice").notes == "from Bangkok\nlikes football"


def test_add_note_unknown_person_returns_none(store):
    assert store.add_note("Nobody", "anything") is None


def test_add_note_blank_is_ignored(store):
    store.enroll("Alice", "cola", ALICE)
    rec = store.add_note("Alice", "   ")
    assert rec is not None and rec.notes == ""


def test_add_note_caps_at_max_notes(store):
    store.enroll("Alice", "cola", ALICE)
    for i in range(6):
        store.add_note("Alice", f"fact {i}", max_notes=4)
    notes = store.get("Alice").notes.split("\n")
    assert notes == ["fact 2", "fact 3", "fact 4", "fact 5"]


def test_add_note_is_case_insensitive_on_name(store):
    store.enroll("John Smith", "water", BOB)
    assert store.add_note("john smith", "plays guitar") is not None
    assert store.get("John Smith").notes == "plays guitar"


def test_clear_empties_both_collections(store):
    store.enroll("Alice", "cola", ALICE, app_embedding=ALICE_ATTIRE)
    store.clear()
    assert store.count() == 0
    assert store.recognize_fused(None, ALICE_ATTIRE) is None
