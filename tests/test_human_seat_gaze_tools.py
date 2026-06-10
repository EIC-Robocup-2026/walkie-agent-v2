"""Tests for the human sub-agent's seat (C5) and gaze (C6) tools.

Fakes object detection + pose estimation so find_empty_seat / locate_person are
exercised without a camera or server. Bbox conventions match the real clients:
object detector → xyxy, pose person bbox → (cx, cy, w, h).
"""

import math
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from agents.human_agent.tools import make_human_tools
from perception import PeopleStore


def _unit(*v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


ALICE = _unit(1.0, 0.05, 0.0)


def _obj(cls, box_xyxy):
    return NS(class_name=cls, confidence=0.9, bbox=box_xyxy)


def _person(box_cxcywh):
    return NS(bbox=box_cxcywh)


def _face(emb, score=0.9, box=(0, 0, 100, 200)):
    w, h = box[2] - box[0], box[3] - box[1]
    return NS(embedding=emb, det_score=score, bbox_xyxy=box, area=lambda: w * h)


def _make(objects=(), people=(), faces=(), store=None, img_size=(200, 100)):
    walkieAI = NS(
        object_detection=NS(detect=lambda im: list(objects)),
        pose_estimation=NS(estimate=lambda im: list(people)),
        face_recognition=NS(embed=lambda im: list(faces)),
    )
    walkie = NS(camera=NS(capture_pil=lambda: Image.new("RGB", img_size)))
    return {t.name: t for t in make_human_tools(walkie, walkieAI, people_store=store)}


# --- find_empty_seat (C5) -------------------------------------------------


def test_empty_seat_reports_unoccupied_chair():
    chairs = [_obj("chair", (0, 0, 50, 100)), _obj("chair", (150, 0, 200, 100))]
    # a person sitting on the left chair (bbox covers it)
    people = [_person((25, 50, 50, 100))]
    tools = _make(objects=chairs, people=people)
    out = tools["find_empty_seat"].invoke({})
    assert "1 free seat(s) of 2" in out
    assert "chair" in out and "right" in out  # the free one is on the right


def test_all_seats_occupied():
    chairs = [_obj("chair", (0, 0, 50, 100))]
    people = [_person((25, 50, 50, 100))]
    out = _make(objects=chairs, people=people)["find_empty_seat"].invoke({})
    assert "occupied" in out


def test_no_seats_in_view():
    out = _make(objects=[_obj("bottle", (0, 0, 10, 10))], people=[])["find_empty_seat"].invoke({})
    assert "don't see any seats" in out


def test_seat_with_no_people_is_free():
    out = _make(objects=[_obj("couch", (10, 10, 90, 90))], people=[])["find_empty_seat"].invoke({})
    assert "1 free seat(s) of 1" in out


def test_seat_vision_error_is_caught():
    walkieAI = NS(
        object_detection=NS(detect=lambda im: (_ for _ in ()).throw(RuntimeError("boom"))),
        pose_estimation=NS(estimate=lambda im: []),
        face_recognition=NS(embed=lambda im: []),
    )
    walkie = NS(camera=NS(capture_pil=lambda: Image.new("RGB", (200, 100))))
    tools = {t.name: t for t in make_human_tools(walkie, walkieAI)}
    assert "vision error" in tools["find_empty_seat"].invoke({})


# --- locate_person (C6) ---------------------------------------------------


def test_locate_nearest_person_left():
    # one person on the far left → turn left
    out = _make(people=[_person((20, 50, 40, 80))])["locate_person"].invoke({})
    assert "nearest person" in out and "left" in out


def test_locate_picks_largest_as_nearest():
    far = _person((180, 50, 20, 40))   # small (far), right
    near = _person((30, 50, 80, 90))   # large (near), left
    out = _make(people=[far, near])["locate_person"].invoke({})
    assert "left" in out  # the nearest (largest) one is on the left


def test_locate_no_one_in_view():
    assert "No one is in view" in _make(people=[])["locate_person"].invoke({})


def test_locate_named_guest_by_face(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("Alice", "cola", ALICE)
    tools = _make(faces=[_face(ALICE, box=(140, 10, 190, 90))], store=store)
    out = tools["locate_person"].invoke({"name": "Alice"})
    assert "Alice" in out and "right" in out  # face on the right side


def test_locate_named_guest_not_present(tmp_path):
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="m")
    store.enroll("Alice", "cola", ALICE)
    tools = _make(faces=[], store=store)
    assert "don't see Alice" in tools["locate_person"].invoke({"name": "Alice"})


def test_locate_named_without_store_falls_back_message():
    out = _make()["locate_person"].invoke({"name": "Alice"})
    assert "Face memory is off" in out
