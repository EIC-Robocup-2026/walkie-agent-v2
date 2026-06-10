"""Tests for the human sub-agent's face tools (enroll / recognize / list).

Uses a fake face-recognition client (preset embeddings) + a real PeopleStore on
a tmp dir, so the enroll→recognize flow is exercised without a server or camera.
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
ALICE_2 = _unit(0.97, 0.10, 0.05)
BOB = _unit(0.0, 1.0, 0.05)
STRANGER = _unit(0.0, 0.05, 1.0)


def _face(emb, score=0.9, box=(0, 0, 100, 200)):
    w, h = box[2] - box[0], box[3] - box[1]
    return NS(embedding=emb, det_score=score, bbox_xyxy=box, area=lambda: w * h)


class _FakeFaceClient:
    """Returns whatever faces the test queued; or raises if `boom` is set."""

    def __init__(self):
        self.queued = []
        self.boom = None

    def embed(self, image):
        if self.boom:
            raise self.boom
        return self.queued


@pytest.fixture
def ctx(tmp_path):
    face = _FakeFaceClient()
    walkieAI = NS(face_recognition=face)
    walkie = NS(camera=NS(capture_pil=lambda: Image.new("RGB", (8, 8))))
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="test")
    tools = {t.name: t for t in make_human_tools(walkie, walkieAI, people_store=store)}
    return NS(face=face, store=store, tools=tools)


def test_enroll_then_recognize(ctx):
    ctx.face.queued = [_face(ALICE)]
    out = ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    assert "Alice" in out and "cola" in out
    assert ctx.store.count() == 1

    ctx.face.queued = [_face(ALICE_2)]
    rec = ctx.tools["recognize_person"].invoke({})
    assert "Alice" in rec and "cola" in rec


def test_recognize_marks_stranger_unknown(ctx):
    ctx.face.queued = [_face(ALICE)]
    ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    # two faces in view: Alice (nearest/biggest) + an unenrolled stranger
    ctx.face.queued = [_face(ALICE_2, box=(0, 0, 120, 240)), _face(STRANGER, box=(0, 0, 40, 40))]
    rec = ctx.tools["recognize_person"].invoke({})
    assert "person 1: Alice" in rec       # biggest face first
    assert "person 2: unknown" in rec


def test_enroll_no_face_is_graceful(ctx):
    ctx.face.queued = []
    out = ctx.tools["enroll_person"].invoke({"name": "Carl", "drink": "tea"})
    assert "don't see a clear face" in out
    assert ctx.store.count() == 0


def test_low_det_score_face_is_ignored(ctx):
    ctx.face.queued = [_face(ALICE, score=0.2)]  # below FACE_MIN_DET_SCORE default 0.5
    out = ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    assert "don't see a clear face" in out


def test_recognize_before_any_enrollment(ctx):
    ctx.face.queued = [_face(ALICE)]
    assert "haven't remembered anyone" in ctx.tools["recognize_person"].invoke({})


def test_list_known_people(ctx):
    ctx.face.queued = [_face(ALICE)]
    ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    ctx.face.queued = [_face(BOB)]
    ctx.tools["enroll_person"].invoke({"name": "Bob", "drink": "milk"})
    out = ctx.tools["list_known_people"].invoke({})
    assert "Alice" in out and "Bob" in out and "2 remembered" in out


def test_face_service_error_is_caught(ctx):
    ctx.face.boom = RuntimeError("server down")
    out = ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    assert "face service error" in out and "server down" in out


def test_remember_person_detail_then_list_shows_it(ctx):
    ctx.face.queued = [_face(ALICE)]
    ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    out = ctx.tools["remember_person_detail"].invoke(
        {"name": "Alice", "detail": "from Bangkok"}
    )
    assert "Noted" in out and "from Bangkok" in out
    ctx.tools["remember_person_detail"].invoke(
        {"name": "Alice", "detail": "likes football"}
    )
    listed = ctx.tools["list_known_people"].invoke({})
    assert "from Bangkok" in listed and "likes football" in listed


def test_remember_detail_requires_enrollment(ctx):
    out = ctx.tools["remember_person_detail"].invoke(
        {"name": "Ghost", "detail": "anything"}
    )
    assert "enroll" in out.lower()
    assert ctx.store.count() == 0


def test_tools_report_off_without_store(tmp_path):
    walkieAI = NS(face_recognition=_FakeFaceClient())
    walkie = NS(camera=NS(capture_pil=lambda: Image.new("RGB", (8, 8))))
    tools = {t.name: t for t in make_human_tools(walkie, walkieAI, people_store=None)}
    assert "off" in tools["enroll_person"].invoke({"name": "A", "drink": "b"}).lower()
    assert "off" in tools["recognize_person"].invoke({}).lower()
    assert "off" in tools["list_known_people"].invoke({}).lower()
    assert "off" in tools["remember_person_detail"].invoke({"name": "A", "detail": "x"}).lower()
