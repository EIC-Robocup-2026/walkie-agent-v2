"""Tests for the human sub-agent's two-modality (face + attire) recognition.

Fakes the face, appearance, and pose clients + a real PeopleStore on a tmp dir,
exercising the fused enroll→recognize flow (pipeline design by Chalk, EIC team)
without a server or camera — including the appearance-only fallback when no
face is visible, and graceful degradation when the appearance route is absent
or failing.
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


ALICE_FACE = _unit(1.0, 0.05, 0.0)
ALICE_FACE_2 = _unit(0.97, 0.10, 0.05)
ALICE_ATTIRE = _unit(1.0, 0.0, 0.1)
ALICE_ATTIRE_2 = _unit(0.95, 0.05, 0.15)
STRANGER_ATTIRE = _unit(0.1, 0.1, 1.0)


def _face(emb, score=0.9, box=(30, 30, 50, 50)):
    w, h = box[2] - box[0], box[3] - box[1]
    return NS(embedding=emb, det_score=score, bbox_xyxy=box, area=lambda: w * h)


def _pose_person(cxcywh=(50, 50, 80, 80)):
    return NS(bbox=cxcywh)


class _FakeFaceClient:
    def __init__(self):
        self.queued = []

    def embed(self, image):
        return self.queued


class _FakeAppearanceClient:
    """Returns the queued attire vector; records crops; raises if `boom`."""

    def __init__(self):
        self.queued = None
        self.boom = None
        self.crops = []

    def embed(self, image):
        if self.boom:
            raise self.boom
        self.crops.append(image.size)
        return self.queued


class _FakePoseClient:
    def __init__(self):
        self.people = []

    def estimate(self, image):
        return self.people


@pytest.fixture
def ctx(tmp_path):
    face = _FakeFaceClient()
    appearance = _FakeAppearanceClient()
    pose = _FakePoseClient()
    walkieAI = NS(face_recognition=face, appearance=appearance, pose_estimation=pose)
    walkie = NS(camera=NS(capture_pil=lambda: Image.new("RGB", (100, 100))))
    store = PeopleStore(persist_dir=tmp_path / "p", embedding_model="test")
    tools = {t.name: t for t in make_human_tools(walkie, walkieAI, people_store=store)}
    return NS(face=face, appearance=appearance, pose=pose, store=store, tools=tools)


def _enroll_alice(ctx):
    ctx.face.queued = [_face(ALICE_FACE)]
    ctx.appearance.queued = ALICE_ATTIRE
    ctx.pose.people = [_pose_person()]
    return ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})


def test_enroll_stores_attire_and_crops_to_person_box(ctx):
    out = _enroll_alice(ctx)
    assert "Alice" in out and "attire" in out.lower()
    # the embed got the pose person crop (80x80), not the 100x100 full frame
    assert ctx.appearance.crops == [(80, 80)]
    # the attire vector is queryable on its own (guest facing away)
    rec = ctx.store.recognize_fused(None, ALICE_ATTIRE_2)
    assert rec is not None and rec.name == "Alice"


def test_recognize_fuses_both_modalities(ctx):
    _enroll_alice(ctx)
    ctx.face.queued = [_face(ALICE_FACE_2)]
    ctx.appearance.queued = ALICE_ATTIRE_2
    out = ctx.tools["recognize_person"].invoke({})
    assert "Alice" in out and "face+appearance" in out


def test_recognize_falls_back_to_appearance_when_no_face(ctx):
    _enroll_alice(ctx)
    ctx.face.queued = []  # guest turned away
    ctx.appearance.queued = ALICE_ATTIRE_2
    ctx.pose.people = [_pose_person()]
    out = ctx.tools["recognize_person"].invoke({})
    assert "probably Alice" in out
    assert "appearance" in out and "less certain" in out


def test_no_face_no_appearance_match_is_honest(ctx):
    _enroll_alice(ctx)
    ctx.face.queued = []
    ctx.appearance.queued = STRANGER_ATTIRE
    ctx.pose.people = [_pose_person()]
    out = ctx.tools["recognize_person"].invoke({})
    assert "nobody matches" in out.lower()


def test_no_face_and_no_people_in_view(ctx):
    _enroll_alice(ctx)
    ctx.face.queued = []
    ctx.pose.people = []
    assert "No clear face" in ctx.tools["recognize_person"].invoke({})


def test_appearance_failure_degrades_to_face_only(ctx):
    """Route missing / server error → enroll and recognize work as before."""
    ctx.face.queued = [_face(ALICE_FACE)]
    ctx.appearance.boom = RuntimeError("404 no such route")
    ctx.pose.people = [_pose_person()]
    out = ctx.tools["enroll_person"].invoke({"name": "Alice", "drink": "cola"})
    assert "Alice" in out and "attire" not in out.lower()

    ctx.face.queued = [_face(ALICE_FACE_2)]
    rec = ctx.tools["recognize_person"].invoke({})
    assert "Alice" in rec


def test_appearance_disabled_via_env(ctx, monkeypatch):
    monkeypatch.setenv("HUMAN_APPEARANCE_ENABLED", "0")
    _enroll_alice(ctx)
    assert ctx.appearance.crops == []  # never called
    ctx.face.queued = []
    ctx.pose.people = [_pose_person()]
    assert "No clear face" in ctx.tools["recognize_person"].invoke({})
