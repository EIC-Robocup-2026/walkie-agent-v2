"""Unit tests for descriptor-prompt detection + CLIP rerank in tasks.skills.grasp.

No robot, no AI server: the detector/embedder is a fake walkieAI client returning
canned detections + embeddings, and the snapshot is a real CameraSnapshot over a
synthetic flat-depth array (so mask_to_points lifts genuine optical-frame points).
We pin the pure pieces (prompt expansion, cosine ranking, cached text embed) and the
load-bearing selection rule: when several boxes are in view, the CLIP-best box is
chosen even when it is NOT the nearest — and the path degrades to nearest-wins when
no embeddings/target are available.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from interfaces.devices.camera import CameraSnapshot
from interfaces.perception.geometry import CameraPose, Intrinsics
from tasks.skills import grasp
from tasks.skills.grasp import (
    _GRASP_DESCRIPTORS,
    _cosine,
    _detect_and_lift,
    _detection_prompts,
    _rank_by_clip,
    _target_text_embedding,
)


@pytest.fixture(autouse=True)
def _clear_text_emb_cache():
    """The success-only text-embed cache is module-global — reset it per test."""
    grasp._TEXT_EMB_CACHE.clear()
    yield
    grasp._TEXT_EMB_CACHE.clear()


# ---------------------------------------------------------------------------
# Snapshot + fake-client scaffolding
# ---------------------------------------------------------------------------
def _intr(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def _snap(depth):
    h, w = depth.shape
    img = SimpleNamespace(size=(w, h))
    return CameraSnapshot(
        ts=0.0, img=img, depth=depth,
        cam=CameraPose(R=np.eye(3), t=np.zeros(3)), intr=_intr(), robot_pose=None,
    )


def _two_object_depth():
    """Flat scene at 4 m with a NEAR object (1 m) and a FAR object (2.5 m)."""
    depth = np.full((480, 640), 4.0, dtype=np.float32)
    depth[200:280, 100:180] = 1.0   # near object, left of centre
    depth[200:280, 460:540] = 2.5   # far object, right of centre
    return depth


def _mask(rows, cols):
    m = np.zeros((480, 640), dtype=np.uint8)
    m[rows[0]:rows[1], cols[0]:cols[1]] = 1
    return m


def _det(mask, *, confidence=0.9, embedding=None):
    return SimpleNamespace(mask=mask, confidence=confidence, embedding=embedding)


class _FakeImage:
    """Stand-in for ctx.walkieAI.image with call counters."""

    def __init__(self, dets, *, text_emb=None, embed_raises_until=0):
        self._dets = dets
        self._text_emb = text_emb
        self._embed_raises_until = embed_raises_until
        self.process_calls = 0
        self.detect_calls: list[list[str]] = []
        self.embed_text_calls: list[str] = []

    def process(self, img, *, detection=None, per_detection=None):
        self.process_calls += 1
        return SimpleNamespace(detection=list(self._dets))

    def detect(self, img, *, prompts=None, return_mask=False):
        self.detect_calls.append(list(prompts or []))
        return list(self._dets)

    def embed_text(self, text):
        self.embed_text_calls.append(text)
        if len(self.embed_text_calls) <= self._embed_raises_until:
            raise RuntimeError("server down")
        return list(self._text_emb) if self._text_emb is not None else []


def _ctx(image):
    return SimpleNamespace(walkieAI=SimpleNamespace(image=image))


# ---------------------------------------------------------------------------
# _detection_prompts
# ---------------------------------------------------------------------------
def test_detection_prompts_expands_known_item_target_first():
    out = _detection_prompts(["coke"])
    assert out[0] == "coke"                       # specific target stays first
    assert out == ["coke", *_GRASP_DESCRIPTORS["coke"]]


def test_detection_prompts_dedups_preserving_order():
    out = _detection_prompts(["coke", "can"])     # "can" also a descriptor
    assert out.count("can") == 1
    assert out[0] == "coke"


def test_detection_prompts_unknown_item_unchanged():
    assert _detection_prompts(["banana"]) == ["banana"]


def test_detection_prompts_empty_unchanged():
    assert _detection_prompts([]) == []


def test_detection_prompts_rerank_off_passes_through(monkeypatch):
    monkeypatch.setenv("WALKIE_GRASP_CLIP_RERANK", "0")
    assert _detection_prompts(["coke"]) == ["coke"]


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------
def test_cosine_identical_orthogonal_and_degenerate():
    assert _cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert _cosine([0, 0, 0], [1, 0, 0]) == 0.0   # zero norm
    assert _cosine([], [1]) == 0.0                # empty / mismatched
    assert _cosine([1, 0], [1, 0, 0]) == 0.0      # size mismatch


# ---------------------------------------------------------------------------
# _rank_by_clip
# ---------------------------------------------------------------------------
def test_rank_by_clip_orders_descending_by_cosine():
    target = [1.0, 0.0, 0.0]
    a = _det(None, embedding=[1.0, 0.0, 0.0])     # cos 1.0
    b = _det(None, embedding=[0.5, 0.5, 0.0])     # cos ~0.707
    ranked = _rank_by_clip([b, a], target, sim_floor=0.0)
    assert [d for d, _ in ranked] == [a, b]       # a (higher cos) first


def test_rank_by_clip_floor_excludes_low_matches():
    target = [1.0, 0.0, 0.0]
    a = _det(None, embedding=[1.0, 0.0, 0.0])     # cos 1.0
    b = _det(None, embedding=[0.0, 1.0, 0.0])     # cos 0.0
    ranked = _rank_by_clip([a, b], target, sim_floor=0.5)
    assert [d for d, _ in ranked] == [a]          # b below the floor


def test_rank_by_clip_no_text_emb_returns_empty():
    a = _det(None, embedding=[1.0, 0.0, 0.0])
    assert _rank_by_clip([a], None, sim_floor=0.0) == []
    assert _rank_by_clip([a], [], sim_floor=0.0) == []


def test_rank_by_clip_skips_detections_without_embedding():
    target = [1.0, 0.0, 0.0]
    a = _det(None, embedding=None)
    b = _det(None, embedding=[1.0, 0.0, 0.0])
    ranked = _rank_by_clip([a, b], target, sim_floor=0.0)
    assert [d for d, _ in ranked] == [b]


# ---------------------------------------------------------------------------
# _target_text_embedding (caching + retry)
# ---------------------------------------------------------------------------
def test_target_text_embedding_caches_one_call():
    img = _FakeImage([], text_emb=[1.0, 2.0, 3.0])
    ctx = _ctx(img)
    first = _target_text_embedding(ctx, "coke")
    second = _target_text_embedding(ctx, "coke")
    assert first == [1.0, 2.0, 3.0] == second
    assert len(img.embed_text_calls) == 1                 # cached, not re-fetched
    assert img.embed_text_calls[0] == "a photo of coke"   # default template applied


def test_target_text_embedding_failure_returns_none_then_retries():
    img = _FakeImage([], text_emb=[1.0, 0.0], embed_raises_until=1)
    ctx = _ctx(img)
    assert _target_text_embedding(ctx, "coke") is None    # 1st call raises -> None, not cached
    assert _target_text_embedding(ctx, "coke") == [1.0, 0.0]  # 2nd call retries + succeeds
    assert len(img.embed_text_calls) == 2


def test_target_text_embedding_empty_target_no_network():
    img = _FakeImage([], text_emb=[1.0])
    assert _target_text_embedding(_ctx(img), "") is None
    assert img.embed_text_calls == []


# ---------------------------------------------------------------------------
# _detect_and_lift — the load-bearing selection behaviour
# ---------------------------------------------------------------------------
def test_detect_and_lift_picks_clip_best_not_nearest():
    near = _det(_mask((200, 280), (100, 180)), embedding=[0.0, 1.0, 0.0])   # nearer, wrong
    far = _det(_mask((200, 280), (460, 540)), embedding=[1.0, 0.0, 0.0])    # farther, target
    img = _FakeImage([near, far], text_emb=[1.0, 0.0, 0.0])
    snap = _snap(_two_object_depth())

    out = _detect_and_lift(
        _ctx(img), snap, ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is not None
    cloud, range_m, conf, clip_sim = out
    assert img.process_calls == 1                  # one round trip, masks + embeddings
    assert clip_sim == pytest.approx(1.0)          # matched the target embedding
    assert range_m > 2.0                           # selected the FAR (target) box, not the near one


def test_detect_and_lift_falls_back_to_nearest_without_embeddings():
    near = _det(_mask((200, 280), (100, 180)), embedding=None)
    far = _det(_mask((200, 280), (460, 540)), embedding=None)
    img = _FakeImage([near, far], text_emb=[1.0, 0.0, 0.0])
    snap = _snap(_two_object_depth())

    out = _detect_and_lift(
        _ctx(img), snap, ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is not None
    _cloud, range_m, _conf, clip_sim = out
    assert clip_sim is None                        # rerank produced nothing -> fallback
    assert range_m < 1.5                           # nearest-wins selected the NEAR box


def test_detect_and_lift_kill_switch_uses_plain_detect(monkeypatch):
    monkeypatch.setenv("WALKIE_GRASP_CLIP_RERANK", "0")
    near = _det(_mask((200, 280), (100, 180)), embedding=[0.0, 1.0, 0.0])
    far = _det(_mask((200, 280), (460, 540)), embedding=[1.0, 0.0, 0.0])
    img = _FakeImage([near, far], text_emb=[1.0, 0.0, 0.0])
    snap = _snap(_two_object_depth())

    out = _detect_and_lift(
        _ctx(img), snap, ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is not None
    _cloud, range_m, _conf, clip_sim = out
    assert img.process_calls == 0                  # no embed round trip
    assert img.detect_calls == [["coke"]]          # unexpanded, today's path
    assert clip_sim is None
    assert range_m < 1.5                           # nearest-wins


def test_detect_and_lift_no_detections_returns_none():
    img = _FakeImage([])
    out = _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is None


def test_detect_and_lift_drops_below_confidence():
    near = _det(_mask((200, 280), (100, 180)), confidence=0.1, embedding=[1.0, 0.0, 0.0])
    img = _FakeImage([near], text_emb=[1.0, 0.0, 0.0])
    out = _detect_and_lift(
        _ctx(img), _snap(_two_object_depth()), ["coke"],
        voxel=0.005, erode_px=2, min_points=10, min_confidence=0.3,
    )
    assert out is None                             # only detection filtered by confidence
