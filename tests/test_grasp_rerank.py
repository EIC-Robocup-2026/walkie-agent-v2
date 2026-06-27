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
def _clear_module_caches():
    """The success-only caches are module-global — reset them per test."""
    grasp._TEXT_EMB_CACHE.clear()
    grasp._DESCRIPTOR_CACHE.clear()
    yield
    grasp._TEXT_EMB_CACHE.clear()
    grasp._DESCRIPTOR_CACHE.clear()


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


class _FakeExtract:
    """Stand-in for ctx.extract — records targets, returns canned descriptors.

    Raises ``RuntimeError`` on the first ``raises_until`` calls (to exercise the
    degrade-to-static-map fallback), then returns the descriptor list.
    """

    def __init__(self, descriptors, *, raises_until=0):
        self._descriptors = descriptors
        self._raises_until = raises_until
        self.calls: list[str] = []

    def __call__(self, schema, instructions, text):
        self.calls.append(text)
        if len(self.calls) <= self._raises_until:
            raise RuntimeError("llm down")
        return SimpleNamespace(descriptors=list(self._descriptors))


def _ctx(image, *, extract=None):
    ns = SimpleNamespace(walkieAI=SimpleNamespace(image=image))
    if extract is not None:
        ns.extract = extract
    return ns


# ---------------------------------------------------------------------------
# _detection_prompts — static-map fallback path (LLM descriptors disabled)
# ---------------------------------------------------------------------------
@pytest.fixture
def _llm_off(monkeypatch):
    """Pin _detection_prompts to the static-map path (WALKIE_GRASP_LLM_DESCRIPTORS=0)."""
    monkeypatch.setenv("WALKIE_GRASP_LLM_DESCRIPTORS", "0")


def test_detection_prompts_static_expands_known_item_target_first(_llm_off):
    out = _detection_prompts(_ctx(_FakeImage([])), ["coke"])
    assert out[0] == "coke"                       # specific target stays first
    assert out == ["coke", *_GRASP_DESCRIPTORS["coke"]]


def test_detection_prompts_static_dedups_preserving_order(_llm_off):
    out = _detection_prompts(_ctx(_FakeImage([])), ["coke", "can"])  # "can" also a descriptor
    assert out.count("can") == 1
    assert out[0] == "coke"


def test_detection_prompts_static_unknown_item_unchanged(_llm_off):
    assert _detection_prompts(_ctx(_FakeImage([])), ["banana"]) == ["banana"]


def test_detection_prompts_empty_unchanged(_llm_off):
    assert _detection_prompts(_ctx(_FakeImage([])), []) == []


def test_detection_prompts_rerank_off_passes_through(monkeypatch):
    monkeypatch.setenv("WALKIE_GRASP_CLIP_RERANK", "0")
    # rerank off short-circuits BEFORE any LLM call, so the extract must never fire.
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["can"]))
    assert _detection_prompts(ctx, ["coke"]) == ["coke"]
    assert ctx.extract.calls == []


# ---------------------------------------------------------------------------
# _detection_prompts — LLM-generated descriptors (the new primary path)
# ---------------------------------------------------------------------------
def test_detection_prompts_llm_generates_descriptors_target_first():
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["can", "red can"]))
    out = _detection_prompts(ctx, ["mystery soda"])
    assert out == ["mystery soda", "can", "red can"]   # target first, LLM phrases appended
    assert ctx.extract.calls == ["mystery soda"]        # asked the LLM for the lowercased target


def test_detection_prompts_llm_cached_one_call_per_target():
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["can"]))
    first = _detection_prompts(ctx, ["mystery soda"])
    second = _detection_prompts(ctx, ["mystery soda"])
    assert first == second == ["mystery soda", "can"]
    assert ctx.extract.calls == ["mystery soda"]        # cached, not re-asked


def test_detection_prompts_llm_caps_descriptor_count():
    many = [f"d{i}" for i in range(20)]
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(many))
    out = _detection_prompts(ctx, ["mystery soda"])
    assert out == ["mystery soda", *many[: grasp._MAX_DESCRIPTORS]]   # trimmed to the cap


def test_detection_prompts_llm_skips_brand_name_and_dupes():
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["Coke", "can", "can", "red can"]))
    out = _detection_prompts(ctx, ["coke"])
    assert out == ["coke", "can", "red can"]            # brand ("coke") + dup "can" dropped


def test_detection_prompts_llm_failure_falls_back_to_static_known():
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["x"], raises_until=99))
    out = _detection_prompts(ctx, ["coke"])             # LLM raises
    assert out == ["coke", *_GRASP_DESCRIPTORS["coke"]]  # static map fills in
    # The miss is cached as [] so a slow/dead endpoint isn't re-hit on every locate;
    # a second call must not re-invoke the LLM.
    assert grasp._DESCRIPTOR_CACHE == {"coke": []}
    out2 = _detection_prompts(ctx, ["coke"])
    assert out2 == out
    assert ctx.extract.calls == ["coke"]                # one attempt, then cached miss


def test_detection_prompts_llm_failure_unknown_item_bare_target():
    ctx = _ctx(_FakeImage([]), extract=_FakeExtract(["x"], raises_until=99))
    assert _detection_prompts(ctx, ["banana"]) == ["banana"]  # no LLM, no static -> bare


def test_detection_prompts_bad_ctx_without_extract_falls_back_to_static():
    # A ctx lacking .extract (AttributeError inside the timeout worker) must degrade,
    # not raise — guards the production grasp module that only ever touched ctx.walkieAI.
    out = _detection_prompts(_ctx(_FakeImage([])), ["coke"])
    assert out == ["coke", *_GRASP_DESCRIPTORS["coke"]]


def test_detection_prompts_llm_timeout_falls_back_to_static(monkeypatch):
    # A slow endpoint must not hang the grasp: the timeout guard fires -> static map.
    import time as _time

    monkeypatch.setenv("WALKIE_GRASP_LLM_DESCRIPTORS_TIMEOUT_S", "0.1")

    def _slow(schema, instructions, text):
        _time.sleep(1.0)
        return SimpleNamespace(descriptors=["can"])

    out = _detection_prompts(_ctx(_FakeImage([]), extract=_slow), ["coke"])
    assert out == ["coke", *_GRASP_DESCRIPTORS["coke"]]
    assert grasp._DESCRIPTOR_CACHE == {"coke": []}    # timed-out miss cached -> no re-stall


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
