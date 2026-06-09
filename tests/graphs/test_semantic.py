"""Tier-3 LLM caption refinement, best-view retention, and LLM edge inference.

All driven by a FakeModel (``.invoke(x) -> obj.content``) — no network/server.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from walkie_graphs.memory import Detection3D
from tests.graphs.conftest import make_det, put_box, unit


class FakeModel:
    def __init__(self, canned="a clean ceramic mug", *, raise_on_invoke=False):
        self.canned = canned
        self.raise_on_invoke = raise_on_invoke
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raise_on_invoke:
            raise RuntimeError("model down")
        return SimpleNamespace(content=self.canned)


def _det_with_crop(center, conf, color, ts):
    rng = np.random.default_rng(int(ts * 1000))
    pts = rng.normal(center, 0.01, size=(40, 3)).astype(np.float32)
    return Detection3D(
        class_name="mug", class_id=0, confidence=conf, bbox_xyxy=(0, 0, 8, 8),
        points_world=pts, clip_emb=unit(1, 0, 0), caption="a mug", ts=ts,
        crop=Image.new("RGB", (8, 8), color),
    )


# ---------------------------------------------------------------------------
# Caption refinement
# ---------------------------------------------------------------------------
def test_refine_updates_best_caption(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a white mug"))
    model = FakeModel(canned="a white ceramic coffee mug")
    assert mem.refine_captions(model) == 1
    node = mem.all_objects()[0]
    assert node.best_caption == "a white ceramic coffee mug"
    assert "a white ceramic coffee mug" in node.captions


def test_refine_passes_existing_captions_to_model(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a chipped blue mug"))
    model = FakeModel()
    mem.refine_captions(model)
    assert model.calls  # invoked once
    assert "a chipped blue mug" in model.calls[0]  # the rough caption is in the prompt


def test_refine_best_effort_on_model_error(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a mug"))
    before = mem.all_objects()[0].best_caption
    assert mem.refine_captions(FakeModel(raise_on_invoke=True)) == 0
    assert mem.all_objects()[0].best_caption == before  # unchanged


def test_refine_noop_without_model(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption="a mug"))
    assert mem.refine_captions(None) == 0


def test_refine_skips_nodes_without_captions(mem):
    mem.upsert(make_det(class_name="mug", center=(1.0, 0.0, 0.5), caption=""))
    assert mem.refine_captions(FakeModel()) == 0


# ---------------------------------------------------------------------------
# Best-N view retention
# ---------------------------------------------------------------------------
def test_best_views_keeps_top_n_by_confidence(mem):
    mem.best_views_n = 3
    confs = [0.5, 0.9, 0.7, 0.3]
    colors = ["red", "green", "blue", "white"]
    for i, (c, col) in enumerate(zip(confs, colors)):
        mem.upsert(_det_with_crop((1.0, 0.0, 0.5), c, col, ts=1.0 + i))
    node = mem.all_objects()[0]
    assert node.n_obs == 4  # all merged into one object
    kept = sorted(c for c, _ in node.best_views)
    assert kept == [0.5, 0.7, 0.9]  # the lowest (0.3) was dropped
    assert len(node.best_views) == 3


def test_best_views_off_by_default(mem):
    assert mem.best_views_n == 0
    mem.upsert(_det_with_crop((1.0, 0.0, 0.5), 0.9, "red", ts=1.0))
    assert mem.all_objects()[0].best_views == []


def test_refine_with_images(mem):
    pytest.importorskip("langchain_core")
    mem.best_views_n = 3
    mem.upsert(_det_with_crop((1.0, 0.0, 0.5), 0.9, "red", ts=1.0))
    model = FakeModel(canned="a red mug")
    assert mem.refine_captions(model, use_images=True) == 1
    assert mem.all_objects()[0].best_caption == "a red mug"


# ---------------------------------------------------------------------------
# LLM edge inference (kept separate from geometric edges)
# ---------------------------------------------------------------------------
def test_infer_edges_adds_llm_edges_without_touching_geometric(mem):
    put_box(mem, "table", "table", [0.0, 0.0, 0.0], [1.0, 1.0, 0.4])
    put_box(mem, "mug", "mug", [0.4, 0.4, 0.4], [0.5, 0.5, 0.5])
    mem.derive_relations()
    geom_before = {(r.src_id, r.dst_id, r.predicate) for r in mem.all_relations()}
    assert ("mug", "table", "on") in geom_before

    written = mem.infer_edges_llm(FakeModel(canned="on"))
    assert written >= 1
    preds = {r.predicate for r in mem.all_relations()}
    assert any(p.startswith("llm:") for p in preds)
    # geometric edges survive
    assert ("mug", "table", "on") in {
        (r.src_id, r.dst_id, r.predicate) for r in mem.all_relations()
    }


def test_infer_edges_skips_none_relation(mem):
    put_box(mem, "a", "a", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    put_box(mem, "b", "b", [0.5, 0.0, 0.0], [0.6, 0.1, 0.1])
    assert mem.infer_edges_llm(FakeModel(canned="none")) == 0
    assert not any(r.predicate.startswith("llm:") for r in mem.all_relations())


def test_infer_edges_rerun_replaces_old_llm_edges(mem):
    put_box(mem, "table", "table", [0.0, 0.0, 0.0], [1.0, 1.0, 0.4])
    put_box(mem, "mug", "mug", [0.4, 0.4, 0.4], [0.5, 0.5, 0.5])
    mem.infer_edges_llm(FakeModel(canned="on"))
    mem.infer_edges_llm(FakeModel(canned="next to"))
    llm = [r.predicate for r in mem.all_relations() if r.predicate.startswith("llm:")]
    assert llm == ["llm:next to"]  # no stale "llm:on" left behind


def test_infer_edges_noop_without_model(mem):
    put_box(mem, "a", "a", [0.0, 0.0, 0.0], [0.1, 0.1, 0.1])
    put_box(mem, "b", "b", [0.5, 0.0, 0.0], [0.6, 0.1, 0.1])
    assert mem.infer_edges_llm(None) == 0
