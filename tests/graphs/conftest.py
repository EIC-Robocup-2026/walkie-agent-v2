"""Shared fixtures/helpers for walkie_graphs tests — no robot/server needed."""

from __future__ import annotations

import math

import numpy as np
import pytest

from walkie_graphs.memory import Detection3D, GraphMemory, ObjectNode


def unit(*vals) -> list[float]:
    """A normalized embedding vector from the given components."""
    v = np.asarray(vals, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def emb_with_cosine(target: float) -> list[float]:
    """A 3-D unit vector whose cosine with unit(1, 0, 0) equals ``target``."""
    return [target, math.sqrt(max(0.0, 1.0 - target * target)), 0.0]


def make_det(
    class_name="mug",
    center=(1.0, 0.0, 0.5),
    emb=None,
    *,
    conf=0.9,
    caption="",
    ts=1.0,
    spread=0.01,
    n=60,
    class_id=0,
) -> Detection3D:
    """A Detection3D with a small gaussian point cloud around ``center``."""
    rng = np.random.default_rng(abs(hash((class_name, tuple(center), ts))) % (2**32))
    pts = rng.normal(center, spread, size=(n, 3)).astype(np.float32)
    return Detection3D(
        class_name=class_name,
        class_id=class_id,
        confidence=conf,
        bbox_xyxy=(0, 0, 10, 10),
        points_world=pts,
        clip_emb=emb if emb is not None else unit(1, 0, 0),
        caption=caption,
        ts=ts,
    )


def put_box(mem: GraphMemory, node_id, cls, cmin, cmax, *, emb=None) -> ObjectNode:
    """Register a node with an explicit AABB (for relation tests)."""
    c = tuple((a + b) / 2 for a, b in zip(cmin, cmax))
    ext = tuple(b - a for a, b in zip(cmin, cmax))
    node = ObjectNode(
        id=node_id, class_name=cls, class_id=0,
        centroid=c, extent=ext, aabb_min=tuple(cmin), aabb_max=tuple(cmax),
        clip_emb=emb if emb is not None else unit(1, 0, 0),
        captions=[], best_caption="", n_obs=1, conf=0.9,
        first_seen_ts=0.0, last_seen_ts=0.0,
    )
    mem._write_node(node)
    return node


@pytest.fixture
def mem(tmp_path):
    """An ephemeral (in-memory Chroma) GraphMemory writing sidecars to tmp_path."""
    return GraphMemory(
        chroma_dir=None,
        pcds_dir=str(tmp_path / "pcds"),
        thumbs_dir=str(tmp_path / "thumbs"),
        edges_path=str(tmp_path / "edges.json"),
    )
