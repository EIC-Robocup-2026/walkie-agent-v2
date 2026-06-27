"""Golden cases for batch constrained-agglomerative association.

Every case is built from synthetic numpy point clouds (filled boxes / point grids)
and deterministic one-hot CLIP embeddings (one axis per class), so cosine is exactly
1.0 within a class and 0.0 across classes — no model, no randomness.

The five cases pin the structural fixes the v2 batch path exists for:

* twin fusion       — two identical chairs 1.5 m apart stay 2 objects (centroid cap);
* table absorption  — a spoon ON a table stays separate (mutual-min overlap + class);
* chaining          — a row of 4 adjacent chairs does not collapse (extent veto);
* ghosting          — two partial views of one sofa do merge (batch re-link);
* singletons        — a lone view becomes one object with ``n_obs == 1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.walkie_graphs.associate import (
    Observation,
    ObjectObservation,
    associate,
)

# A class → embedding-axis map; one-hot so cosine is 1.0 within a class, 0.0 across.
_CLASS_AXIS = {"chair": 0, "sofa": 1, "spoon": 2, "table": 3}


def _emb(class_name: str) -> list[float]:
    v = np.zeros(len(_CLASS_AXIS), dtype=np.float64)
    v[_CLASS_AXIS[class_name]] = 1.0
    return v.tolist()


def _box(center, size, step=0.02) -> np.ndarray:
    """A dense, axis-aligned filled box of points centred at ``center``.

    ``size`` is the full ``(sx, sy, sz)`` extent; ``step`` the point spacing.
    """
    cx, cy, cz = center
    sx, sy, sz = size
    xs = np.arange(-sx / 2, sx / 2 + 1e-9, step)
    ys = np.arange(-sy / 2, sy / 2 + 1e-9, step)
    zs = np.arange(-sz / 2, sz / 2 + 1e-9, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    pts += np.asarray([cx, cy, cz], dtype=np.float64)
    return pts.astype(np.float64)


def _obs(class_name, points, *, ts=0.0, conf=0.9, caption=None, class_id=None) -> Observation:
    return Observation(
        class_name=class_name,
        class_id=class_id if class_id is not None else _CLASS_AXIS[class_name],
        conf=conf,
        bbox=(0.0, 0.0, 1.0, 1.0),
        caption=caption if caption is not None else f"a {class_name}",
        clip_emb=_emb(class_name),
        ts=ts,
        points=points,
    )


def _centroids(objs: list[ObjectObservation]) -> np.ndarray:
    return np.asarray([o.centroid for o in objs], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. TWIN FUSION — two identical chairs 1.5 m apart stay two objects.
# ---------------------------------------------------------------------------
def test_two_identical_chairs_stay_separate():
    a = _box((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    b = _box((1.5, 0.0, 0.0), (0.4, 0.4, 0.4))  # identical shape + class + CLIP
    objs = associate([_obs("chair", a, ts=1.0), _obs("chair", b, ts=2.0)])

    assert len(objs) == 2
    assert all(o.n_obs == 1 for o in objs)
    # The two clusters sit at the two original centroids.
    cs = sorted(c[0] for c in _centroids(objs))
    assert cs[0] == pytest.approx(0.0, abs=0.05)
    assert cs[1] == pytest.approx(1.5, abs=0.05)


# ---------------------------------------------------------------------------
# 2. TABLE ABSORPTION — a spoon resting ON a table stays separate.
# ---------------------------------------------------------------------------
def test_spoon_on_table_not_absorbed():
    # A broad flat table top, and a small spoon fully inside its XY footprint, resting
    # on top of it. Different classes, and the table never has many points inside the
    # spoon → mutual-min overlap ≈ 0.
    table = _box((0.0, 0.0, 0.0), (1.2, 0.8, 0.04), step=0.02)
    spoon = _box((0.0, 0.0, 0.04), (0.10, 0.04, 0.03), step=0.01)
    objs = associate(
        [_obs("table", table, ts=1.0), _obs("spoon", spoon, ts=2.0)],
        max_dist_m=0.6,  # centroids are close; the gate must come from overlap+class
    )

    assert len(objs) == 2
    classes = sorted(o.class_name for o in objs)
    assert classes == ["spoon", "table"]
    assert all(o.n_obs == 1 for o in objs)


def test_spoon_on_table_separate_even_same_class():
    # Even if we drop the class gate, the mutual-min overlap alone keeps them apart:
    # table→spoon overlap is tiny. (require_same_class=False isolates the geometry.)
    table = _box((0.0, 0.0, 0.0), (1.2, 0.8, 0.04), step=0.02)
    spoon = _box((0.0, 0.0, 0.04), (0.10, 0.04, 0.03), step=0.01)
    objs = associate(
        [_obs("table", table, ts=1.0), _obs("table", spoon, ts=2.0)],
        require_same_class=False,
        max_dist_m=0.6,
    )
    assert len(objs) == 2


# ---------------------------------------------------------------------------
# 3. CHAINING — a row of 4 adjacent overlapping chairs does not collapse.
# ---------------------------------------------------------------------------
def test_row_of_four_chairs_does_not_collapse():
    # Four chairs in a row, each ~0.5 m wide, spaced 0.45 m so neighbours overlap a bit.
    # Single-link union-find would chain all four into one ~1.85 m blob; the extent veto
    # (cap 0.8 m for a chair) forbids that.
    obs = []
    for k in range(4):
        c = _box((0.45 * k, 0.0, 0.0), (0.5, 0.5, 0.5), step=0.025)
        obs.append(_obs("chair", c, ts=float(k)))
    objs = associate(
        obs,
        max_dist_m=0.6,                       # neighbours are candidate pairs
        max_extent_by_class={"chair": 0.8},   # one chair fits, a chained row does not
    )

    # Must not be a single blob; every cluster's long-axis extent stays within the cap.
    assert len(objs) >= 2
    for o in objs:
        assert o.extent[0] <= 0.8 + 1e-6
    # And no cluster spans more than ~2 of the original chairs.
    assert all(o.n_obs <= 2 for o in objs)


# ---------------------------------------------------------------------------
# 4. GHOSTING — two genuine partial views of ONE sofa merge into one.
# ---------------------------------------------------------------------------
def test_two_partial_views_of_one_sofa_merge():
    # One sofa spanning x∈[-0.6, 0.6]; two overlapping partial views (left & right
    # halves) that share the centre band. Same class, identical CLIP.
    left = _box((-0.2, 0.0, 0.0), (0.8, 0.4, 0.4), step=0.03)   # x ∈ [-0.6, 0.2]
    right = _box((0.2, 0.0, 0.0), (0.8, 0.4, 0.4), step=0.03)   # x ∈ [-0.2, 0.6]
    objs = associate(
        [_obs("sofa", left, ts=1.0, caption="left of sofa"),
         _obs("sofa", right, ts=2.0, caption="right of sofa")],
        max_extent_by_class={"sofa": 2.5},
    )

    assert len(objs) == 1
    o = objs[0]
    assert o.n_obs == 2
    assert o.ts_first == 1.0 and o.ts_last == 2.0
    # Union of captions kept for keyword recall.
    assert set(o.captions) == {"left of sofa", "right of sofa"}
    # Fused cloud spans the whole sofa.
    assert o.aabb_max[0] - o.aabb_min[0] == pytest.approx(1.2, abs=0.1)
    # Mean of two identical one-hot embeddings is still the unit one-hot.
    emb = np.asarray(o.clip_emb)
    assert emb[_CLASS_AXIS["sofa"]] == pytest.approx(1.0, abs=1e-6)
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. SINGLETON — a lone single view becomes one object with n_obs == 1.
# ---------------------------------------------------------------------------
def test_lone_object_is_one_observation():
    c = _box((3.0, 3.0, 0.0), (0.3, 0.3, 0.3))
    objs = associate([_obs("chair", c, ts=5.0, conf=0.42)])

    assert len(objs) == 1
    o = objs[0]
    assert o.n_obs == 1
    assert o.class_name == "chair"
    assert o.conf == pytest.approx(0.42)
    assert o.ts_first == 5.0 and o.ts_last == 5.0
    assert o.captions == ["a chair"]
    assert o.centroid[0] == pytest.approx(3.0, abs=0.05)


def test_empty_and_too_few_points_dropped():
    empty = _obs("chair", np.zeros((0, 3)), ts=1.0)
    tiny = _obs("chair", np.array([[0.0, 0.0, 0.0]]), ts=2.0)  # 1 point < min
    real = _box((0.0, 0.0, 0.0), (0.4, 0.4, 0.4))
    objs = associate([empty, tiny, _obs("chair", real, ts=3.0)])
    assert len(objs) == 1
    assert objs[0].n_obs == 1


def test_empty_input_returns_empty():
    assert associate([]) == []
