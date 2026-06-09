"""Association math: nn_ratio overlap, AABB prefilter, normalized additive score."""

from __future__ import annotations

import numpy as np
import pytest

from walkie_graphs.fusion import (
    aabb_overlap,
    additive_similarity,
    nn_ratio,
    nn_ratio_symmetric,
    phi_sem,
)


def _grid(center, n=8, step=0.01):
    """A small deterministic lattice of points around ``center``."""
    a = np.arange(n) * step
    xs, ys = np.meshgrid(a, a)
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    return (pts + np.asarray(center) - pts.mean(axis=0)).astype(np.float32)


def test_nn_ratio_identical_is_one():
    pts = _grid((0, 0, 0))
    assert nn_ratio(pts, pts, voxel_m=0.025) == pytest.approx(1.0)


def test_nn_ratio_disjoint_is_zero():
    a = _grid((0, 0, 0))
    b = _grid((10, 10, 10))
    assert nn_ratio(a, b, voxel_m=0.025) == 0.0


def test_nn_ratio_partial_overlap():
    obj = _grid((0, 0, 0), n=10, step=0.01)  # spans ~0..0.09
    # Shift the detection so ~half its points land within voxel_m of an obj point.
    det = obj + np.array([0.05, 0.0, 0.0], dtype=np.float32)
    r = nn_ratio(obj, det, voxel_m=0.025)
    assert 0.2 < r < 0.8


def test_nn_ratio_voxel_threshold_boundary():
    obj = np.zeros((1, 3), dtype=np.float32)
    det = np.array([[0.02, 0, 0], [0.03, 0, 0]], dtype=np.float32)
    # voxel 0.025: first point (0.02) counts, second (0.03) does not -> 0.5
    assert nn_ratio(obj, det, voxel_m=0.025) == pytest.approx(0.5)


def test_nn_ratio_empty():
    assert nn_ratio(np.zeros((0, 3)), _grid((0, 0, 0)), 0.025) == 0.0
    assert nn_ratio(_grid((0, 0, 0)), np.zeros((0, 3)), 0.025) == 0.0


def test_nn_ratio_symmetric_picks_max():
    big = _grid((0, 0, 0), n=12, step=0.01)
    small = _grid((0, 0, 0), n=3, step=0.01)  # fully inside big
    # small-in-big ~1.0, big-in-small <1.0 -> symmetric ~1.0
    assert nn_ratio_symmetric(big, small, 0.025) == pytest.approx(1.0, abs=0.05)


def test_aabb_overlap():
    assert aabb_overlap((0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (2, 2, 2))
    assert not aabb_overlap((0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3))
    # pad bridges a small gap
    assert aabb_overlap((0, 0, 0), (1, 1, 1), (1.02, 0, 0), (2, 1, 1), pad=0.03)
    assert not aabb_overlap((0, 0, 0), (1, 1, 1), (1.02, 0, 0), (2, 1, 1), pad=0.0)


def test_phi_sem_maps_to_unit_interval():
    assert phi_sem(1.0) == pytest.approx(1.0)
    assert phi_sem(-1.0) == pytest.approx(0.0)
    assert phi_sem(0.0) == pytest.approx(0.5)


def test_additive_weights():
    # phi = w_geo*nnratio + w_sem*phi_sem(cos)
    assert additive_similarity(0.5, 1.0, w_geo=1.0, w_sem=1.0) == pytest.approx(1.5)
    assert additive_similarity(0.0, 1.0, w_geo=2.0, w_sem=0.5) == pytest.approx(0.5)


def test_pure_visual_never_reaches_default_threshold():
    # With nnratio=0 and any cosine, additive (w=1,1) tops out at phi_sem(1)=1.0 < 1.1.
    for cos in (-1.0, 0.0, 0.5, 1.0):
        assert additive_similarity(0.0, cos) < 1.1
