"""DBSCAN largest-cluster denoising (no robot/server).

Every test runs against BOTH backends: sklearn's DBSCAN (the fast path) and the
pure scipy/union-find fallback (used when sklearn is absent).
"""

from __future__ import annotations

import numpy as np
import pytest

import services.walkie_graphs.dbscan as dbscan_mod
from services.walkie_graphs.dbscan import (
    dbscan_labels,
    dbscan_largest_cluster,
    dbscan_remove_noise,
    statistical_outlier_removal,
)


@pytest.fixture(autouse=True, params=["sklearn", "scipy-fallback"])
def backend(request, monkeypatch):
    if request.param == "scipy-fallback":
        monkeypatch.setattr(dbscan_mod, "_SKDBSCAN", None)
    elif dbscan_mod._SKDBSCAN is None:
        pytest.skip("scikit-learn not installed")
    return request.param


def _blob(center, n, spread=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(center, spread, size=(n, 3)).astype(np.float32)


def test_keeps_larger_of_two_clusters():
    big = _blob((0, 0, 0), 60, seed=1)
    small = _blob((2, 2, 2), 12, seed=2)  # far enough to be a separate cluster
    pts = np.vstack([big, small])
    kept = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    assert len(kept) == 60
    # Every kept point belongs to the big blob (near origin, not near (2,2,2)).
    assert np.linalg.norm(kept.mean(axis=0)) < 0.1


def test_strips_outlier_noise():
    blob = _blob((0, 0, 0), 60, seed=3)
    outliers = np.array([[5, 5, 5], [5.0, 5.0, 5.2], [-4, 3, 1]], dtype=np.float32)
    pts = np.vstack([blob, outliers])
    kept = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    assert len(kept) == 60  # the 3 scattered outliers are dropped


def test_small_cluster_fallback_returns_original():
    # Only 6 points, all isolated -> no cluster reaches min_cluster_size(5) cleanly;
    # with min_points=10 there are no core points at all -> fallback to original.
    pts = _blob((0, 0, 0), 6, spread=0.005, seed=4)
    kept = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    assert np.array_equal(kept, pts)


def test_single_cluster_passthrough():
    pts = _blob((1, 1, 1), 50, seed=5)
    kept = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    assert len(kept) == 50


def test_labels_mark_noise_as_minus_one():
    blob = _blob((0, 0, 0), 40, seed=6)
    outlier = np.array([[9, 9, 9]], dtype=np.float32)
    pts = np.vstack([blob, outlier])
    labels = dbscan_labels(pts, eps=0.05, min_points=10)
    assert labels[-1] == -1  # the lone far point is noise
    assert set(labels[:-1].tolist()) == {0}  # the blob is one cluster


def test_empty_and_too_few():
    assert len(dbscan_largest_cluster(np.zeros((0, 3), np.float32), 0.05, 10)) == 0
    few = _blob((0, 0, 0), 3, seed=7)
    assert np.array_equal(dbscan_largest_cluster(few, 0.05, 10), few)


def test_determinism():
    pts = np.vstack([_blob((0, 0, 0), 50, seed=8), _blob((3, 0, 0), 50, seed=9)])
    a = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    b = dbscan_largest_cluster(pts, eps=0.05, min_points=10)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# dbscan_remove_noise — keep every cluster, drop only isolated scatter
# ---------------------------------------------------------------------------
def test_remove_noise_keeps_multiple_clusters():
    big = _blob((0, 0, 0), 60, seed=1)
    small = _blob((2, 2, 2), 20, seed=2)  # a second REAL cluster, far from the first
    strays = np.array([[9, 9, 9], [-8, 0, 4]], dtype=np.float32)
    pts = np.vstack([big, small, strays])
    kept = dbscan_remove_noise(pts, eps=0.05, min_points=10)
    assert len(kept) == 80  # both clusters survive; only the 2 strays are dropped
    # (largest-cluster keep would have returned just the 60-point blob)
    assert len(dbscan_largest_cluster(pts, eps=0.05, min_points=10)) == 60


def test_remove_noise_all_noise_falls_back_to_original():
    # 12 isolated points, no cluster possible → return the original cloud untouched.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-5, 5, size=(12, 3)).astype(np.float32)
    kept = dbscan_remove_noise(pts, eps=0.05, min_points=10)
    assert np.array_equal(kept, pts)


def test_remove_noise_too_few_points_passthrough():
    few = _blob((0, 0, 0), 4, seed=4)
    assert np.array_equal(dbscan_remove_noise(few, eps=0.05, min_points=10), few)
    empty = np.zeros((0, 3), np.float32)
    assert len(dbscan_remove_noise(empty, eps=0.05, min_points=10)) == 0


def test_remove_noise_clean_cloud_unchanged():
    pts = _blob((1, 1, 1), 50, seed=5)
    kept = dbscan_remove_noise(pts, eps=0.05, min_points=10)
    assert len(kept) == 50


# ---------------------------------------------------------------------------
# statistical_outlier_removal — Open3D fast path AND scipy fallback
# ---------------------------------------------------------------------------
@pytest.fixture(params=["open3d", "scipy-fallback"])
def sor_backend(request, monkeypatch):
    if request.param == "scipy-fallback":
        monkeypatch.setattr(dbscan_mod, "_O3D", False)  # force the cKDTree path
    elif not dbscan_mod._open3d():
        pytest.skip("open3d not installed")
    return request.param


def _grid(n=20, step=0.01):
    gx, gy = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1).astype(np.float32)


def test_sor_strips_halo_keeps_dense_blob(sor_backend):
    surface = _grid()
    rng = np.random.default_rng(1)
    # sparse halo floating clearly OFF the surface (z >= 0.2, scattered)
    halo = surface[::20].copy()
    halo[:, 2] = 0.2 + rng.uniform(0, 0.3, len(halo)).astype(np.float32)
    kept = statistical_outlier_removal(np.vstack([surface, halo]), k=8, std_ratio=1.0)
    # the sparse halo is gone, the dense surface survives (allow a few edge points out)
    assert len(surface) * 0.9 <= len(kept) <= len(surface)
    assert kept[:, 2].max() < 0.05  # no halo point remains


def test_sor_keeps_disjoint_dense_clusters(sor_backend):
    # Two dense clusters far apart (two partial views of one object) + strays: both
    # clusters must survive — the property a largest-cluster keep cannot provide.
    a = _grid()
    b = _grid() + np.array([3.0, 0, 0], dtype=np.float32)
    strays = np.array([[1.5, 1.5, 1.0], [-1.0, 2.0, 0.7]], dtype=np.float32)
    kept = statistical_outlier_removal(np.vstack([a, b, strays]), k=8, std_ratio=1.0)
    assert (kept[:, 0] < 1.0).sum() >= len(a) * 0.9  # cluster A intact
    assert (kept[:, 0] > 2.0).sum() >= len(b) * 0.9  # cluster B intact
    assert np.abs(kept[:, 2]).max() < 0.05  # strays gone


def test_sor_disabled_and_small_cloud_passthrough(sor_backend):
    pts = _blob((0, 0, 0), 30, seed=6)
    assert np.array_equal(statistical_outlier_removal(pts, k=0, std_ratio=2.0), pts)
    assert np.array_equal(statistical_outlier_removal(pts, k=16, std_ratio=0.0), pts)
    few = _blob((0, 0, 0), 10, seed=7)
    assert np.array_equal(statistical_outlier_removal(few, k=16, std_ratio=2.0), few)
    empty = np.zeros((0, 3), np.float32)
    assert len(statistical_outlier_removal(empty, k=16, std_ratio=2.0)) == 0
