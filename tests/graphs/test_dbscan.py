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
