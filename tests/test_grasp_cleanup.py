"""Unit tests for the pre-GraspNet cloud cleanup (tasks.skills.grasp._clean_object_cloud).

No robot, no AI server — synthesise an object cloud plus the two kinds of junk a masked
detection lifts (a locally-dense background-bleed blob + sparse flying-pixel scatter) and
assert the cleanup keeps the object and drops both. The load-bearing case is the *nearest*
cluster guard: when the bleed blob is bigger than the object, a largest-cluster rule would
drop the object — nearest-to-centroid must not.
"""

import numpy as np
import pytest

from tasks.skills.grasp import _clean_object_cloud


def _grid(center, extent, spacing=0.006):
    """A dense axis-aligned grid of points — one DBSCAN cluster (spacing < eps)."""
    cx, cy, cz = center
    ex, ey, ez = extent
    ax = np.arange(-ex / 2, ex / 2 + 1e-9, spacing)
    ay = np.arange(-ey / 2, ey / 2 + 1e-9, spacing)
    az = np.arange(-ez / 2, ez / 2 + 1e-9, spacing)
    gx, gy, gz = np.meshgrid(cx + ax, cy + ay, cz + az, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)


# Object ~6 cm cube ~0.6 m in front of the (optical-frame) camera.
OBJ_CENTER = np.array([0.0, 0.0, 0.60], dtype=np.float32)
OBJECT = _grid(OBJ_CENTER, (0.06, 0.06, 0.06))
# Background "table/wall" the mask bled onto: a wide thin patch 0.4 m behind the object,
# a clean DBSCAN gap (0.37 m >> eps 0.02) away from it.
BG_CENTER = np.array([0.0, 0.0, 1.00], dtype=np.float32)
BG_SMALL = _grid(BG_CENTER, (0.10, 0.10, 0.004))   # fewer points than the object
BG_LARGE = _grid(BG_CENTER, (0.34, 0.34, 0.004))   # MORE points than the object


def _scatter(n=25, seed=0):
    """Sparse isolated flying pixels spread across a wide volume (DBSCAN noise + SOR junk)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform([-0.4, -0.4, 0.3], [0.4, 0.4, 1.3], size=(n, 3))).astype(np.float32)


def _near(cloud, center, radius):
    return np.all(np.linalg.norm(cloud - center, axis=1) <= radius)


def _far(cloud, center, radius):
    return np.all(np.linalg.norm(cloud - center, axis=1) >= radius)


def test_drops_background_blob_and_scatter():
    cloud = np.vstack([OBJECT, BG_SMALL, _scatter()])
    cleaned = _clean_object_cloud(cloud, ref_optical=OBJ_CENTER)
    # Everything kept is the object; the bleed blob and scatter are gone.
    assert cleaned.shape[0] >= OBJECT.shape[0] * 0.8
    assert _near(cleaned, OBJ_CENTER, 0.08)
    assert _far(cleaned, BG_CENTER, 0.2)


def test_nearest_cluster_beats_largest():
    # The bleed blob has MORE points than the object — a largest-cluster rule would keep
    # the table. nearest-to-ref must keep the object instead.
    cloud = np.vstack([OBJECT, BG_LARGE, _scatter()])
    assert BG_LARGE.shape[0] > OBJECT.shape[0]
    cleaned = _clean_object_cloud(cloud, ref_optical=OBJ_CENTER)
    assert _near(cleaned, OBJ_CENTER, 0.08)
    assert _far(cleaned, BG_CENTER, 0.2)


def test_median_fallback_keeps_object_when_majority():
    # No ref supplied: the median lands on the object (it's the majority), so the
    # nearest-cluster-to-median keep still selects it.
    cloud = np.vstack([OBJECT, BG_SMALL, _scatter()])
    cleaned = _clean_object_cloud(cloud)  # ref_optical=None
    assert _near(cleaned, OBJ_CENTER, 0.08)
    assert _far(cleaned, BG_CENTER, 0.2)


def test_small_cloud_passthrough():
    tiny = np.random.default_rng(1).uniform(0, 1, size=(10, 3)).astype(np.float32)
    cleaned = _clean_object_cloud(tiny)
    assert np.array_equal(cleaned, tiny)  # below MIN_KEEP → never trimmed


def test_nan_rows_dropped():
    cloud = np.vstack([OBJECT, BG_SMALL]).copy()
    cloud[5] = np.nan
    cloud[123] = np.inf
    cleaned = _clean_object_cloud(cloud, ref_optical=OBJ_CENTER)
    assert np.isfinite(cleaned).all()


def test_disabled_returns_input(monkeypatch):
    monkeypatch.setenv("WALKIE_GRASP_CLEAN_ENABLE", "0")
    cloud = np.vstack([OBJECT, BG_LARGE]).astype(np.float32)
    cleaned = _clean_object_cloud(cloud, ref_optical=OBJ_CENTER)
    assert np.array_equal(cleaned, cloud)  # only the finite filter runs, all finite here
