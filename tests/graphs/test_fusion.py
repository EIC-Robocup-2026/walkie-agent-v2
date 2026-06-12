"""Association math: nn_ratio overlap, AABB prefilter, normalized additive score."""

from __future__ import annotations

import numpy as np
import pytest

from services.walkie_graphs.fusion import (
    aabb_overlap,
    additive_similarity,
    icp_align,
    nn_ratio,
    nn_ratio_symmetric,
    pairs_within,
    phi_sem,
    subtract_contained_masks,
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


# ---------------------------------------------------------------------------
# mask_subtract_contained (CG): the table's mask loses the mug's pixels
# ---------------------------------------------------------------------------
def _mask_of(bbox, shape=(100, 100)):
    m = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = bbox
    m[y1:y2, x1:x2] = True
    return m


def test_subtract_contained_removes_inner_from_outer():
    table_box, mug_box = (10, 10, 90, 90), (40, 40, 55, 55)
    masks = [_mask_of(table_box), _mask_of(mug_box)]
    out = subtract_contained_masks([table_box, mug_box], masks)
    # mug pixels removed from the table's mask...
    assert not out[0][45, 45]
    assert out[0][20, 20]  # ...but the rest of the table survives
    # the mug's own mask is untouched
    assert np.array_equal(out[1], masks[1])
    # inputs not mutated
    assert masks[0][45, 45]


def test_subtract_contained_ignores_partial_overlap():
    # Two side-by-side boxes overlapping a sliver: neither contains the other.
    a, b = (10, 10, 60, 60), (50, 10, 100, 60)
    masks = [_mask_of(a), _mask_of(b)]
    out = subtract_contained_masks([a, b], masks)
    assert np.array_equal(out[0], masks[0])
    assert np.array_equal(out[1], masks[1])


def test_subtract_contained_passes_none_through():
    a, b = (10, 10, 90, 90), (40, 40, 55, 55)
    out = subtract_contained_masks([a, b], [None, _mask_of(b)])
    assert out[0] is None
    assert np.array_equal(out[1], _mask_of(b))


def test_pairs_within():
    pts = [(0, 0, 0), (0.3, 0, 0), (5, 0, 0)]
    assert pairs_within(pts, 0.5) == [(0, 1)]
    assert pairs_within(pts, 10.0) == [(0, 1), (0, 2), (1, 2)]
    assert pairs_within([(0, 0, 0)], 1.0) == []


# ---------------------------------------------------------------------------
# icp_align — cancel residual camera-pose error before fusing clouds
# ---------------------------------------------------------------------------
def _corner_cloud(n=400, seed=3):
    """An L-shaped corner (two perpendicular planes) of APERIODIC points.

    A single flat plane could slide in-plane (the corner pins all translations), and
    the points must be irregular like a real depth scan — a perfect lattice would let
    ICP lock into a lattice-shifted local minimum that real clouds don't have.
    """
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(0, 0.25, n), rng.uniform(0, 0.25, n), np.zeros(n)], axis=1)
    wall = np.stack([rng.uniform(0, 0.25, n), np.zeros(n), rng.uniform(0, 0.25, n)], axis=1)
    return np.vstack([floor, wall]).astype(np.float32)


def test_icp_recovers_pose_offset():
    pytest.importorskip("open3d")
    target = _corner_cloud()
    offset = np.array([0.05, 0.03, 0.02], dtype=np.float32)  # typical pose error
    source = target + offset
    aligned, fitness = icp_align(source, target, max_corr_dist=0.1, min_points=50)
    assert fitness > 0.9
    residual = np.abs(aligned - target).max()
    assert residual < 0.005  # 5cm offset reduced to < 5mm


def test_icp_skips_barely_overlapping_clouds():
    pytest.importorskip("open3d")
    target = _corner_cloud()
    source = target + np.array([5.0, 0, 0], dtype=np.float32)  # no overlap at all
    aligned, fitness = icp_align(source, target, max_corr_dist=0.1, min_points=50)
    assert fitness < 0.6
    assert np.array_equal(aligned, source)  # unchanged — never snapped together


def test_icp_disabled_and_small_cloud_passthrough():
    target = _corner_cloud()
    source = target + 0.05
    out, fit = icp_align(source, target, max_corr_dist=0.0)  # disabled
    assert np.array_equal(out, source) and fit == 0.0
    small = source[:20]
    out, fit = icp_align(small, target, max_corr_dist=0.1, min_points=150)
    assert np.array_equal(out, small) and fit == 0.0


def test_icp_max_translation_rejects_extension_slide():
    """A degenerate slide of a partial extension view is rejected; a real offset isn't.

    A flat strip is translation-degenerate along its length, so when a new sighting
    overlaps the stored cloud at one end and extends past it, ICP slides the whole view
    bodily onto the overlap (high fitness, but it CRUSHES the new region — why big
    objects don't fill in). max_translation caps the accepted slide to a plausible
    pose-error correction, so the extension is kept raw for the union to preserve.
    """
    pytest.importorskip("open3d")
    rng = np.random.default_rng(4)

    def strip(x0, x1, n=600):
        return np.stack(
            [rng.uniform(x0, x1, n), rng.uniform(0, 0.25, n), rng.normal(0, 0.002, n)],
            axis=1,
        ).astype(np.float32)

    target = strip(0.0, 0.4)
    source = strip(0.3, 0.9)  # overlaps [0.3,0.4], NEW area [0.4,0.9]
    # Uncapped: ICP slides the strip far onto the target.
    slid, _ = icp_align(source, target, max_corr_dist=0.2, min_points=50)
    assert slid.mean(axis=0)[0] < source.mean(axis=0)[0] - 0.1
    # Capped: the big slide is rejected, raw source returned so the union keeps the new part.
    kept, _ = icp_align(source, target, max_corr_dist=0.2, min_points=50, max_translation=0.1)
    assert np.array_equal(kept, source)
    # A genuine small pose offset is still corrected with the cap on.
    src2 = target + np.array([0.05, 0.03, 0.02], dtype=np.float32)
    aligned, _ = icp_align(src2, target, max_corr_dist=0.2, min_points=50, max_translation=0.2)
    assert not np.array_equal(aligned, src2)


def test_icp_passthrough_without_open3d(monkeypatch):
    import services.walkie_graphs.dbscan as dbscan_mod

    monkeypatch.setattr(dbscan_mod, "_O3D", False)
    target = _corner_cloud()
    source = target + 0.05
    out, fit = icp_align(source, target, max_corr_dist=0.1, min_points=50)
    assert np.array_equal(out, source) and fit == 0.0
