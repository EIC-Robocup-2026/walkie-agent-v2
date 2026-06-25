"""Pure-geometry unit tests for horizontal-surface detection and placement search.

No robot, no AI server, no Open3D — synthetic numpy point clouds exercise the
height-clustering, support-surface lookup, object-to-surface assignment, and
empty-space search in interfaces/perception/surfaces.py. The arm/nav placement path
(execute_place, place_object) is integration-level and validated on-robot.
"""

import numpy as np
import pytest

from interfaces.perception import surfaces as S
from interfaces.perception.surfaces import (
    SurfacePlane,
    assign_objects_to_surfaces,
    detect_horizontal_surfaces,
    find_free_placement,
    support_surface_for,
)


def slab(x0, x1, y0, y1, z, *, step=0.02):
    """A dense horizontal sheet of points over an XY rectangle at height z."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, float(z))])


def column(x0, x1, y0, y1, z0, z1, *, step=0.02):
    """A small box of points (an object / obstacle sitting above a surface)."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    zs = np.arange(z0, z1, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


# --- detect_horizontal_surfaces ---------------------------------------------
def test_two_tables_distinct_z():
    cloud = np.vstack([slab(0, 1, 0, 1, 0.5), slab(2, 3, 0, 1, 0.8)])
    surfaces = detect_horizontal_surfaces(cloud)
    assert len(surfaces) == 2
    # Sorted highest-first.
    assert surfaces[0].z == pytest.approx(0.8, abs=0.02)
    assert surfaces[1].z == pytest.approx(0.5, abs=0.02)
    # Disjoint in X (one table at x[0,1], the other at x[2,3]).
    hi, lo = surfaces[0], surfaces[1]
    assert hi.aabb_min[0] >= 1.9 and lo.aabb_max[0] <= 1.1


def test_same_height_two_tables_split_in_xy():
    # Two slabs at the SAME height, separated in X -> the XY DBSCAN must split them.
    # Realistic density (1 cm) so DBSCAN forms cores; a sparser band intentionally
    # degrades to one merged surface (no reliable split signal).
    cloud = np.vstack([
        slab(0, 0.6, 0, 0.6, 0.7, step=0.01),
        slab(1.5, 2.1, 0, 0.6, 0.7, step=0.01),
    ])
    surfaces = detect_horizontal_surfaces(cloud)
    assert len(surfaces) == 2
    for s in surfaces:
        assert s.z == pytest.approx(0.7, abs=0.02)


def test_single_table_one_surface():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    assert len(surfaces) == 1
    s = surfaces[0]
    assert s.z == pytest.approx(0.7, abs=0.02)
    assert s.area == pytest.approx(1.0, abs=0.1)
    assert s.contains_xy(0.5, 0.5)
    assert not s.contains_xy(2.0, 2.0)


def test_too_few_points_returns_empty():
    assert detect_horizontal_surfaces(np.zeros((10, 3))) == []


def test_object_on_surface_does_not_raise_z():
    # Regression: an object resting ON a table must NOT pull the detected surface height
    # up. The table is a dense sheet at z=0.70; an object sits on it, its base contiguous
    # with the table (within z_gap_m) and its footprint inside the table's, so its points
    # join the same height-band cluster. The old 90th-percentile estimator reported ~0.80
    # here (biased upward by the object); the modal (densest-layer) estimator reports the
    # true tabletop 0.70.
    table = slab(0, 0.6, 0, 0.6, 0.70)              # dense flat sheet, 900 pts @ 0.70
    obj = column(0.2, 0.4, 0.2, 0.4, 0.70, 0.84)    # 14 cm tall box resting on the table
    surfaces = detect_horizontal_surfaces(np.vstack([table, obj]))
    assert len(surfaces) == 1
    assert surfaces[0].z == pytest.approx(0.70, abs=0.02)


# --- minimum flat-area threshold --------------------------------------------
def test_area_is_true_coverage_not_bbox():
    # An L-shaped table: bounding box ~1 m^2, but the L only covers ~0.28 m^2. The
    # reported area must be the true coverage, not the inflated bounding box.
    L = np.vstack([
        slab(0, 1.0, 0, 0.15, 0.7, step=0.01),  # bottom bar
        slab(0, 0.15, 0, 1.0, 0.7, step=0.01),  # left bar
    ])
    surfaces = detect_horizontal_surfaces(L)
    assert len(surfaces) == 1
    s = surfaces[0]
    assert s.bbox_area > 0.9              # bounding box spans the full 1x1
    assert s.area_m2 == pytest.approx(0.28, abs=0.08)  # true coverage is far smaller
    assert s.area == s.area_m2 < s.bbox_area


def test_min_area_rejects_small_surface():
    # A dense 10x10 cm patch (~0.01 m^2) has plenty of points but too little flat
    # area -> rejected by the default 0.05 m^2 threshold.
    patch = slab(0, 0.10, 0, 0.10, 0.7, step=0.005)
    assert patch.shape[0] > 150  # not rejected for lack of points
    assert detect_horizontal_surfaces(patch) == []
    # ...but a lower threshold lets it through.
    surfaces = detect_horizontal_surfaces(patch, min_area_m2=0.005)
    assert len(surfaces) == 1
    assert surfaces[0].area_m2 == pytest.approx(0.01, abs=0.005)


def test_min_area_accepts_large_surface_and_reports_area():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    assert len(surfaces) == 1
    assert surfaces[0].area_m2 == pytest.approx(1.0, abs=0.1)


# --- support_surface_for / assign_objects_to_surfaces -----------------------
def test_object_assigned_to_correct_surface():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    assert len(surfaces) == 1
    obj = (0.5, 0.5, 0.78)  # sitting 8 cm above the table top
    sup = support_surface_for(surfaces, *obj)
    assert sup is surfaces[0]

    assignment = assign_objects_to_surfaces(surfaces, [("can", obj)])
    on = assignment[surfaces[0].id]
    assert len(on) == 1 and on[0]["label"] == "can"
    assert on[0]["height_above"] == pytest.approx(0.08, abs=0.02)


def test_support_surface_picks_nearest_below():
    # Two stacked surfaces; an object just above the upper one supports on the upper.
    cloud = np.vstack([slab(0, 1, 0, 1, 0.5), slab(0, 1, 0, 1, 0.9)])
    surfaces = detect_horizontal_surfaces(cloud)
    assert len(surfaces) == 2
    sup = support_surface_for(surfaces, 0.5, 0.5, 0.95)
    assert sup is not None and sup.z == pytest.approx(0.9, abs=0.02)


def test_object_above_max_gap_has_no_support():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    assert support_surface_for(surfaces, 0.5, 0.5, 1.1, max_gap_m=0.20) is None


def test_object_off_surface_xy_has_no_support():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    assert support_surface_for(surfaces, 5.0, 5.0, 0.78) is None
    assignment = assign_objects_to_surfaces(surfaces, [("can", (5.0, 5.0, 0.78))])
    assert assignment[-1] and assignment[-1][0]["label"] == "can"


# --- find_free_placement ----------------------------------------------------
def test_find_free_placement_avoids_obstacle():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    surface = surfaces[0]
    obstacle = column(0.16, 0.26, 0.16, 0.26, 0.75, 0.82)
    cloud = np.vstack([slab(0, 1, 0, 1, 0.7), obstacle])
    xy = find_free_placement(surface, cloud, footprint_m=0.12, clearance_m=0.03,
                             cell_m=0.04, edge_margin_m=0.05, prefer="center")
    assert xy is not None
    x, y = xy
    # Inside the edge-shrunk surface.
    assert 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95
    # Clear of the obstacle by at least the half-footprint + clearance.
    assert np.hypot(x - 0.21, y - 0.21) >= 0.09


def test_find_free_placement_prefers_near_xy():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    xy = find_free_placement(surfaces[0], slab(0, 1, 0, 1, 0.7),
                             prefer="near", prefer_xy=(0.1, 0.1))
    assert xy is not None
    # The empty table -> chosen cell hugs the requested corner, not the centre.
    assert xy[0] < 0.4 and xy[1] < 0.4


def test_find_free_placement_full_surface_returns_none():
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7))
    covered = np.vstack([slab(0, 1, 0, 1, 0.7), slab(0, 1, 0, 1, 0.85)])
    assert find_free_placement(surfaces[0], covered) is None


def test_surface_skin_ignores_own_noise_tail():
    # surface.z is the modal (band-centre) plane; the surface's OWN sensor-noise tail can
    # sit a few cm above it. surface_skin_m lifts the occupancy floor clear of that skin,
    # so a layer 4 cm above the plane (within skin+clearance = 0.05) is NOT mistaken for
    # clutter and the bare table stays placeable. Without the skin the floor sits at +0.03
    # and this layer would self-occupy the whole surface -> a false "no free space".
    s = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.70))[0]
    skin_tail = np.vstack([slab(0, 1, 0, 1, 0.70), slab(0, 1, 0, 1, 0.74)])
    assert find_free_placement(s, skin_tail, surface_skin_m=0.02, clearance_m=0.03) is not None
    # ...but a real object well above the skin DOES occupy and fill the surface.
    clutter = np.vstack([slab(0, 1, 0, 1, 0.70), slab(0, 1, 0, 1, 0.80)])
    assert find_free_placement(s, clutter, surface_skin_m=0.02, clearance_m=0.03) is None


# --- graceful degradation without Open3D ------------------------------------
def test_detect_degrades_without_open3d(monkeypatch):
    # Normal gating needs Open3D; with it forced off, the gate is skipped and the
    # pure height-clustering path still finds the surface.
    monkeypatch.setattr(S, "_O3D", False)
    monkeypatch.setattr(S, "_normal_warned", False)
    surfaces = detect_horizontal_surfaces(slab(0, 1, 0, 1, 0.7), normal_z_min=0.9)
    assert len(surfaces) == 1
    assert surfaces[0].z == pytest.approx(0.7, abs=0.02)


# --- SurfacePlane helpers ---------------------------------------------------
def test_surface_distance_xy():
    s = SurfacePlane(
        id=0, z=0.7, centroid=(0.5, 0.5, 0.7),
        aabb_min=(0.0, 0.0, 0.7), aabb_max=(1.0, 1.0, 0.7),
        extent=(1.0, 1.0, 0.0), n_points=100,
    )
    assert s.distance_xy(0.5, 0.5) == pytest.approx(0.0)  # inside
    assert s.distance_xy(2.0, 0.5) == pytest.approx(1.0)  # 1 m past the +x edge
