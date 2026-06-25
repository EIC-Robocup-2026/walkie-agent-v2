"""Horizontal-surface extraction from a map-frame point cloud — pure numpy.

The map frame is gravity-aligned (+Z up — the grasp skill already relies on this,
deriving world-up as ``snap.cam.R.T @ [0, 0, 1]``). That collapses *horizontal*
surface finding from a 4-DOF plane fit to a 1-D problem: a tabletop / shelf / floor
is a dense band of roughly-constant Z. So instead of RANSAC (``segment_plane`` isn't
even exposed in this repo), we **cluster points by height**: sort Z, split into runs
at vertical gaps, then split each height band in XY (DBSCAN) so two tables at the
same height come back as two surfaces.

Everything here is pure ``numpy`` + ``scipy`` and reuses
:func:`interfaces.perception.dbscan.dbscan_labels` (sklearn fast path, scipy
fallback). Open3D is only touched for the *optional* surface-normal gate
(:func:`detect_horizontal_surfaces` ``normal_z_min``), and it degrades to skipping
the gate when Open3D is missing — placement never requires Open3D.

    surfaces = detect_horizontal_surfaces(cloud)            # tables/shelves, top-down
    sup = support_surface_for(surfaces, gx, gy, gz)         # which one an object sits on
    xy = find_free_placement(surfaces[0], cloud, footprint_m=0.12)  # a clear spot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dbscan import dbscan_labels, statistical_outlier_removal
from .geometry import voxel_downsample  # noqa: F401 — re-exported for callers' convenience

try:  # scipy is a hard dep; binary_dilation grows the placement occupancy grid
    from scipy.ndimage import binary_dilation
except Exception:  # pragma: no cover — partial install
    binary_dilation = None

# Open3D import is lazy + cached (same pattern as dbscan._open3d); tests monkeypatch
# this to False to exercise the no-normals path. ``None`` = not yet attempted.
_O3D = None


def _open3d():
    """Return the open3d module, importing on first use (False if unavailable)."""
    global _O3D
    if _O3D is None:
        try:
            import open3d  # noqa: PLC0415 — deliberate lazy import (slow)

            _O3D = open3d
        except Exception:  # pragma: no cover — wheel missing on this platform
            _O3D = False
    return _O3D


_normal_warned = False


@dataclass
class SurfacePlane:
    """A horizontal support surface in the map frame (gravity = +Z up).

    ``z`` is the surface plane height — the **modal** (densest) Z layer of the band,
    not a percentile or the mean, so it stays on the true tabletop and an object resting
    on it sits just above ``z`` instead of dragging ``z`` up. ``aabb_min``/``aabb_max`` are
    the axis-aligned bounds of the surface points; ``extent`` is their span
    ``(w, d, h)``. ``points`` is the lifted surface cloud when ``keep_points=True``,
    else ``None``.
    """

    id: int
    z: float
    centroid: tuple[float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]
    extent: tuple[float, float, float]
    n_points: int
    area_m2: float = 0.0  # true flat area: occupied XY cells, not the bounding box
    points: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> float:
        """The surface's true flat area in m^2 (occupied XY-cell coverage).

        This is the real usable area, not the bounding box: an L-shaped table or a
        sparse scatter has a large ``bbox_area`` but a much smaller ``area``.
        """
        return float(self.area_m2)

    @property
    def bbox_area(self) -> float:
        """Area of the XY bounding box (m^2) — an upper bound on the real area."""
        return float(self.extent[0] * self.extent[1])

    def contains_xy(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        """Whether ``(x, y)`` lies inside the XY AABB, optionally shrunk by ``margin``."""
        return (
            self.aabb_min[0] + margin <= x <= self.aabb_max[0] - margin
            and self.aabb_min[1] + margin <= y <= self.aabb_max[1] - margin
        )

    def distance_xy(self, x: float, y: float) -> float:
        """Planar distance from ``(x, y)`` to the AABB (0 when inside)."""
        dx = max(self.aabb_min[0] - x, 0.0, x - self.aabb_max[0])
        dy = max(self.aabb_min[1] - y, 0.0, y - self.aabb_max[1])
        return float(np.hypot(dx, dy))


def _gate_by_normal(points: np.ndarray, normal_z_min: float) -> np.ndarray:
    """Keep points whose estimated surface normal is near-vertical (|n_z| >= thresh).

    Rejects walls and other vertical structure before height clustering. Needs
    Open3D's ``estimate_normals``; degrades to a no-op (returns *points*) with a
    one-time warning when Open3D is unavailable.
    """
    global _normal_warned
    o3d = _open3d()
    if not o3d:
        if not _normal_warned:
            print("[surfaces] Open3D unavailable; skipping normal gate")
            _normal_warned = True
        return points
    if len(points) < 3:
        return points
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pc.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pc.normals)
    if normals.shape[0] != len(points):  # pragma: no cover — degenerate estimate
        return points
    keep = np.abs(normals[:, 2]) >= float(normal_z_min)
    return points[keep] if keep.any() else points


def _split_runs(values_sorted: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """Split a sorted 1-D array into contiguous runs wherever the step exceeds *gap*.

    Returns ``(start, stop)`` index pairs into *values_sorted* (stop exclusive).
    """
    if values_sorted.size == 0:
        return []
    breaks = np.where(np.diff(values_sorted) > gap)[0] + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [values_sorted.size]))
    return list(zip(starts.tolist(), stops.tolist()))


def _plane_height(z: np.ndarray, bin_m: float) -> float:
    """Robust horizontal-plane height: the densest Z layer (mode), not a percentile.

    A real surface contributes a dense *sheet* of points at one height; objects resting
    on it add comparatively few points spread above. A high percentile is biased upward
    by those objects (the place height then reconstructs too high), so we take the modal
    Z — the peak of a fine histogram (bin width ``bin_m``) — and refine it with the mean
    of the points that fall in the peak bin. Degrades to the median for tiny inputs or a
    Z span below one bin, so a flat slab returns its exact height.
    """
    z = np.asarray(z, dtype=np.float64).ravel()
    if z.size == 0:
        return 0.0
    lo, hi = float(z.min()), float(z.max())
    if bin_m <= 0 or z.size < 8 or (hi - lo) < bin_m:
        return float(np.median(z))
    nbins = max(1, int(np.ceil((hi - lo) / bin_m)))
    counts, edges = np.histogram(z, bins=nbins)
    k = int(np.argmax(counts))
    in_peak = (z >= edges[k]) & (z <= edges[k + 1])
    return float(np.mean(z[in_peak])) if in_peak.any() else float(np.median(z))


def _covered_area_m2(xy: np.ndarray, cell_m: float) -> float:
    """True flat area: count of distinct occupied (x, y) cells times the cell area.

    Unlike the XY bounding box — which an L-shaped table, a ring, or a sparse scatter
    inflate — this counts only cells that actually hold surface points, giving a
    faithful m^2 measure of the usable flat region. ``cell_m`` is the grid resolution
    of the area estimate (coarser than the cloud voxel is fine and cheaper).
    """
    if cell_m <= 0 or len(xy) == 0:
        return 0.0
    ij = np.floor(xy / cell_m).astype(np.int64)
    ij -= ij.min(axis=0)
    flat = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    return int(np.unique(flat).size) * cell_m * cell_m


def detect_horizontal_surfaces(
    points: np.ndarray,
    *,
    z_gap_m: float = 0.04,
    min_points_per_surface: int = 150,
    xy_cluster_eps: float = 0.05,
    xy_min_points: int = 30,
    min_area_m2: float = 0.05,
    area_cell_m: float = 0.05,
    normal_z_min: float | None = None,
    z_bin_m: float = 0.01,
    keep_points: bool = False,
    sor_k: int = 0,
) -> list[SurfacePlane]:
    """Group a gravity-aligned (+Z up) cloud into horizontal :class:`SurfacePlane`\\ s.

    Pipeline:

    1. (optional) **normal gate** — keep near-vertical-normal points when
       ``normal_z_min`` is set (needs Open3D; skipped/degraded otherwise).
    2. **height clustering** — sort Z and split into runs wherever the gap between
       consecutive heights exceeds ``z_gap_m``. Each run is a candidate surface band
       (robust to histogram bin phase, unlike peak-picking).
    3. **XY split** — within each band run DBSCAN on the (x, y) of the points
       (``xy_cluster_eps`` / ``xy_min_points``) so two tables at the same height come
       back as two surfaces.
    4. **stats + filter** — per cluster compute AABB / centroid / extent / count and
       the **true covered area** (occupied XY cells at ``area_cell_m``, not the
       bounding box); drop clusters below ``min_points_per_surface`` or
       ``min_area_m2`` (square metres of real flat area). The surface ``z`` is the
       **modal** (densest) Z layer of the cluster at ``z_bin_m`` resolution — robust to
       objects resting on the surface, which a percentile would let bias ``z`` upward.

    Bands thinner than a real surface (a stray Z-run) are filtered by the point/area
    minimums. ``min_area_m2`` is a genuine flat-area threshold: an L-shaped table or a
    sparse scatter with a big bounding box but little real coverage is rejected. A
    surface's own thickness (sensor noise) is absorbed by the run split and the modal
    height estimate. Returns surfaces sorted by ``z``, highest first.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < min_points_per_surface:
        return []

    if normal_z_min is not None:
        pts = _gate_by_normal(pts, normal_z_min)
        if pts.shape[0] < min_points_per_surface:
            return []
    if sor_k and sor_k > 0:
        pts = statistical_outlier_removal(pts, k=int(sor_k))

    order = np.argsort(pts[:, 2])
    z_sorted = pts[order, 2]
    runs = _split_runs(z_sorted, z_gap_m)

    out: list[SurfacePlane] = []
    sid = 0
    for start, stop in runs:
        if stop - start < min_points_per_surface:
            continue
        band = pts[order[start:stop]]
        labels = dbscan_labels(band[:, :2], xy_cluster_eps, xy_min_points)
        # No core points (sparse band) -> treat the whole band as one cluster so a
        # thin-but-broad surface isn't dropped just because DBSCAN found no core.
        cluster_ids = [c for c in np.unique(labels) if c >= 0]
        groups = (
            [band[labels == c] for c in cluster_ids]
            if cluster_ids
            else [band]
        )
        for cluster in groups:
            if cluster.shape[0] < min_points_per_surface:
                continue
            cmin = cluster.min(axis=0)
            cmax = cluster.max(axis=0)
            extent = cmax - cmin
            area = _covered_area_m2(cluster[:, :2], area_cell_m)
            if area < min_area_m2:
                continue
            out.append(
                SurfacePlane(
                    id=sid,
                    z=_plane_height(cluster[:, 2], z_bin_m),
                    centroid=tuple(float(v) for v in cluster.mean(axis=0)),
                    aabb_min=tuple(float(v) for v in cmin),
                    aabb_max=tuple(float(v) for v in cmax),
                    extent=tuple(float(v) for v in extent),
                    n_points=int(cluster.shape[0]),
                    area_m2=float(area),
                    points=cluster.astype(np.float32) if keep_points else None,
                )
            )
            sid += 1

    out.sort(key=lambda s: s.z, reverse=True)
    return out


def support_surface_for(
    surfaces: list[SurfacePlane],
    x: float,
    y: float,
    z: float,
    *,
    max_gap_m: float = 0.20,
) -> SurfacePlane | None:
    """The surface a point at ``(x, y, z)`` rests on.

    A surface qualifies when its XY AABB contains ``(x, y)`` and its top ``z`` is
    below the point (within ``max_gap_m``). Among qualifiers, the one with the
    smallest ``z_point - surface.z`` (the closest support directly underneath) wins.
    Returns ``None`` when nothing qualifies (e.g. the object floats above any known
    surface, or the table band was too sparse to detect).
    """
    best: SurfacePlane | None = None
    best_gap = float("inf")
    for s in surfaces:
        if not s.contains_xy(x, y):
            continue
        gap = z - s.z
        if 0.0 <= gap <= max_gap_m and gap < best_gap:
            best, best_gap = s, gap
    return best


def assign_objects_to_surfaces(
    surfaces: list[SurfacePlane],
    objects: list[tuple[str, tuple[float, float, float]]],
    *,
    max_gap_m: float = 0.20,
) -> dict[int, list[dict]]:
    """Map each surface id to the objects resting on it (for the LLM placement agent).

    ``objects`` is ``[(label, (x, y, z))]``. Returns ``{surface_id: [{label, xyz,
    height_above, xy_dist_to_centroid}]}``; objects that match no surface are
    collected under key ``-1``. Uses :func:`support_surface_for` per object so the
    "on" decision matches the rest of the module.
    """
    result: dict[int, list[dict]] = {s.id: [] for s in surfaces}
    result[-1] = []
    for label, (x, y, z) in objects:
        sup = support_surface_for(surfaces, x, y, z, max_gap_m=max_gap_m)
        key = sup.id if sup is not None else -1
        height_above = (z - sup.z) if sup is not None else None
        cxy = sup.centroid[:2] if sup is not None else (0.0, 0.0)
        result[key].append(
            {
                "label": label,
                "xyz": (float(x), float(y), float(z)),
                "height_above": (float(height_above) if height_above is not None else None),
                "xy_dist_to_centroid": float(np.hypot(x - cxy[0], y - cxy[1])),
            }
        )
    return result


def find_free_placement(
    surface: SurfacePlane,
    obstacle_points: np.ndarray,
    *,
    footprint_m: float = 0.12,
    clearance_m: float = 0.03,
    surface_skin_m: float = 0.02,
    cell_m: float = 0.04,
    edge_margin_m: float = 0.05,
    tall_cap_m: float = 0.40,
    prefer: str = "center",
    prefer_xy: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Find a clear ``(x, y)`` on *surface* big enough for the held object.

    Builds a 2-D occupancy grid over the surface AABB at ``cell_m`` resolution. A
    cell is **occupied** when any point of *obstacle_points* (typically the whole
    scene cloud) falls in it with height in ``(surface.z + surface_skin_m + clearance_m,
    surface.z + tall_cap_m)`` — i.e. something already sits there. ``tall_cap_m``
    keeps a *higher* shelf's underside or the ceiling from masking this surface.

    ``surface_skin_m`` lifts the occupancy floor clear of the surface's own sensor-noise
    band. ``surface.z`` is the modal (band-centre) height, so without a skin the upper
    tail of the surface's own points would read as obstacles, get dilated, and falsely
    fill the surface. Set it to ~the depth noise at working range (a couple cm).

    The occupancy is then dilated by the object half-footprint + clearance so the
    returned cell has room for the whole object, and the grid border is shrunk by
    ``edge_margin_m`` to keep the object off the table edge. Among the free cells the
    pick is by ``prefer``: ``"center"`` → nearest the surface centroid; ``"near"`` →
    nearest ``prefer_xy`` (e.g. the holding arm's reach). Returns the cell centre in
    map-frame ``(x, y)``, or ``None`` when no cell is free.
    """
    x0, y0 = surface.aabb_min[0], surface.aabb_min[1]
    x1, y1 = surface.aabb_max[0], surface.aabb_max[1]
    nx = max(1, int(np.ceil((x1 - x0) / cell_m)))
    ny = max(1, int(np.ceil((y1 - y0) / cell_m)))

    occ = np.zeros((nx, ny), dtype=bool)
    pts = np.asarray(obstacle_points, dtype=np.float64)
    if pts.size:
        on = (
            (pts[:, 0] >= x0)
            & (pts[:, 0] < x1)
            & (pts[:, 1] >= y0)
            & (pts[:, 1] < y1)
            & (pts[:, 2] > surface.z + surface_skin_m + clearance_m)
            & (pts[:, 2] < surface.z + tall_cap_m)
        )
        if on.any():
            ox = np.clip(((pts[on, 0] - x0) / cell_m).astype(np.int64), 0, nx - 1)
            oy = np.clip(((pts[on, 1] - y0) / cell_m).astype(np.int64), 0, ny - 1)
            occ[ox, oy] = True

    # Grow obstacles by the object half-footprint + clearance so a free cell centre
    # leaves room for the whole object. Radius in cells.
    grow = int(np.ceil((footprint_m / 2.0 + clearance_m) / cell_m))
    if grow > 0 and occ.any():
        if binary_dilation is not None:
            size = 2 * grow + 1
            occ = binary_dilation(occ, structure=np.ones((size, size), dtype=bool))
        else:  # pragma: no cover — scipy missing; coarse manual dilation
            grown = occ.copy()
            idx = np.argwhere(occ)
            for ix, iy in idx:
                grown[
                    max(0, ix - grow) : ix + grow + 1,
                    max(0, iy - grow) : iy + grow + 1,
                ] = True
            occ = grown

    free = ~occ
    # Drop cells whose centre is within edge_margin of the AABB border.
    em = int(np.ceil(edge_margin_m / cell_m))
    if em > 0:
        if em * 2 >= nx or em * 2 >= ny:  # surface too small once trimmed
            return None
        free[:em, :] = False
        free[nx - em :, :] = False
        free[:, :em] = False
        free[:, ny - em :] = False

    cells = np.argwhere(free)
    if cells.size == 0:
        return None

    cx = x0 + (cells[:, 0] + 0.5) * cell_m
    cy = y0 + (cells[:, 1] + 0.5) * cell_m
    if prefer == "near" and prefer_xy is not None:
        tx, ty = prefer_xy
    else:  # "center"
        tx, ty = surface.centroid[0], surface.centroid[1]
    d = (cx - tx) ** 2 + (cy - ty) ** 2
    i = int(np.argmin(d))
    return float(cx[i]), float(cy[i])
