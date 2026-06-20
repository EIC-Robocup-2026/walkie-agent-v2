"""Point-cloud denoising: DBSCAN clustering + statistical outlier removal.

ConceptGraphs denoises every per-detection point cloud by running DBSCAN and
keeping only the **largest cluster**, which drops the depth outliers that bleed in
around a mask's edges (flying pixels, background slivers) and would otherwise
inflate the object's bounding box and shift its centroid. The reference uses
``open3d``'s ``cluster_dbscan`` (``concept-graphs/.../slam/utils.py::pcd_denoise_dbscan``).
On top of that we use Open3D's ``remove_statistical_outlier`` (SOR) for the fuzz
*accumulated* clouds collect across sightings — see
:func:`statistical_outlier_removal`.

Every operation has two backends, same semantics:

- **library fast path** (when installed): scikit-learn's C DBSCAN; Open3D's C++ SOR.
- **pure numpy + scipy fallback**: textbook DBSCAN via :class:`scipy.spatial.cKDTree`
  + union-find, and a cKDTree kNN implementation of SOR — so a partial install still
  works. For DBSCAN: a point is a **core** point if it has at least ``min_points``
  neighbours within ``eps`` (counting itself); core points within ``eps`` of each
  other join one cluster; non-core points within ``eps`` of a core point are
  **border** points; the rest is noise (label ``-1``).
"""

from __future__ import annotations

import numpy as np

try:  # scipy is a hard dep (pyproject); guard only so a partial install fails late
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:  # sklearn is a hard dep too, but the scipy fallback keeps us running without it
    from sklearn.cluster import DBSCAN as _SKDBSCAN
except Exception:  # pragma: no cover
    _SKDBSCAN = None

# Open3D is imported lazily (the import costs ~2 s); _O3D is the cached module or
# False once the import has been attempted and failed. Tests monkeypatch this.
_O3D = None


def _open3d():
    """Return the open3d module, importing it on first use (False if unavailable)."""
    global _O3D
    if _O3D is None:
        try:
            import open3d  # noqa: PLC0415 — deliberate lazy import (slow)

            # Mute Open3D's C++ logger (writes straight to stderr, bypassing
            # Python logging). Its Warning level is pure noise for our loop —
            # chiefly ICP's "0 correspondence present between the pointclouds",
            # which just means fitness=0; every caller already gates on fitness
            # and falls back to "no correction", so the message is informational.
            try:
                open3d.utility.set_verbosity_level(
                    open3d.utility.VerbosityLevel.Error
                )
            except Exception:  # noqa: BLE001 — verbosity API is best-effort
                pass

            _O3D = open3d
        except Exception:  # pragma: no cover — wheel missing on this platform
            _O3D = False
    return _O3D


class _UnionFind:
    """Minimal array-backed union-find with path compression + union by size."""

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]


def dbscan_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Return per-point DBSCAN cluster labels (``-1`` = noise) for an ``(N, 3)`` cloud.

    Uses scikit-learn's DBSCAN when available (fast C path), otherwise the scipy
    fallback below (deterministic: border points join the lowest-indexed core
    neighbour). Returns an all-``-1`` array when there are too few points to form a
    core point.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0 or min_points <= 0 or n < min_points:
        return labels

    if _SKDBSCAN is not None:
        return _SKDBSCAN(eps=float(eps), min_samples=int(min_points)).fit_predict(pts)
    if cKDTree is None:  # pragma: no cover — scipy and sklearn both absent
        return labels

    tree = cKDTree(pts)
    # Neighbour index lists within eps (each list includes the point itself).
    neighbors = tree.query_ball_tree(tree, r=float(eps))
    is_core = np.fromiter(
        (len(neighbors[i]) >= min_points for i in range(n)), dtype=bool, count=n
    )
    if not is_core.any():
        return labels

    # Union core points that are mutually within eps.
    uf = _UnionFind(n)
    for i in range(n):
        if not is_core[i]:
            continue
        for j in neighbors[i]:
            if is_core[j]:
                uf.union(i, j)

    # Assign a dense cluster id per core-point root.
    root_to_cluster: dict[int, int] = {}
    for i in range(n):
        if is_core[i]:
            r = uf.find(i)
            if r not in root_to_cluster:
                root_to_cluster[r] = len(root_to_cluster)
            labels[i] = root_to_cluster[r]

    # Border points: join the lowest-indexed core neighbour's cluster.
    for i in range(n):
        if is_core[i]:
            continue
        core_neighbors = [j for j in neighbors[i] if is_core[j]]
        if core_neighbors:
            labels[i] = labels[min(core_neighbors)]

    return labels


def dbscan_largest_cluster(
    points: np.ndarray,
    eps: float,
    min_points: int,
    *,
    min_cluster_size: int = 5,
    subsample: int = 0,
) -> np.ndarray:
    """Keep only the largest DBSCAN cluster of ``points`` (an ``(N, 3)`` cloud).

    Mirrors ConceptGraphs' ``pcd_denoise_dbscan``: cluster, drop the noise label,
    keep the most populous cluster. Falls back to the **original** cloud when there
    are too few points to cluster or the largest cluster is smaller than
    ``min_cluster_size`` (so a sparse-but-real detection is never thrown away).

    ``subsample`` > 0 bounds the DBSCAN cost on dense clouds: the clustering runs on
    a uniform-stride subset of at most that many points, then every original point
    within ``eps`` of the winning cluster is kept — same verdict, a fraction of the
    cost (DBSCAN's neighbour queries scale superlinearly with density).
    """
    pts = np.asarray(points)
    if len(pts) < max(min_points, min_cluster_size):
        return pts

    if subsample and len(pts) > subsample and cKDTree is not None:
        idx = np.linspace(0, len(pts) - 1, int(subsample)).astype(np.int64)
        sub = pts[idx]
        labels = dbscan_labels(sub, eps, min_points)
        valid = labels[labels >= 0]
        if valid.size == 0:
            return pts
        counts = np.bincount(valid)
        best = int(counts.argmax())
        if counts[best] < min_cluster_size:
            return pts
        cluster = sub[labels == best]
        # Map the verdict back: keep every full-res point within eps of the cluster.
        d, _ = cKDTree(cluster).query(pts, k=1)
        return pts[np.asarray(d) <= eps]

    labels = dbscan_labels(pts, eps, min_points)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pts

    counts = np.bincount(valid)
    best = int(counts.argmax())
    if counts[best] < min_cluster_size:
        return pts
    return pts[labels == best]


def dbscan_remove_noise(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """Drop only DBSCAN **noise** points (label ``-1``), keeping every real cluster.

    The right cleanup for an object cloud *accumulated across views*: disjoint partial
    sightings (the two ends of a bed, middle never seen) legitimately form multiple
    clusters, which a largest-cluster keep would truncate — here they all survive, and
    only isolated scatter is removed. Falls back to the **original** cloud when there
    are too few points to cluster or everything is labelled noise (a sparse-but-real
    cloud is never thrown away).
    """
    pts = np.asarray(points)
    if len(pts) < min_points:
        return pts
    labels = dbscan_labels(pts, eps, min_points)
    keep = labels >= 0
    if not keep.any():
        return pts
    return pts[keep]


def statistical_outlier_removal(
    points: np.ndarray, k: int = 16, std_ratio: float = 2.0
) -> np.ndarray:
    """Drop low-density outlier points (the classic flying-pixel / halo cleaner).

    A point is removed when its mean distance to its ``k`` nearest neighbours exceeds
    ``global_mean + std_ratio * global_std`` of those means — i.e. it sits in
    anomalously sparse space. This both strips per-frame flying pixels off a lifted
    cloud (regardless of how far the real surface spreads in depth — a grazing bed
    stays dense and survives whole) and erases the fuzzy halo an *accumulated* cloud
    collects across sightings, while legitimately dense structure — including disjoint
    multi-view clusters — survives, which neither a largest-cluster keep nor noise-only
    DBSCAN can guarantee.

    Uses Open3D's C++ ``remove_statistical_outlier`` when available (the library
    ConceptGraphs builds on); otherwise an equivalent cKDTree kNN fallback. No-op when
    disabled (``k <= 0`` / ``std_ratio <= 0``) or the cloud is too small (``n <= k``)
    to estimate local density.
    """
    pts = np.asarray(points)
    if k <= 0 or std_ratio <= 0 or len(pts) <= k:
        return pts

    o3d = _open3d()
    if o3d:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        _, idx = pc.remove_statistical_outlier(nb_neighbors=int(k), std_ratio=float(std_ratio))
        if len(idx) == 0:  # degenerate spread → keep everything rather than wipe the cloud
            return pts
        return pts[np.asarray(idx, dtype=np.int64)]

    if cKDTree is None:  # pragma: no cover — scipy and open3d both absent
        return pts
    # Fallback: the same algorithm on cKDTree kNN distances (k+1 includes self).
    dists, _ = cKDTree(pts.astype(np.float64)).query(pts.astype(np.float64), k=k + 1)
    mean_d = dists[:, 1:].mean(axis=1)
    keep = mean_d <= mean_d.mean() + std_ratio * mean_d.std()
    if not keep.any():  # degenerate spread → keep everything rather than wipe the cloud
        return pts
    return pts[keep]
