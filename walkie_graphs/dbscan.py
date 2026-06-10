"""DBSCAN point-cloud denoising.

ConceptGraphs denoises every per-detection point cloud by running DBSCAN and
keeping only the **largest cluster**, which drops the depth outliers that bleed in
around a mask's edges (flying pixels, background slivers) and would otherwise
inflate the object's bounding box and shift its centroid. The reference uses
``open3d``'s ``cluster_dbscan`` (``concept-graphs/.../slam/utils.py::pcd_denoise_dbscan``).

Two backends, same semantics:

- **scikit-learn** (preferred when installed): its C-implemented DBSCAN is the
  battle-tested fast path — several times quicker on dense clouds.
- **pure numpy + scipy fallback**: textbook DBSCAN via :class:`scipy.spatial.cKDTree`
  + union-find, so a partial install (no sklearn) still works. A point is a **core**
  point if it has at least ``min_points`` neighbours within ``eps`` (counting
  itself); core points within ``eps`` of each other join one cluster; non-core
  points within ``eps`` of a core point are **border** points; the rest is noise
  (label ``-1``).
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
    points: np.ndarray, eps: float, min_points: int, *, min_cluster_size: int = 5
) -> np.ndarray:
    """Keep only the largest DBSCAN cluster of ``points`` (an ``(N, 3)`` cloud).

    Mirrors ConceptGraphs' ``pcd_denoise_dbscan``: cluster, drop the noise label,
    keep the most populous cluster. Falls back to the **original** cloud when there
    are too few points to cluster or the largest cluster is smaller than
    ``min_cluster_size`` (so a sparse-but-real detection is never thrown away).
    """
    pts = np.asarray(points)
    if len(pts) < max(min_points, min_cluster_size):
        return pts

    labels = dbscan_labels(pts, eps, min_points)
    valid = labels[labels >= 0]
    if valid.size == 0:
        return pts

    counts = np.bincount(valid)
    best = int(counts.argmax())
    if counts[best] < min_cluster_size:
        return pts
    return pts[labels == best]
