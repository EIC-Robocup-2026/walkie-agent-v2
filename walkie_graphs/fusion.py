"""ConceptGraphs-style object-association math — pure numpy + scipy.

The reference associates a new detection with an existing object from two cues
(``concept-graphs/.../slam/mapping.py``):

* **geometric** ``phi_geo`` = *nearest-neighbour overlap ratio* (``nn_ratio``): the
  fraction of the detection's points that have a neighbour in the object's cloud
  within a small radius (the voxel size / ``δ_nn`` ≈ 2.5 cm). This is the proportion
  of the detection that physically coincides with the stored object — far sharper
  than comparing centroids, which can't tell a mug from the table under it.
* **semantic** ``phi_sem`` = CLIP cosine, normalised from ``[-1, 1]`` to ``[0, 1]``
  as ``0.5·cos + 0.5`` (the paper's form).

They are combined additively, ``phi = w_geo·phi_geo + w_sem·phi_sem`` (the paper's
``(1+phys_bias)`` / ``(1-phys_bias)`` weighting with ``phys_bias = 0`` → both 1), and
the detection merges into the highest-scoring object above ``δ_sim`` (default 1.1).

Kept out of :mod:`walkie_graphs.memory` so the math is unit-testable on bare arrays.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def nn_ratio(obj_points: np.ndarray, det_points: np.ndarray, voxel_m: float) -> float:
    """Fraction of ``det_points`` with a nearest neighbour in ``obj_points`` ≤ ``voxel_m``.

    This is ConceptGraphs' ``compute_overlap_matrix_2set`` overlap (one map object vs
    one detection): build a KD-tree on the object cloud, query each detection point's
    nearest neighbour, count those within ``voxel_m``, divide by the detection size.
    Returns 0.0 when either cloud is empty. Result is in ``[0, 1]``.
    """
    obj = np.asarray(obj_points, dtype=np.float64)
    det = np.asarray(det_points, dtype=np.float64)
    if len(obj) == 0 or len(det) == 0 or cKDTree is None:
        return 0.0
    tree = cKDTree(obj)
    dists, _ = tree.query(det, k=1)
    within = int(np.count_nonzero(np.asarray(dists) <= voxel_m))
    return within / len(det)


def nn_ratio_symmetric(a_points: np.ndarray, b_points: np.ndarray, voxel_m: float) -> float:
    """Symmetric overlap = ``max(nn_ratio(a, b), nn_ratio(b, a))``.

    Used when comparing two established nodes (periodic re-merge), where neither is
    privileged as "the detection"; the max captures "one is largely contained in the
    other" regardless of which is bigger.
    """
    return max(
        nn_ratio(a_points, b_points, voxel_m),
        nn_ratio(b_points, a_points, voxel_m),
    )


def aabb_overlap(a_min, a_max, b_min, b_max, *, pad: float = 0.0) -> bool:
    """Do two axis-aligned boxes intersect (optionally grown by ``pad`` on every side)?

    The cheap O(1) prefilter before the O(points) ``nn_ratio`` — ConceptGraphs gates
    its overlap computation on 3D-bbox IoU > 0 for the same reason.
    """
    for i in range(3):
        if a_max[i] + pad < b_min[i] - pad or b_max[i] + pad < a_min[i] - pad:
            return False
    return True


def phi_sem(cos: float) -> float:
    """Normalise a CLIP cosine from ``[-1, 1]`` to ``[0, 1]`` (paper's ``0.5·cos + 0.5``)."""
    return 0.5 * float(cos) + 0.5


def additive_similarity(
    nnratio: float, cos: float, *, w_geo: float = 1.0, w_sem: float = 1.0
) -> float:
    """Combined association score ``w_geo·nnratio + w_sem·phi_sem(cos)`` (range ``[0, 2]``)."""
    return w_geo * float(nnratio) + w_sem * phi_sem(cos)
