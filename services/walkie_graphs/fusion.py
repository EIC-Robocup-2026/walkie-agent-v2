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


def pairs_within(centroids: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Index pairs (i < j) of ``centroids`` within ``radius`` of each other.

    KD-tree pair query — the prefilter that keeps the periodic node-merge pass
    O(nearby pairs) instead of O(N²) over the whole map.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    if len(pts) < 2 or cKDTree is None:
        return []
    return sorted(cKDTree(pts).query_pairs(r=float(radius)))


def aabb_overlap(a_min, a_max, b_min, b_max, *, pad: float = 0.0) -> bool:
    """Do two axis-aligned boxes intersect (optionally grown by ``pad`` on every side)?

    The cheap O(1) prefilter before the O(points) ``nn_ratio`` — ConceptGraphs gates
    its overlap computation on 3D-bbox IoU > 0 for the same reason.
    """
    for i in range(3):
        if a_max[i] + pad < b_min[i] - pad or b_max[i] + pad < a_min[i] - pad:
            return False
    return True


def subtract_contained_masks(
    bboxes: np.ndarray,
    masks: list,
    *,
    th_contained: float = 0.8,
    th_container: float = 0.7,
) -> list:
    """Subtract each contained detection's mask from its container's mask.

    ConceptGraphs' ``mask_subtract_contained``: when box *j* sits mostly inside box *i*
    (the intersection covers > ``th_contained`` of box *j* but < ``th_container`` of
    box *i*), the mug-on-the-table case, then ``mask_i &= ~mask_j`` — so the table's
    pixels no longer include the mug. Without this, the container's point cloud and
    CLIP crop are polluted by every object resting on/in it.

    ``bboxes`` is ``(N, 4)`` xyxy; ``masks`` a list of ``(H, W)`` bool arrays (``None``
    entries pass through untouched). Returns a new list; inputs are not mutated.
    """
    n = len(masks)
    out = [m.copy() if m is not None else None for m in masks]
    if n < 2:
        return out
    xyxy = np.asarray(bboxes, dtype=np.float64).reshape(n, 4)
    areas = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])

    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    inter = np.clip(rb - lt, 0.0, None)
    inter_areas = inter[:, :, 0] * inter[:, :, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        inter_over_i = np.where(areas[:, None] > 0, inter_areas / areas[:, None], 0.0)
    inter_over_j = inter_over_i.T

    # contained[i, j] = "box j is contained by box i" → subtract mask j from mask i.
    contained = (inter_over_i < th_container) & (inter_over_j > th_contained)
    np.fill_diagonal(contained, False)
    for i, j in zip(*contained.nonzero()):
        if out[i] is None or masks[j] is None:
            continue
        if out[i].shape != masks[j].shape:
            continue
        out[i] &= ~masks[j].astype(bool)
    return out


def phi_sem(cos: float) -> float:
    """Normalise a CLIP cosine from ``[-1, 1]`` to ``[0, 1]`` (paper's ``0.5·cos + 0.5``)."""
    return 0.5 * float(cos) + 0.5


def additive_similarity(
    nnratio: float, cos: float, *, w_geo: float = 1.0, w_sem: float = 1.0
) -> float:
    """Combined association score ``w_geo·nnratio + w_sem·phi_sem(cos)`` (range ``[0, 2]``)."""
    return w_geo * float(nnratio) + w_sem * phi_sem(cos)


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_corr_dist: float,
    *,
    min_fitness: float = 0.6,
    min_points: int = 150,
) -> tuple[np.ndarray, float]:
    """Rigidly align ``source`` onto ``target`` with Open3D ICP before fusing them.

    The camera pose carries a few cm of error per sighting, so two clouds of the same
    object land slightly offset and a naive union double-exposes the shape. ICP
    (Iterative Closest Point — what object-level SLAM systems like Fusion++/MaskFusion
    run per object) recovers that residual offset from the clouds' own geometry.

    Returns ``(aligned_source, fitness)``. The alignment is applied only when ICP's
    fitness (the fraction of source points that found a correspondence within
    ``max_corr_dist``) reaches ``min_fitness`` — a low score means the clouds barely
    overlap (e.g. a partial view showing a *new* part of the object), where ICP would
    wrongly snap distinct surfaces together; the source is then returned unchanged.
    Also unchanged when either cloud is below ``min_points`` (too little shape to
    constrain alignment — a small cup can't anchor ICP), when disabled
    (``max_corr_dist <= 0``), or when Open3D is unavailable (fitness 0.0).
    """
    src = np.asarray(source)
    if (
        max_corr_dist <= 0
        or len(src) < min_points
        or len(np.asarray(target)) < min_points
    ):
        return src, 0.0

    from .dbscan import _open3d

    o3d = _open3d()
    if not o3d:
        return src, 0.0

    reg = o3d.pipelines.registration
    # Estimate the transform on bounded subsamples (uniform stride — keeps coverage),
    # then apply it to the FULL source: rigid-transform accuracy barely depends on
    # density past ~2000 points, while ICP cost is superlinear in it.
    cap = 2000
    src_est = src if len(src) <= cap else src[np.linspace(0, len(src) - 1, cap).astype(int)]
    tgt = np.asarray(target, dtype=np.float64)
    tgt_est = tgt if len(tgt) <= cap else tgt[np.linspace(0, len(tgt) - 1, cap).astype(int)]
    src_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_est.astype(np.float64)))
    tgt_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_est))
    result = reg.registration_icp(
        src_pc,
        tgt_pc,
        float(max_corr_dist),
        np.eye(4),
        reg.TransformationEstimationPointToPoint(),
        # Defaults stop after 30 iterations / loose epsilons; a few-cm offset on a
        # dense cloud needs the extra headroom (still ~ms at our cloud sizes).
        reg.ICPConvergenceCriteria(
            max_iteration=100, relative_fitness=1e-9, relative_rmse=1e-9
        ),
    )
    fitness = float(result.fitness)
    if fitness < min_fitness:
        return src, fitness
    T = np.asarray(result.transformation)
    aligned = src.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
    return aligned.astype(np.float32), fitness
