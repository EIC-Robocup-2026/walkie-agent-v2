"""Batch constrained-agglomerative association for the walkie_graphs v2 build.

The v1 path associated detections *one at a time, online* against a growing store,
which structurally produced four failure modes:

* **twin fusion** — two identical objects a metre apart (same class, near-identical
  CLIP) merged because semantics alone carried the decision;
* **flat-object-into-table absorption** — a spoon lying on a table has ~100 % of its
  points inside the table cloud, so a one-sided overlap test let the table swallow it;
* **ghost duplicates** — a single object seen from two partial views never re-linked
  because online insertion locked the first node in;
* **chaining** — single-link transitive merging walked a row of adjacent chairs into
  one blob (A~B, B~C, C~D ⇒ one cluster) and, symmetrically, fragmented disjoint
  partial views of one object.

:func:`associate` runs **once** over a whole build window of already pose-corrected
world-frame detections and replaces the online path. It is deliberately pure
numpy/scipy/sklearn so the association logic is unit-testable on synthetic clouds.

The fixes, mapped to the failure modes above:

* a **hard centroid cap** (``max_dist_m``) — kills twin fusion regardless of CLIP;
* a **mutual-min overlap** gate — kills flat-object-into-table (spoon→table ≈ 1 but
  table→spoon ≈ 0, so the min ≈ 0);
* **batch** clustering over the full window — re-links the two partial views that the
  online path would have ghosted;
* **constrained agglomerative** clustering with an **anti-chaining extent veto** and
  **complete-linkage** support — stops the adjacent-chairs blob and the disjoint
  fragmentation that single-link union-find causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # scipy is a hard dep; guard so a partial install fails late, like sibling modules
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from interfaces.perception.dbscan import (
    dbscan_largest_cluster,
    statistical_outlier_removal,
)
from interfaces.perception.geometry import voxel_downsample
from services.walkie_graphs.fusion import nn_ratio
from services.walkie_graphs.scene import aabb_of, cosine

# Cap on the per-observation point count used as the *query* side of the overlap
# nearest-neighbour ratio. The ratio is statistically stable on a uniform-stride
# subsample (±3% on ~1k samples), and each candidate pair queries twice, so this bounds
# the whole batch's cost. The *tree* side keeps the full cloud — subsampling both sides
# independently misaligns the two grids and corrupts the overlap measurement, whereas a
# dense tree + a strided query is exactly what ``nn_ratio``'s ``max_query`` is for.
_MAX_QUERY_POINTS = 800


@dataclass
class Observation:
    """One raw detection in the world frame, ready for batch association.

    ``points`` is the lifted, pose-corrected ``(N, 3)`` world-frame cloud of this
    single detection. ``clip_emb`` is the (ideally L2-normalised) CLIP image embedding
    of the crop, ``[]`` if unknown.
    """

    class_name: str
    class_id: Optional[int]
    conf: float
    bbox: tuple
    caption: str
    clip_emb: list[float]
    ts: float
    points: np.ndarray

    @property
    def centroid(self) -> np.ndarray:
        """Mean of this observation's points (``(3,)`` array)."""
        return np.asarray(self.points, dtype=np.float64).mean(axis=0)


@dataclass
class ObjectObservation:
    """One associated object cluster — duck-compatible with ``SceneStore.merge``'s input.

    The denoised fused cloud and its AABB summary, a union of member captions, the
    L2-normalised mean CLIP embedding, and the provenance counters merged from the
    member :class:`Observation` s.
    """

    class_name: str
    class_id: Optional[int]
    conf: float
    captions: list[str]
    clip_emb: list[float]
    ts_first: float
    ts_last: float
    n_obs: int
    points: np.ndarray
    centroid: tuple[float, float, float]
    extent: tuple[float, float, float]
    aabb_min: tuple[float, float, float]
    aabb_max: tuple[float, float, float]


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------
@dataclass
class _Item:
    """Per-observation working state: the original obs plus prepared geometry."""

    obs: Observation
    centroid: np.ndarray            # (3,) float64
    pts: np.ndarray                 # full (N, 3) world cloud (tree side of the overlap)
    tree: object                    # cKDTree over pts (or None if unavailable)
    emb: Optional[np.ndarray]       # (D,) float64 clip embedding, or None


def _emb_of(clip_emb) -> Optional[np.ndarray]:
    """Return a float64 array for a non-empty embedding, else None."""
    if clip_emb is None:
        return None
    arr = np.asarray(clip_emb, dtype=np.float64).ravel()
    if arr.size == 0 or not np.any(arr):
        return None
    return arr


def _pair_ok(
    a: _Item,
    b: _Item,
    *,
    overlap_min: float,
    clip_min: float,
    max_dist_m: float,
    require_same_class: bool,
    voxel_m: float,
) -> bool:
    """The merge predicate for a single candidate pair (all conditions must hold)."""
    # (a) hard centroid cap — kills identical twins regardless of CLIP.
    if float(np.linalg.norm(a.centroid - b.centroid)) > max_dist_m:
        return False

    # (b) mutual-min overlap — kills flat-object→table absorption. A spoon's points are
    # ~all within the table cloud (a→b ≈ 1) but only a sliver of the table is within the
    # spoon (b→a ≈ 0), so the *minimum* is ≈ 0 and the pair is rejected. The tree side
    # stays the full cloud; only the query side is strided (``max_query``) — subsampling
    # both sides would misalign the grids and crush a real overlap to ~0.
    ab = nn_ratio(a.pts, b.pts, voxel_m, obj_tree=a.tree, max_query=_MAX_QUERY_POINTS)
    if ab < overlap_min:
        return False
    ba = nn_ratio(b.pts, a.pts, voxel_m, obj_tree=b.tree, max_query=_MAX_QUERY_POINTS)
    if min(ab, ba) < overlap_min:
        return False

    # (c) class + semantic gate.
    if require_same_class:
        if a.obs.class_name != b.obs.class_name:
            return False
        if a.emb is None or b.emb is None:
            return False
        if cosine(a.emb, b.emb) < clip_min:
            return False
    return True


def _merged_extent_ok(
    members_a: list[int],
    members_b: list[int],
    items: list[_Item],
    *,
    class_name: str,
    max_extent_by_class: dict[str, float] | None,
    default_max_extent: float,
) -> bool:
    """Anti-chaining veto: reject if the union AABB would exceed the per-class extent.

    Uses the (subsampled) member clouds — cheap and monotone enough for the guard: a
    real partial-view pair stays compact, a chained row of chairs blows past the cap on
    its long axis.
    """
    clouds = [items[k].pts for k in (members_a + members_b)]
    pts = np.concatenate(clouds, axis=0)
    span = pts.max(axis=0) - pts.min(axis=0)
    limit = default_max_extent
    if max_extent_by_class is not None and class_name in max_extent_by_class:
        limit = max_extent_by_class[class_name]
    return bool(np.all(span <= limit))


def _cluster(
    items: list[_Item],
    eligible: dict[tuple[int, int], bool],
    *,
    require_same_class: bool,
    max_extent_by_class: dict[str, float] | None,
    default_max_extent: float,
) -> list[list[int]]:
    """Constrained complete-linkage agglomerative clustering to a fixed point.

    Start with every observation its own cluster. Repeatedly pick the *best* eligible
    pair of clusters and merge them, where a cluster pair is mergeable only when

    * **complete linkage** — the merge predicate holds for the *worst* (here: every)
      cross-cluster member pair, not just one bridging link. This is the key difference
      from single-link union-find: a chain A~B~C~D never collapses because A and the
      far chair D are not directly compatible, so {A,B} and {C,D} are not complete-link
      mergeable;
    * **anti-chaining extent veto** — the merged cluster's AABB extent stays within the
      per-class cap on every axis.

    Iterates until no eligible cluster pair remains.
    """
    clusters: list[list[int]] = [[i] for i in range(len(items))]

    def link_ok(ca: list[int], cb: list[int]) -> bool:
        # Complete linkage: require an eligible base-pair between *every* member of ca
        # and *every* member of cb. A single missing/ineligible cross pair vetoes.
        for i in ca:
            for j in cb:
                key = (i, j) if i < j else (j, i)
                if not eligible.get(key, False):
                    return False
        return True

    while True:
        best: Optional[tuple[int, int]] = None
        best_score = -1
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                ca, cb = clusters[x], clusters[y]
                if require_same_class and items[ca[0]].obs.class_name != items[cb[0]].obs.class_name:
                    continue
                if not link_ok(ca, cb):
                    continue
                if not _merged_extent_ok(
                    ca,
                    cb,
                    items,
                    class_name=items[ca[0]].obs.class_name,
                    max_extent_by_class=max_extent_by_class,
                    default_max_extent=default_max_extent,
                ):
                    continue
                # Prefer the merge with the most cross-pair support (largest combined
                # size); deterministic tie-break by encounter order.
                score = len(ca) * len(cb)
                if score > best_score:
                    best_score, best = score, (x, y)
        if best is None:
            break
        x, y = best
        clusters[x] = clusters[x] + clusters[y]
        del clusters[y]

    return clusters


def _build_object(items: list[_Item], member_idx: list[int], *, voxel_m, dbscan_eps,
                  dbscan_min_points, sor_k) -> ObjectObservation:
    """Fuse + denoise one cluster's member observations into an ObjectObservation."""
    members = [items[k].obs for k in member_idx]

    # Concatenate every member's FULL cloud, then denoise.
    cloud = np.concatenate(
        [np.asarray(o.points, dtype=np.float64).reshape(-1, 3) for o in members], axis=0
    )
    cloud = voxel_downsample(cloud, voxel_m)
    cloud = dbscan_largest_cluster(cloud, eps=dbscan_eps, min_points=dbscan_min_points)
    cloud = statistical_outlier_removal(cloud, k=sor_k)
    cloud = np.asarray(cloud, dtype=np.float64).reshape(-1, 3)

    centroid, aabb_min, aabb_max, extent = aabb_of(cloud)

    # Mean CLIP embedding over non-empty members, then L2-normalise.
    embs = [e for e in (_emb_of(o.clip_emb) for o in members) if e is not None]
    if embs:
        dim = max(e.size for e in embs)
        stacked = np.stack([e for e in embs if e.size == dim], axis=0)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        clip_emb = (mean / norm).tolist() if norm > 0 else mean.tolist()
    else:
        clip_emb = []

    # De-duped union of member captions, preserving first-seen order.
    captions: list[str] = []
    seen: set[str] = set()
    for o in members:
        c = (o.caption or "").strip()
        if c and c not in seen:
            seen.add(c)
            captions.append(c)

    # Majority-vote class (ties broken by first encounter).
    counts: dict[str, int] = {}
    for o in members:
        counts[o.class_name] = counts.get(o.class_name, 0) + 1
    class_name = max(counts, key=lambda k: counts[k])
    class_id = next((o.class_id for o in members if o.class_name == class_name), None)

    ts_vals = [float(o.ts) for o in members]
    return ObjectObservation(
        class_name=class_name,
        class_id=class_id,
        conf=max(float(o.conf) for o in members),
        captions=captions,
        clip_emb=clip_emb,
        ts_first=min(ts_vals),
        ts_last=max(ts_vals),
        n_obs=len(members),
        points=cloud,
        centroid=centroid,
        extent=extent,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def associate(
    observations: list[Observation],
    *,
    overlap_min: float = 0.2,
    clip_min: float = 0.85,
    max_dist_m: float = 0.5,
    require_same_class: bool = True,
    voxel_m: float = 0.025,
    dbscan_eps: float = 0.05,
    dbscan_min_points: int = 10,
    sor_k: int = 16,
    max_extent_by_class: dict[str, float] | None = None,
    default_max_extent: float = 2.5,
) -> list[ObjectObservation]:
    """Batch-associate a build window of detections into fused object clusters.

    Args:
        observations: Pose-corrected world-frame detections from one build window.
        overlap_min: Minimum *mutual-min* nearest-neighbour overlap ratio for a pair to
            be a merge candidate (the flat-object→table guard).
        clip_min: Minimum CLIP cosine for a same-class pair (only used when
            ``require_same_class``).
        max_dist_m: Hard cap on the centroid distance of a candidate pair (the
            twin-fusion guard) and the spatial-prefilter radius.
        require_same_class: Require equal ``class_name`` and the CLIP gate for a merge.
        voxel_m: Voxel size for overlap NN radius and the final cloud downsample.
        dbscan_eps: DBSCAN epsilon for the per-cluster largest-cluster denoise.
        dbscan_min_points: DBSCAN ``min_points`` for that denoise.
        sor_k: Statistical-outlier-removal neighbour count for the final cloud.
        max_extent_by_class: Per-class AABB-extent caps for the anti-chaining veto.
        default_max_extent: Fallback extent cap for classes not in the mapping.

    Returns:
        One :class:`ObjectObservation` per final cluster (every input observation that
        survives the point-count filter lands in exactly one).
    """
    # 1. Drop empties / too-few points; prepare per-observation geometry.
    items: list[_Item] = []
    min_pts = max(1, dbscan_min_points // 3)  # tolerate sparse single views
    for obs in observations:
        pts_full = np.asarray(obs.points, dtype=np.float64).reshape(-1, 3)
        if len(pts_full) < min_pts:
            continue
        tree = cKDTree(pts_full) if cKDTree is not None and len(pts_full) else None
        items.append(
            _Item(
                obs=obs,
                centroid=pts_full.mean(axis=0),
                pts=pts_full,
                tree=tree,
                emb=_emb_of(obs.clip_emb),
            )
        )

    if not items:
        return []

    # 2. Spatial prefilter — candidate pairs within max_dist_m via a centroid KD-tree.
    centroids = np.stack([it.centroid for it in items], axis=0)
    if cKDTree is not None and len(items) > 1:
        cand_pairs = sorted(cKDTree(centroids).query_pairs(r=float(max_dist_m)))
    else:  # pragma: no cover — scipy absent
        cand_pairs = [
            (i, j)
            for i in range(len(items))
            for j in range(i + 1, len(items))
            if np.linalg.norm(centroids[i] - centroids[j]) <= max_dist_m
        ]

    # 3. Evaluate the merge predicate per candidate pair → eligible base-pair set.
    eligible: dict[tuple[int, int], bool] = {}
    for i, j in cand_pairs:
        eligible[(i, j)] = _pair_ok(
            items[i],
            items[j],
            overlap_min=overlap_min,
            clip_min=clip_min,
            max_dist_m=max_dist_m,
            require_same_class=require_same_class,
            voxel_m=voxel_m,
        )

    # 4. Constrained complete-linkage agglomerative clustering to a fixed point.
    clusters = _cluster(
        items,
        eligible,
        require_same_class=require_same_class,
        max_extent_by_class=max_extent_by_class,
        default_max_extent=default_max_extent,
    )

    # 5. Fuse + denoise each cluster into an ObjectObservation.
    return [
        _build_object(
            items,
            member_idx,
            voxel_m=voxel_m,
            dbscan_eps=dbscan_eps,
            dbscan_min_points=dbscan_min_points,
            sor_k=sor_k,
        )
        for member_idx in clusters
    ]
