"""Geometric / distance-based relation derivation for the scene graph.

A pure function: given the current object nodes it emits the spatial edges
(``near`` / ``on`` / ``above`` / ``inside``) implied by their AABBs and centroids.
No I/O, no graph state — callers persist the returned list.
"""

from __future__ import annotations

from typing import Optional

from walkie_world.scene.store import ObjectNode, Relation, l2

__all__ = ["derive_relations"]


def _xy_overlap(a: ObjectNode, b: ObjectNode) -> float:
    """Footprint overlap ratio = intersection / smaller AABB area (XY plane).

    Ratio (not IoU) so a small object fully over a large surface scores ~1.0 — IoU
    would be tiny there and miss every "mug on table".
    """
    ax0, ay0 = a.aabb_min[0], a.aabb_min[1]
    ax1, ay1 = a.aabb_max[0], a.aabb_max[1]
    bx0, by0 = b.aabb_min[0], b.aabb_min[1]
    bx1, by1 = b.aabb_max[0], b.aabb_max[1]
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _volume(n: ObjectNode) -> float:
    return n.extent[0] * n.extent[1] * n.extent[2]


def _vertical(
    x: ObjectNode,
    y: ObjectNode,
    *,
    xy_overlap_min: float,
    z_tol: float,
    on_gap: float,
) -> Optional[str]:
    """'x on/above y': x sits over y with overlapping footprint."""
    if _xy_overlap(x, y) < xy_overlap_min:
        return None
    gap = x.aabb_min[2] - y.aabb_max[2]
    if gap < -z_tol:  # x's base is well below y's top → not on top
        return None
    return "on" if gap <= on_gap else "above"


def _inside(x: ObjectNode, y: ObjectNode, *, contain_tol: float) -> bool:
    """'x inside y': x's AABB is contained in y's (with slack) and smaller."""
    t = contain_tol
    contained = all(
        y.aabb_min[i] - t <= x.aabb_min[i] and x.aabb_max[i] <= y.aabb_max[i] + t
        for i in range(3)
    )
    return contained and _volume(x) < _volume(y)


def derive_relations(
    nodes,
    *,
    relation_max_dist: float = 1.0,
    near_m: float = 0.6,
    xy_overlap_min: float = 0.15,
    z_tol: float = 0.05,
    on_gap: float = 0.08,
    contain_tol: float = 0.02,
) -> list[Relation]:
    """Recompute all geometric edges from the given node geometry.

    Args:
        nodes: Iterable of :class:`ObjectNode` (anything with ``id``, ``centroid``,
            ``extent``, ``aabb_min``, ``aabb_max``).
        relation_max_dist: Max centroid distance (m) for any relation to be considered.
        near_m: Centroid distance (m) below which a ``near`` edge is emitted; its
            weight is ``1 - d/near_m`` (1.0 when ``near_m <= 0``), rounded to 3 dp.
        xy_overlap_min: Min footprint overlap ratio for an ``on``/``above`` edge.
        z_tol: Vertical slack (m); x's base may dip this far below y's top.
        on_gap: Vertical gap (m) at or below which the edge is ``on`` rather than ``above``.
        contain_tol: AABB containment slack (m) for an ``inside`` edge.

    Returns:
        Flat list of :class:`Relation` (unordered pairs expand to both directions
        for the vertical / containment checks).
    """
    rels: list[Relation] = []
    nodes = list(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            d = l2(a.centroid, b.centroid)
            if d > relation_max_dist:
                continue
            if d <= near_m:
                w = 1.0 - d / near_m if near_m > 0 else 1.0
                rels.append(Relation(a.id, b.id, "near", round(w, 3)))
            for x, y in ((a, b), (b, a)):
                pred = _vertical(
                    x,
                    y,
                    xy_overlap_min=xy_overlap_min,
                    z_tol=z_tol,
                    on_gap=on_gap,
                )
                if pred:
                    rels.append(Relation(x.id, y.id, pred, 1.0))
                if _inside(x, y, contain_tol=contain_tol):
                    rels.append(Relation(x.id, y.id, "inside", 1.0))
    return rels
