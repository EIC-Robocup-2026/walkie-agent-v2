"""walkie_graphs DB bridge for the manipulation pipeline.

Thin, defensive accessors over the :class:`~services.walkie_graphs.WalkieGraphs`
facade (``ctx.graphs``). They resolve object/surface nodes by description,
hand back an object's stored world cloud as a ``PointCloud2`` dict for GraspNet,
and turn a surface node's axis-aligned bbox into the ``set_table`` collision box.

Everything degrades to ``None`` rather than raising — the GraspNet path falls
back to the stub planner when the DB is unavailable or a lookup misses.
"""

from __future__ import annotations

from .cloud import numpy_to_pointcloud2
from .types import Vec3


def resolve_object_node(graphs, query: str):
    """Best stored object node matching *query* (caption/class), or None.

    *graphs* is a ``WalkieGraphs`` facade (``ctx.graphs``); None -> None.
    """
    if graphs is None or not query:
        return None
    try:
        hits = graphs.query_text(query, k=1)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.db] resolve_object_node({query!r}) failed ({exc})")
        return None
    return hits[0] if hits else None


def resolve_surface_node(graphs, query: str, *, near: tuple[float, float] | None = None):
    """Best stored surface node (table/shelf/...) matching *query*, or None.

    When *near* (map-frame x, y) is given, prefer matches close to it — useful
    when several tables share a class name.
    """
    if graphs is None or not query:
        return None
    try:
        hits = graphs.query_text(query, k=5, near=near)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.db] resolve_surface_node({query!r}) failed ({exc})")
        return None
    return hits[0] if hits else None


def object_cloud_pc2(graphs, node, *, frame_id: str = "map"):
    """The node's stored world cloud as a PointCloud2 dict for GraspNet, or None.

    Reads the ``.npz`` sidecar via ``GraphMemory.load_pcd`` (an ``(N, 3)``
    map-frame array) and packs it with :func:`numpy_to_pointcloud2`.
    """
    if graphs is None or node is None:
        return None
    try:
        pts = graphs.memory.load_pcd(node.id)
    except Exception as exc:  # noqa: BLE001
        print(f"[manipulation.db] load_pcd({getattr(node, 'id', '?')}) failed ({exc})")
        return None
    if pts is None or len(pts) == 0:
        return None
    return numpy_to_pointcloud2(pts, frame_id=frame_id)


def node_table_box(node) -> tuple[list[float], list[float]] | None:
    """A surface node's ``(pose, size)`` for ``arm.set_table``, or None.

    ``pose`` = ``[x, y, top_z, yaw]`` (yaw 0 — the stored bbox is axis-aligned),
    ``size`` = ``[depth_x, width_y]``. The box spans floor -> ``top_z``, so the
    arm avoids the whole table volume, not just its lip.
    """
    if node is None:
        return None
    try:
        mnx, mny, _mnz = node.aabb_min
        mxx, mxy, mxz = node.aabb_max
    except Exception:  # noqa: BLE001
        return None
    cx, cy = (mnx + mxx) / 2.0, (mny + mxy) / 2.0
    depth_x, width_y = abs(mxx - mnx), abs(mxy - mny)
    top_z = float(mxz)
    return [float(cx), float(cy), top_z, 0.0], [float(depth_x), float(width_y)]


def node_centroid(node) -> Vec3 | None:
    """Map-frame ``(x, y, z)`` centroid of a node, or None."""
    if node is None:
        return None
    try:
        cx, cy, cz = node.centroid
        return float(cx), float(cy), float(cz)
    except Exception:  # noqa: BLE001
        return None
