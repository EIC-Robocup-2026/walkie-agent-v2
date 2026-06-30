"""Pure-numpy 2D polygon helpers for the arena map.

The world editor emits a ``polygon = [[x, y], ...]`` for rooms (boundary/walls),
locations (furniture footprint) and doors (doorway region), and an XY bounding box
+ Z height for known objects. These helpers answer "which room am I in?"
(:func:`point_in_polygon`) and convert a bbox to a footprint for object shapes —
no Shapely, no heavy deps, so the map layer stays import-light and offline-testable.

Polygons follow the world.toml convention: an ordered list of ``[x, y]`` vertices,
CCW, implicitly closed (the last vertex joins the first — do NOT repeat the first).
"""

from __future__ import annotations

from typing import Sequence

Point = tuple[float, float]
Polygon = Sequence[Sequence[float]]


def point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """True if point ``(x, y)`` lies inside ``polygon`` (ray-casting, O(n)).

    Uses the even-odd rule: cast a ray to +x and count edge crossings. Points
    exactly on an edge/vertex are treated as inside (best-effort; the boundary is
    a measure-zero case the caller rarely hits). Returns False for a degenerate
    polygon (< 3 vertices).
    """
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        # Does the horizontal ray at height y cross edge (j -> i)?
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_cross:
                inside = not inside
        j = i
    return inside


def polygon_centroid(polygon: Polygon) -> Point | None:
    """Mean of a polygon's vertices, or None if empty.

    A cheap label/anchor position (vertex average, not the area centroid — good
    enough for placing a room name or an object marker).
    """
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if not pts:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    return (sx / len(pts), sy / len(pts))


def bbox_to_polygon(x_min: float, y_min: float, x_max: float, y_max: float) -> list[list[float]]:
    """An axis-aligned XY bounding box as a CCW 4-vertex footprint polygon."""
    return [
        [float(x_min), float(y_min)],
        [float(x_max), float(y_min)],
        [float(x_max), float(y_max)],
        [float(x_min), float(y_max)],
    ]


def polygon_bounds(polygon: Polygon) -> tuple[float, float, float, float] | None:
    """Axis-aligned (x_min, y_min, x_max, y_max) of a polygon, or None if empty."""
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))
