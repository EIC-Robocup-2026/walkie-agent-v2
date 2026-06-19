"""Lift detector bboxes to map-frame points + a people-position blackboard.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

from tasks.base import TaskContext

from .geometry import BBox


def _cxcywh_to_world_position(ctx: TaskContext, bbox: BBox) -> tuple[float, float] | None:
    """Lift a bbox to a map-frame (x, y) via the perception service."""
    cx, cy, w, h = bbox
    cxcywh = (cx, cy, w, h)
    try:
        positions = ctx.walkie.tools.bboxes_to_positions([cxcywh])
    except Exception as exc:
        print(f"[skills] get world position failed ({exc})")
        return None
    if not positions:
        return None
    x, y, _z = positions[0]
    return x, y


def bboxes_world_position(ctx: TaskContext, bboxes: BBox) -> tuple[float, float] | None:
    """FALLBACK lift: bbox to map-frame (x, y) via the ROS perception service.

    The service deprojects against the camera's *current* depth frame, not the
    scanned image — only accurate when called immediately after the scan with
    the robot still. Prefer lift_bbox_world_xy, which lifts against the
    snapshot's frozen capture-time geometry.
    """
    x1, y1, x2, y2 = bboxes
    cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return _cxcywh_to_world_position(ctx, cxcywh)


def lift_bbox_world_xy(
    ctx: TaskContext, snap, bbox_xyxy: BBox, *, use_edge_filter: bool = True
) -> tuple[float, float] | None:
    """Lift a bbox to a map-frame (x, y): snapshot geometry first, service fallback.

    The snapshot lift (CameraSnapshot.bbox_world_xy) deprojects the bbox's
    pixels against the depth + camera pose frozen at capture time — exact no
    matter how much detection/LLM latency has elapsed since the scan. Only when
    the snapshot is missing or carries no geometry does this fall back to the
    legacy live-depth service lift.

    *use_edge_filter* False skips the per-frame full-resolution depth-discontinuity
    mask — the lift's dominant cost. The deprojection itself is already cropped to
    the bbox and capped, so dropping the edge filter makes the lift cheap enough to
    run every tick; the median over the (shrunk) central box stays on the person
    even without the flying-pixel cleanup. Use it on the fast follow path.
    """
    if snap is not None and getattr(snap, "has_geometry", False):
        try:
            xy = snap.bbox_world_xy(bbox_xyxy, use_edge_filter=use_edge_filter)
            if xy is not None:
                return xy
            print("[skills] snapshot lift returned no points; trying service fallback")
        except Exception as exc:
            print(f"[skills] snapshot lift failed ({exc}); trying service fallback")
    return bboxes_world_position(ctx, bbox_xyxy)


def remember_person_xy(
    ctx: TaskContext, person_id: str, xy: tuple[float, float]
) -> None:
    """Persist a person's latest map-frame (x, y) on the blackboard (latest wins)."""
    ctx.data.setdefault("people_xy", {})[person_id] = xy


def recall_person_xy(ctx: TaskContext, person_id: str) -> tuple[float, float] | None:
    """The last persisted map-frame (x, y) for a person, or None if unknown."""
    return ctx.data.get("people_xy", {}).get(person_id)


def reset_people_positions(ctx: TaskContext) -> None:
    """Forget every persisted people position (call at the start of a fresh run)."""
    ctx.data.pop("people_xy", None)


def remember_located_positions(
    ctx: TaskContext, located: dict[str, tuple[int, BBox]], snaps: list
) -> None:
    """Lift each recognized person to a map point and persist it (latest wins).

    *located* is :func:`identity.locate_people`'s ``{id: (frame_index, bbox)}``;
    *snaps* is the aligned snapshot list, so each box lifts against the geometry
    of the very frame it was found in.
    """
    for pid, (fi, box) in located.items():
        if 0 <= fi < len(snaps):
            xy = lift_bbox_world_xy(ctx, snaps[fi], box)
            if xy is not None:
                remember_person_xy(ctx, pid, xy)
