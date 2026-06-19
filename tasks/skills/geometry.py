"""Pure geometry / bbox / pose-keypoint math — no robot I/O.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

from client import PersonPose


BBox = tuple[float, float, float, float]


def parse_pose(s: str) -> tuple[float, float, float]:
    """Parse a waypoint string "x,y,heading_rad" -> (x, y, heading_rad)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,heading_rad', got {s!r}")
    x, y, heading_rad = (float(p) for p in parts)
    return x, y, heading_rad


def cxcywh_to_xyxy(bbox: BBox) -> BBox:
    """Pose-estimation bboxes are (cx, cy, w, h); detections are xyxy."""
    cx, cy, w, h = bbox
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def overlap_fraction(person_xyxy: BBox, seat_xyxy: BBox) -> float:
    """Intersection area as a fraction of the seat's area."""
    ax1, ay1, ax2, ay2 = person_xyxy
    bx1, by1, bx2, by2 = seat_xyxy
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    seat_area = (bx2 - bx1) * (by2 - by1)
    if seat_area <= 0:
        return 0.0
    return (iw * ih) / seat_area


def _point_in_box(point: tuple[float, float], box: BBox) -> bool:
    """True when (x, y) lies inside the xyxy *box* (inclusive bounds)."""
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


# COCO keypoint indices for the seating anchor (the hips are the part of the
# body resting on the cushion). Same convention as Restaurant/skills.py.
_LEFT_HIP, _RIGHT_HIP = 11, 12


def person_hip_anchor(
    person: PersonPose, conf_thresh: float = 0.3
) -> tuple[float, float] | None:
    """The hip midpoint (COCO 11/12) when at least one hip is confidently seen.

    This is where the person actually rests on a seat — far more reliable for
    deciding *which* seat (or sofa cushion) they occupy than their bounding box,
    which sprawls sideways over neighbouring seats. Returns None when neither hip
    clears *conf_thresh* (pose keypoints missing or occluded), so the caller can
    fall back to a coarser bbox-overlap test only for those people.
    """
    kpts = {kp.index: kp for kp in person.keypoints}
    lh, rh = kpts.get(_LEFT_HIP), kpts.get(_RIGHT_HIP)
    lo = lh if (lh is not None and lh.confidence > conf_thresh) else None
    ro = rh if (rh is not None and rh.confidence > conf_thresh) else None
    if lo is not None and ro is not None:
        return ((lo.x + ro.x) / 2, (lo.y + ro.y) / 2)
    if lo is not None or ro is not None:
        h = lo or ro
        return (h.x, h.y)
    return None


def person_seat_anchor(
    person: PersonPose, conf_thresh: float = 0.3
) -> tuple[float, float]:
    """Best estimate of where *person* is seated, always a point.

    The reliable hip midpoint (:func:`person_hip_anchor`) when available, else
    the bbox's lower-centre — roughly where the seat is under a person whose
    hips weren't detected.
    """
    anchor = person_hip_anchor(person, conf_thresh)
    if anchor is not None:
        return anchor
    cx, cy, _w, h = person.bbox
    return (cx, cy + h * 0.25)
