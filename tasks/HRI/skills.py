"""Reusable perception/geometry skills for the HRI task.

Plain functions over a TaskContext — no state. Anything generic enough for
other tasks should graduate to tasks/base.py; these are HRI-flavored
(seats, guests, look-at) but written so GPSR-style tasks can lift them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from client.pose_estimation import PersonPose
from tasks.base import TaskContext

BBox = tuple[float, float, float, float]


def parse_pose(s: str) -> tuple[float, float, float]:
    """Parse a waypoint string "x,y,heading_deg" -> (x, y, heading_rad)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 'x,y,heading_deg', got {s!r}")
    x, y, heading_deg = (float(p) for p in parts)
    return x, y, math.radians(heading_deg)


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


@dataclass
class SeatCandidate:
    bbox_xyxy: BBox
    class_name: str
    confidence: float
    center_px: tuple[float, float]
    occupied: bool


def find_persons(ctx: TaskContext) -> list[PersonPose]:
    """Capture a frame and return detected people (empty list on any failure)."""
    img = ctx.capture()
    if img is None:
        return []
    try:
        return ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[skills] pose estimation failed ({exc})")
        return []


def scan_seats(ctx: TaskContext) -> tuple[list[SeatCandidate], int]:
    """One frame: detect seats (open-vocab) + people, mark occupancy.

    Returns (seats, image_width_px); ([], 0) on capture/detection failure.
    A whole sofa counts as one seat for now.
    """
    img = ctx.capture()
    if img is None:
        return [], 0
    seat_classes = [
        c.strip()
        for c in os.getenv("HRI_SEAT_CLASSES", "chair,sofa,armchair,stool").split(",")
        if c.strip()
    ]
    occupied_overlap = float(os.getenv("HRI_SEAT_OCCUPIED_OVERLAP", "0.25"))
    try:
        detections = ctx.walkieAI.object_detection.detect(img, prompts=seat_classes)
    except Exception as exc:
        print(f"[skills] seat detection failed ({exc})")
        return [], 0
    try:
        persons = ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[skills] pose estimation failed ({exc}); assuming all seats free")
        persons = []
    person_boxes = [cxcywh_to_xyxy(p.bbox) for p in persons]

    seats: list[SeatCandidate] = []
    for det in detections:
        if det.class_name and det.class_name.lower() not in [c.lower() for c in seat_classes]:
            continue
        x1, y1, x2, y2 = det.bbox
        occupied = any(
            overlap_fraction(pb, (x1, y1, x2, y2)) >= occupied_overlap
            for pb in person_boxes
        )
        seats.append(
            SeatCandidate(
                bbox_xyxy=(x1, y1, x2, y2),
                class_name=det.class_name or "seat",
                confidence=det.confidence or 0.0,
                center_px=((x1 + x2) / 2, (y1 + y2) / 2),
                occupied=occupied,
            )
        )
    return seats, img.width


def pick_free_seat(seats: list[SeatCandidate]) -> SeatCandidate | None:
    """Best free seat: highest confidence, then largest bbox."""
    free = [s for s in seats if not s.occupied]
    if not free:
        return None

    def area(s: SeatCandidate) -> float:
        x1, y1, x2, y2 = s.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    return max(free, key=lambda s: (s.confidence, area(s)))


def direction_phrase(center_x: float, img_w: int) -> str:
    """Frame thirds -> a spoken direction (camera faces forward)."""
    if img_w <= 0:
        return "in front of me"
    third = center_x / img_w
    if third < 1 / 3:
        return "to my left"
    if third > 2 / 3:
        return "to my right"
    return "in front of me"


def heading_to_pixel(ctx: TaskContext, px_x: float, img_w: int) -> float:
    """Map-frame heading that would center the given pixel column."""
    hfov = math.radians(float(os.getenv("HRI_CAMERA_HFOV_DEG", "90")))
    offset = (0.5 - px_x / img_w) * hfov  # left of center -> positive (CCW)
    return ctx.current_pose()["heading"] + offset


def face_pixel(ctx: TaskContext, px_x: float, img_w: int) -> bool:
    """Rotate the base to look toward a pixel column. Best-effort one-shot."""
    if img_w <= 0:
        return False
    return ctx.rotate_to(heading_to_pixel(ctx, px_x, img_w))
