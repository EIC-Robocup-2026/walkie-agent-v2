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

from . import prompts

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


def scan_seats(ctx: TaskContext) -> tuple[list[SeatCandidate], list[PersonPose], int]:
    """One frame: detect seats (open-vocab) + people, mark occupancy.

    Returns (seats, persons, image_width_px); ([], [], 0) on capture/detection
    failure. A whole sofa counts as one seat for now. The persons come from the
    same frame as the seats, so their pixel coordinates are comparable.
    """
    img = ctx.capture()
    if img is None:
        return [], [], 0
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
        return [], [], 0
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
    return seats, persons, img.width


def pick_free_seat(seats: list[SeatCandidate]) -> SeatCandidate | None:
    """Best free seat: highest confidence, then largest bbox."""
    free = [s for s in seats if not s.occupied]
    if not free:
        return None

    def area(s: SeatCandidate) -> float:
        x1, y1, x2, y2 = s.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    return max(free, key=lambda s: (s.confidence, area(s)))


def seat_world_position(ctx: TaskContext, seat: SeatCandidate) -> tuple[float, float] | None:
    """Lift a seat's bbox to a map-frame (x, y) via the perception service.

    Call right after the scan, before the robot moves — the service deprojects
    against the camera's *current* depth frame, not the scanned image.
    """
    x1, y1, x2, y2 = seat.bbox_xyxy
    cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    try:
        positions = ctx.walkie.tools.bboxes_to_positions([cxcywh])
    except Exception as exc:
        print(f"[skills] seat 3D lift failed ({exc})")
        return None
    if not positions:
        return None
    x, y, _z = positions[0]
    return x, y


def heading_to_point(ctx: TaskContext, x: float, y: float) -> float | None:
    """Map-frame heading from the robot's odometry position toward (x, y).

    None when odometry has no fix yet — ctx.current_pose()'s zeros fallback
    would make the atan2 aim somewhere arbitrary, so it is not used here.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] heading_to_point: odometry unavailable ({exc})")
        return None
    if not pose:
        return None
    return math.atan2(y - pose["y"], x - pose["x"])


def face_point(ctx: TaskContext, x: float, y: float) -> bool:
    """Rotate the base to face a map-frame point. Best-effort one-shot."""
    heading = heading_to_point(ctx, x, y)
    return ctx.rotate_to(heading) if heading is not None else False


def describe_seating_scene(
    seats: list[SeatCandidate],
    persons: list[PersonPose],
    img_w: int,
    guest: int,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
) -> str:
    """Text rendering of one seat scan for the LLM seat picker.

    Everything the model needs to decide: each seat's position in the frame
    (pixels + spoken direction), size, confidence and occupancy, each person's
    position, per-seat person overlap, and where an earlier guest was seated.
    """
    lines = [
        f"The camera frame is {img_w}px wide; x=0 is the robot's far left, "
        f"x={img_w} its far right.",
        f"Guest {guest} has just arrived, is standing next to the robot, and "
        f"needs a seat.",
        "",
        f"Detected seats ({len(seats)}):",
    ]
    person_boxes = [cxcywh_to_xyxy(p.bbox) for p in persons]
    for i, seat in enumerate(seats):
        x1, y1, x2, y2 = seat.bbox_xyxy
        overlap = max(
            (overlap_fraction(pb, seat.bbox_xyxy) for pb in person_boxes),
            default=0.0,
        )
        status = "OCCUPIED" if seat.occupied else "free"
        if overlap > 0:
            status += f" (a person's box covers {overlap:.0%} of it)"
        lines.append(
            f"  [{i}] {seat.class_name} — center x={seat.center_px[0]:.0f}px "
            f"({direction_phrase(seat.center_px[0], img_w)}), "
            f"{x2 - x1:.0f}x{y2 - y1:.0f}px, "
            f"detection confidence {seat.confidence:.2f}, {status}"
        )
    lines.append("")
    if persons:
        lines.append(f"Detected people ({len(persons)}):")
        for p in persons:
            cx, _cy, w, h = p.bbox
            lines.append(
                f"  - person at x={cx:.0f}px "
                f"({direction_phrase(cx, img_w)}), {w:.0f}x{h:.0f}px"
            )
    else:
        lines.append("No people detected in the frame.")
    for n, (seat, _w, _xy) in (prior_seats or {}).items():
        if n == guest:
            continue
        lines.append(
            f"Guest {n} was earlier offered the {seat.class_name} around "
            f"x={seat.center_px[0]:.0f}px (from an earlier scan, so it may "
            f"have shifted) and is probably sitting there now."
        )
    return "\n".join(lines)


def llm_pick_seat(
    ctx: TaskContext,
    seats: list[SeatCandidate],
    persons: list[PersonPose],
    img_w: int,
    guest: int,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
) -> SeatCandidate | None:
    """Let the LLM choose which seat to offer, with pick_free_seat as fallback.

    The model sees the whole frame (seats, people, the other guest's seat) so
    it can avoid seats the overlap heuristic missed and seat guests near each
    other. An explicit null from the model means "nothing suitable"; an
    extraction failure or out-of-range index degrades to the deterministic pick.
    """
    if not seats:
        return None
    scene = describe_seating_scene(seats, persons, img_w, guest, prior_seats)
    choice = ctx.extract(prompts.SeatChoice, prompts.PICK_SEAT_INSTRUCTIONS, scene)
    if choice is None:
        print("[skills] seat choice extraction failed; using heuristic pick")
        return pick_free_seat(seats)
    if choice.seat_index is None:
        print(f"[skills] LLM declined to pick a seat ({choice.reason or 'no reason given'})")
        return None
    if not 0 <= choice.seat_index < len(seats):
        print(f"[skills] LLM seat index {choice.seat_index} out of range; using heuristic pick")
        return pick_free_seat(seats)
    seat = seats[choice.seat_index]
    print(f"[skills] LLM picked seat [{choice.seat_index}] {seat.class_name}"
          f" ({choice.reason or 'no reason given'})")
    return seat


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
