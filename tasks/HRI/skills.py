"""Reusable perception/geometry skills for the HRI task.

Plain functions over a TaskContext — no state. Anything generic enough for
other tasks should graduate to tasks/base.py; these are HRI-flavored
(seats, guests, look-at) but written so GPSR-style tasks can lift them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from PIL import Image

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


def scan_seats(ctx: TaskContext) -> tuple[list[SeatCandidate], list[PersonPose], Image.Image | None]:
    """One frame: detect seats (open-vocab) + people, mark occupancy.

    Returns (seats, persons, frame); ([], [], None) on capture failure and
    ([], [], frame) on detection failure. A whole sofa counts as one seat for
    now. Seats, persons, and frame all come from the same capture, so pixel
    coordinates are comparable and crops line up.
    """
    img = ctx.capture()
    if img is None:
        return [], [], None
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
        return [], [], img
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
    return seats, persons, img


def pick_free_seat(seats: list[SeatCandidate]) -> SeatCandidate | None:
    """Best free seat: highest confidence, then largest bbox."""
    free = [s for s in seats if not s.occupied]
    if not free:
        return None

    def area(s: SeatCandidate) -> float:
        x1, y1, x2, y2 = s.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    return max(free, key=lambda s: (s.confidence, area(s)))


def find_seated_person_bbox(
    persons: list[PersonPose], seats: list[SeatCandidate]
) -> BBox | None:
    """The person overlapping an occupied seat (a lone person is accepted too)."""
    occupied = [s.bbox_xyxy for s in seats if s.occupied]
    for p in persons:
        pb = cxcywh_to_xyxy(p.bbox)
        if any(overlap_fraction(pb, sb) > 0 for sb in occupied):
            return pb
    if len(persons) == 1:
        return cxcywh_to_xyxy(persons[0].bbox)
    return None


def describe_seated_person(
    ctx: TaskContext,
    img: Image.Image,
    persons: list[PersonPose],
    seats: list[SeatCandidate],
) -> str | None:
    """Caption the appearance of the person sitting on a detected seat.

    Crops to the person overlapping an occupied seat when pose detection found
    one (a lone detected person is accepted too); otherwise captions the whole
    frame and lets the prompt single out the seated person. None on failure.
    """
    target = find_seated_person_bbox(persons, seats)
    crop = img
    if target is not None:
        x1, y1, x2, y2 = target
        m = 20  # px padding so clothing isn't clipped at the bbox edge
        crop = img.crop((
            max(0, int(x1 - m)), max(0, int(y1 - m)),
            min(img.width, int(x2 + m)), min(img.height, int(y2 + m)),
        ))
    try:
        return ctx.walkieAI.image_caption.caption(
            crop, prompt=prompts.HOST_APPEARANCE_CAPTION_PROMPT
        )
    except Exception as exc:
        print(f"[skills] seated-person appearance caption failed ({exc})")
        return None


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
    """Lift a seat's bbox to a map-frame (x, y) via the perception service.

    Call right after the scan, before the robot moves — the service deprojects
    against the camera's *current* depth frame, not the scanned image.
    """
    x1, y1, x2, y2 = bboxes
    cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return _cxcywh_to_world_position(ctx, cxcywh)


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
    guest_name: str | None = None,
    host_name: str | None = None,
    host_drink: str | None = None,
    host_appearance: str | None = None,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
) -> str:
    """Text rendering of one seat scan for the LLM seat picker.

    Everything the model needs to decide and to word the offer: each seat's
    position in the frame (pixel x, where x=0 is far left), size, confidence and
    occupancy, each person's position, per-seat person overlap, the host
    (always present and seated, with drink/appearance when known), and where
    an earlier guest was seated.
    """
    guest_label = f"Guest {guest}" + (f", named {guest_name}," if guest_name else "")
    host_line = (
        f"The party host{f', {host_name},' if host_name else ''} is already "
        f"in the room and seated — one of the seated people is the host."
    )
    if host_drink:
        host_line += f" The host's favorite drink is {host_drink}."
    if host_appearance:
        host_line += f" The host's appearance: {host_appearance}"
    lines = [
        f"The camera frame is {img_w}px wide; x=0 is the robot's far left, "
        f"x={img_w} its far right.",
        f"{guest_label} has just arrived, is standing next to the robot, and "
        f"needs a seat.",
        host_line,
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
            f"  [{i}] {seat.class_name} — center x={seat.center_px[0]:.0f}px, "
            f"{x2 - x1:.0f}x{y2 - y1:.0f}px, "
            f"detection confidence {seat.confidence:.2f}, {status}"
        )
    lines.append("")
    if persons:
        lines.append(f"Detected people ({len(persons)}):")
        for p in persons:
            cx, _cy, w, h = p.bbox
            lines.append(
                f"  - person at x={cx:.0f}px, {w:.0f}x{h:.0f}px"
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
    guest_name: str | None = None,
    host_name: str | None = None,
    host_drink: str | None = None,
    host_appearance: str | None = None,
    prior_seats: dict[int, tuple[SeatCandidate, int, tuple[float, float] | None]] | None = None,
) -> tuple[SeatCandidate | None, str | None]:
    """Let the LLM choose which seat to offer and word the spoken offer.

    Returns (seat, announcement). The model sees the whole frame (seats,
    people, the host, the other guest's seat) so it can avoid seats the
    overlap heuristic missed, seat guests near the host, and refer to the host
    in the announcement. A null announcement means "use the default line". An
    explicit null seat from the model means "nothing suitable"; an extraction
    failure or out-of-range index degrades to the deterministic pick_free_seat.
    """
    if not seats:
        return None, None
    scene = describe_seating_scene(
        seats, persons, img_w, guest,
        guest_name=guest_name, host_name=host_name,
        host_drink=host_drink, host_appearance=host_appearance,
        prior_seats=prior_seats,
    )
    choice = ctx.extract(prompts.SeatChoice, prompts.PICK_SEAT_INSTRUCTIONS, scene)
    if choice is None:
        print("[skills] seat choice extraction failed; using heuristic pick")
        return pick_free_seat(seats), None
    if choice.seat_index is None:
        print(f"[skills] LLM declined to pick a seat ({choice.reason or 'no reason given'})")
        return None, None
    if not 0 <= choice.seat_index < len(seats):
        print(f"[skills] LLM seat index {choice.seat_index} out of range; using heuristic pick")
        return pick_free_seat(seats), None
    seat = seats[choice.seat_index]
    print(f"[skills] LLM picked seat [{choice.seat_index}] {seat.class_name}"
          f" ({choice.reason or 'no reason given'})")
    return seat, (choice.announcement or "").strip() or None


def llm_intro_speeches(ctx: TaskContext, people: dict[str, dict]) -> dict[str, str]:
    """Word every introduction in ONE LLM call: {person_id: spoken sentence(s)}.

    *people* maps "host"/"guest-1"/"guest-2" to {"name", "drink", "appearance"}
    records (fields may be None). Always returns a speech for all three —
    on extraction failure each falls back to the per-person template.
    """
    generic = {
        "host": prompts.GENERIC_HOST,
        "guest-1": prompts.GENERIC_FIRST_GUEST,
        "guest-2": prompts.GENERIC_SECOND_GUEST,
    }

    def fallback(pid: str) -> str:
        p = people.get(pid, {})
        return prompts.PERSON_INTRO_TEMPLATE.format(
            name=p.get("name") or generic[pid],
            drink=p.get("drink") or prompts.GENERIC_DRINK,
        )

    lines = []
    for pid, label in (("host", "Host"), ("guest-1", "First guest"), ("guest-2", "Second guest")):
        p = people.get(pid, {})
        lines.append(
            f"{label}: name={p.get('name') or 'unknown'}; "
            f"favorite drink={p.get('drink') or 'unknown'}; "
            f"appearance={p.get('appearance') or 'unknown'}"
        )
    speeches = ctx.extract(
        prompts.IntroSpeeches, prompts.INTRO_SPEECHES_INSTRUCTIONS, "\n".join(lines)
    )
    if speeches is None:
        print("[skills] intro speech extraction failed; using template lines")
        return {pid: fallback(pid) for pid in generic}
    by_id = {
        "host": (speeches.host or "").strip(),
        "guest-1": (speeches.guest_1 or "").strip(),
        "guest-2": (speeches.guest_2 or "").strip(),
    }
    return {pid: text or fallback(pid) for pid, text in by_id.items()}
