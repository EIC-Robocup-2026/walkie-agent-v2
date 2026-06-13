"""Reusable perception/geometry skills for the HRI task.

Plain functions over a TaskContext — no state. Anything generic enough for
other tasks should graduate to tasks/base.py; these are HRI-flavored
(seats, guests, look-at) but written so GPSR-style tasks can lift them.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
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


def scan_seats(ctx: TaskContext):
    """One frame: detect seats (open-vocab) + people, mark occupancy.

    Returns (seats, persons, snap) where ``snap`` is the CameraSnapshot the
    frame came from — its frozen depth/pose geometry lifts seat/person bboxes
    to map-frame points later (lift_bbox_world_xy), immune to the LLM/detection
    latency that follows. ([], [], None) on capture failure and ([], [], snap)
    on detection failure. A whole sofa counts as one seat for now. Seats,
    persons, and snap.img all come from the same capture, so pixel coordinates
    are comparable and crops line up.
    """
    snap = ctx.snapshot()
    if snap is None:
        return [], [], None
    img = snap.img
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
        return [], [], snap
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
    return seats, persons, snap


def sweep_snapshots(
    ctx: TaskContext, offsets_deg: Sequence[float] = (10.0, 0.0, -10.0)
) -> list:
    """Capture a snapshot at each base-heading offset, then face forward again.

    The head servo only tilts (no pan), so looking left/right is a small in-place
    base rotation around the heading at entry. *offsets_deg* are degrees relative
    to that heading (positive = left/CCW, e.g. the default sweeps left, center,
    right). Returns the successful CameraSnapshots in the given order; each
    carries its own capture-time depth/pose geometry, so a bbox found in any of
    them lifts to the correct map-frame point regardless of where the robot was
    pointing when it was taken. With no odometry fix the base heading is unknown,
    so a rotation would aim arbitrarily — it falls back to a single forward
    snapshot. Best-effort: a failed capture is dropped, never raised.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] sweep_snapshots: odometry unavailable ({exc}); single frame")
        pose = None
    if not pose:
        snap = ctx.snapshot()
        return [snap] if snap is not None else []
    center = pose["heading"]
    settle = float(os.getenv("HRI_SWEEP_SETTLE_SEC", "1.0"))
    snaps = []
    for off in offsets_deg:
        ctx.rotate_to(center + math.radians(off))
        if settle > 0:
            time.sleep(settle)  # let the base + depth settle before capturing
        snap = ctx.snapshot()
        if snap is not None:
            snaps.append(snap)
    ctx.rotate_to(center)  # leave the robot facing forward again
    return snaps


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
    """FALLBACK lift: bbox to map-frame (x, y) via the ROS perception service.

    The service deprojects against the camera's *current* depth frame, not the
    scanned image — only accurate when called immediately after the scan with
    the robot still. Prefer lift_bbox_world_xy, which lifts against the
    snapshot's frozen capture-time geometry.
    """
    x1, y1, x2, y2 = bboxes
    cxcywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return _cxcywh_to_world_position(ctx, cxcywh)


def lift_bbox_world_xy(ctx: TaskContext, snap, bbox_xyxy: BBox) -> tuple[float, float] | None:
    """Lift a bbox to a map-frame (x, y): snapshot geometry first, service fallback.

    The snapshot lift (CameraSnapshot.bbox_world_xy) deprojects the bbox's
    pixels against the depth + camera pose frozen at capture time — exact no
    matter how much detection/LLM latency has elapsed since the scan. Only when
    the snapshot is missing or carries no geometry does this fall back to the
    legacy live-depth service lift.
    """
    if snap is not None and getattr(snap, "has_geometry", False):
        try:
            xy = snap.bbox_world_xy(bbox_xyxy)
            if xy is not None:
                return xy
            print("[skills] snapshot lift returned no points; trying service fallback")
        except Exception as exc:
            print(f"[skills] snapshot lift failed ({exc}); trying service fallback")
    return bboxes_world_position(ctx, bbox_xyxy)


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


def match_people_to_seats(
    located: dict[str, tuple[int, BBox]],
    seats: list[SeatCandidate],
    *,
    frame_index: int = 0,
    min_overlap: float | None = None,
) -> tuple[dict[int, str], dict[str, BBox]]:
    """Tie each recognized person to the detected seat they occupy.

    *located* is :func:`identity.locate_people`'s output
    (``{id: (frame_index, person_bbox)}``); only entries from *frame_index* are
    used, since the seats come from one frame and a person seen in another sweep
    view can't be lined up against them. A person claims the seat their box
    overlaps most (at least *min_overlap*, default ``HRI_SEAT_OCCUPIED_OVERLAP``
    — the same floor :func:`scan_seats` uses for occupancy), strongest overlap
    first so a confident match isn't blocked by a weak one; one person per seat.

    Returns ``(seat_occupants, seatless)``: *seat_occupants* maps a seat index
    to the id sitting in it; *seatless* maps the id of every person recognized
    in the frame whose box overlapped NO detected seat to that box — the
    detector missed their chair, or they're on something off-vocabulary (a couch
    arm, a stool, the floor). The caller still knows those people are present
    and roughly where, so the spot isn't wrongly offered as free.
    """
    if min_overlap is None:
        min_overlap = float(os.getenv("HRI_SEAT_OCCUPIED_OVERLAP", "0.25"))
    present: dict[str, BBox] = {
        pid: box for pid, (fi, box) in located.items() if fi == frame_index
    }
    pairs = [
        (overlap_fraction(box, seat.bbox_xyxy), pid, i)
        for pid, box in present.items()
        for i, seat in enumerate(seats)
    ]
    seat_occupants: dict[int, str] = {}
    claimed_seats: set[int] = set()
    claimed_pids: set[str] = set()
    for ov, pid, i in sorted(pairs, key=lambda p: p[0], reverse=True):
        if ov < min_overlap or pid in claimed_pids or i in claimed_seats:
            continue
        seat_occupants[i] = pid
        claimed_seats.add(i)
        claimed_pids.add(pid)
    seatless = {pid: box for pid, box in present.items() if pid not in claimed_pids}
    return seat_occupants, seatless


def _person_label(pid: str, names: dict[str, str | None] | None = None) -> str:
    """Human-readable label for an enrolled-person id, with name when known."""
    name = (names or {}).get(pid)
    if pid == "host":
        return f"the host ({name})" if name else "the host"
    if pid.startswith("guest-"):
        n = pid.removeprefix("guest-")
        return f"Guest {n} ({name})" if name else f"Guest {n}"
    return name or pid


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
    seat_occupants: dict[int, str] | None = None,
    seatless_people: dict[str, BBox] | None = None,
    person_names: dict[str, str | None] | None = None,
) -> str:
    """Text rendering of one seat scan for the LLM seat picker.

    Everything the model needs to decide and to word the offer: each seat's
    position in the frame (pixel x, where x=0 is far left), size, confidence and
    occupancy, each person's position, per-seat person overlap, the host
    (always present and seated, with drink/appearance when known), and where
    an earlier guest was seated.

    When *seat_occupants* (seat index -> recognized person id, from
    :func:`match_people_to_seats`) is given, each named seat says WHO is sitting
    in it; *seatless_people* (id -> box) flags people recognized in the frame
    that no detected seat lined up with, so the model neither offers their spot
    nor double-seats them. *person_names* supplies display names for the ids.
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
    occupants = seat_occupants or {}
    for i, seat in enumerate(seats):
        x1, y1, x2, y2 = seat.bbox_xyxy
        overlap = max(
            (overlap_fraction(pb, seat.bbox_xyxy) for pb in person_boxes),
            default=0.0,
        )
        if i in occupants:
            status = f"OCCUPIED — {_person_label(occupants[i], person_names)} is sitting here"
        else:
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
    for pid, box in (seatless_people or {}).items():
        cx = (box[0] + box[2]) / 2
        lines.append(
            f"{_person_label(pid, person_names)} is seated around x={cx:.0f}px, "
            f"but no detected seat lines up with them — their seat is likely one "
            f"the detector missed (a couch, stool, or surface). They are there: "
            f"don't offer that spot, and don't seat the new guest on top of them."
        )
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
    seat_occupants: dict[int, str] | None = None,
    seatless_people: dict[str, BBox] | None = None,
    person_names: dict[str, str | None] | None = None,
) -> tuple[SeatCandidate | None, str | None]:
    """Let the LLM choose which seat to offer and word the spoken offer.

    Returns (seat, announcement). The model sees the whole frame (seats,
    people, who is recognized in which seat, the host, the other guest's seat)
    so it can avoid seats the overlap heuristic missed, seat guests near the
    host, and refer to the host in the announcement. A null announcement means
    "use the default line". An explicit null seat from the model means "nothing
    suitable"; an extraction failure or out-of-range index degrades to the
    deterministic pick_free_seat.
    """
    if not seats:
        return None, None
    scene = describe_seating_scene(
        seats, persons, img_w, guest,
        guest_name=guest_name, host_name=host_name,
        host_drink=host_drink, host_appearance=host_appearance,
        prior_seats=prior_seats,
        seat_occupants=seat_occupants,
        seatless_people=seatless_people,
        person_names=person_names,
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
