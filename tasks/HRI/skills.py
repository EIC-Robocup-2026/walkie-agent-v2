"""Reusable perception/geometry skills for the HRI task.

Plain functions over a TaskContext — no state. Anything generic enough for
other tasks should graduate to tasks/base.py; these are HRI-flavored
(seats, guests, look-at) but written so GPSR-style tasks can lift them.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

from PIL import Image

from client.face_recognition import FaceEmbedding
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


@dataclass
class SeatPart:
    """One cushion of a multi-seat sofa, with its own occupancy."""

    label: str  # "LEFT" | "MIDDLE" | "RIGHT"
    bbox_xyxy: BBox
    center_px: tuple[float, float]
    occupied: bool


@dataclass
class SeatCandidate:
    bbox_xyxy: BBox
    class_name: str
    confidence: float
    center_px: tuple[float, float]
    occupied: bool
    # For a sofa/couch: its LEFT/MIDDLE/RIGHT cushions with per-cushion
    # occupancy (None for an ordinary single seat). ``occupied`` above is then
    # True only when every cushion is taken.
    parts: list[SeatPart] | None = None


# --- Sofa cushion parsing ----------------------------------------------------
# A sofa seats several people, so treating it as one occupied/free unit either
# wastes free cushions (one guest on a 3-seater hides two empty spots) or, with
# a low overlap floor, marks a free chair taken because a neighbour's box edges
# onto it. Both are fixed by deciding occupancy per cushion from where people
# actually sit (their hip anchor), not from whole-bbox overlap.


def _sofa_classes() -> list[str]:
    return [
        c.strip().lower()
        for c in os.getenv("HRI_SOFA_CLASSES", "sofa,couch,bench,loveseat").split(",")
        if c.strip()
    ]


def is_sofa_class(class_name: str | None) -> bool:
    """True for detector classes that seat several people (split into cushions)."""
    return (class_name or "").lower() in _sofa_classes()


def split_seat_regions(bbox: BBox, *, has_middle: bool = True) -> list[tuple[str, BBox]]:
    """Split a sofa bbox into evenly-spaced cushion regions, left→right.

    x grows rightward and x=0 is the robot's far left, so the first column is the
    robot's LEFT. *has_middle* False gives a two-cushion (LEFT, RIGHT) sofa — set
    it for sofas that only have two seats.
    """
    x1, y1, x2, y2 = bbox
    labels = ("LEFT", "MIDDLE", "RIGHT") if has_middle else ("LEFT", "RIGHT")
    n = len(labels)
    step = (x2 - x1) / n
    regions: list[tuple[str, BBox]] = []
    for i, label in enumerate(labels):
        rx1, rx2 = x1 + i * step, x1 + (i + 1) * step
        regions.append((label, (rx1, y1, rx2, y2)))
    return regions


def _occupied_region_indices(
    persons: list[PersonPose],
    region_boxes: list[BBox],
    *,
    hard_overlap: float,
    conf_thresh: float,
) -> set[int]:
    """Which of *region_boxes* a person is sitting on (one region per person).

    Each person is placed on the single region containing their hip anchor; a
    person whose hips weren't detected falls back to the region their box covers
    most, but only when that coverage reaches *hard_overlap* — high enough that a
    neighbour's box merely clipping a cushion's edge doesn't count as sitting on
    it.
    """
    occupied: set[int] = set()
    for p in persons:
        anchor = person_hip_anchor(p, conf_thresh)
        if anchor is not None:
            for i, rb in enumerate(region_boxes):
                if _point_in_box(anchor, rb):
                    occupied.add(i)
                    break  # a person sits on one cushion
            continue
        pb = cxcywh_to_xyxy(p.bbox)
        best_i, best_ov = None, hard_overlap
        for i, rb in enumerate(region_boxes):
            ov = overlap_fraction(pb, rb)
            if ov >= best_ov:
                best_i, best_ov = i, ov
        if best_i is not None:
            occupied.add(best_i)
    return occupied


def parse_sofa_parts(
    sofa_bbox: BBox,
    persons: list[PersonPose],
    *,
    has_middle: bool = True,
    hard_overlap: float = 0.5,
    conf_thresh: float = 0.3,
) -> list[SeatPart]:
    """Split a sofa into cushions and mark each one occupied independently.

    A cushion is occupied when a person's seating anchor falls on it (see
    :func:`_occupied_region_indices`). The result lets the picker offer a free
    cushion of a sofa that already has someone on it, instead of writing the
    whole sofa off as taken.
    """
    regions = split_seat_regions(sofa_bbox, has_middle=has_middle)
    occ = _occupied_region_indices(
        persons, [rb for _label, rb in regions],
        hard_overlap=hard_overlap, conf_thresh=conf_thresh,
    )
    parts: list[SeatPart] = []
    for i, (label, rb) in enumerate(regions):
        rx1, ry1, rx2, ry2 = rb
        parts.append(SeatPart(
            label=label,
            bbox_xyxy=rb,
            center_px=((rx1 + rx2) / 2, (ry1 + ry2) / 2),
            occupied=i in occ,
        ))
    return parts


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
    # Occupancy is decided from where people actually sit (their hip anchor); the
    # overlap floor is only a fallback for a person whose hips weren't detected,
    # so it's set high enough that a neighbour's box clipping a free seat's edge
    # no longer marks it taken.
    hard_overlap = float(os.getenv("HRI_SEAT_OCCUPIED_HARD_OVERLAP", "0.5"))
    conf_thresh = float(os.getenv("HRI_POSE_KP_CONF", "0.3"))
    sofa_has_middle = os.getenv("HRI_SOFA_HAS_MIDDLE", "1").lower() in ("1", "true", "yes")
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

    seats: list[SeatCandidate] = []
    for det in detections:
        if det.class_name and det.class_name.lower() not in [c.lower() for c in seat_classes]:
            continue
        x1, y1, x2, y2 = det.bbox
        bbox = (x1, y1, x2, y2)
        if is_sofa_class(det.class_name):
            # A sofa is parsed into cushions; it only counts as fully occupied
            # when every cushion is taken — otherwise a free cushion is offered.
            parts = parse_sofa_parts(
                bbox, persons, has_middle=sofa_has_middle,
                hard_overlap=hard_overlap, conf_thresh=conf_thresh,
            )
            occupied = bool(parts) and all(p.occupied for p in parts)
        else:
            parts = None
            occupied = bool(_occupied_region_indices(
                persons, [bbox], hard_overlap=hard_overlap, conf_thresh=conf_thresh,
            ))
        seats.append(
            SeatCandidate(
                bbox_xyxy=bbox,
                class_name=det.class_name or "seat",
                confidence=det.confidence or 0.0,
                center_px=((x1 + x2) / 2, (y1 + y2) / 2),
                occupied=occupied,
                parts=parts,
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


def resolve_free_part(
    seat: SeatCandidate | None, label: str | None = None
) -> SeatPart | None:
    """The cushion to actually offer on a sofa, or None for an ordinary seat.

    With a *label* ("LEFT"/"MIDDLE"/"RIGHT") the matching cushion is returned
    when it's free; otherwise (no/invalid/taken label) the first free cushion is
    used. None when the seat has no cushions or every cushion is taken — the
    caller then falls back to the whole-seat bbox.
    """
    if seat is None or not seat.parts:
        return None
    free = [p for p in seat.parts if not p.occupied]
    if label:
        for p in free:
            if p.label == (label or "").upper():
                return p
    return free[0] if free else None


def find_seated_person_bbox(
    persons: list[PersonPose], seats: list[SeatCandidate]
) -> BBox | None:
    """The bbox of a person actually sitting on one of the detected seats.

    A person is seated when their seating anchor falls within a seat's bbox (any
    cushion of a sofa counts, since a sofa with a free cushion is no longer
    flagged occupied). A lone detected person is accepted too; None when nobody
    lines up with a seat.
    """
    seat_boxes = [s.bbox_xyxy for s in seats]
    for p in persons:
        if any(_point_in_box(person_seat_anchor(p), sb) for sb in seat_boxes):
            return cxcywh_to_xyxy(p.bbox)
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


# --- Persisted people positions (ctx.data blackboard) ------------------------
# Where each seated person ("host"/"guest-1"/"guest-2") was last seen, in the
# map frame, so a later step can face them without re-scanning. People may
# switch seats, so every write is latest-wins (a fresh sighting overwrites the
# old point) and reset_people_positions clears the lot at the start of a run.


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


def approach_point(
    ctx: TaskContext,
    x: float,
    y: float,
    *,
    stop_distance: float,
    blocking: bool = True,
    settle: float = 0.0,
) -> bool:
    """Drive toward a map-frame point, halting *stop_distance* m short of it.

    The goal is computed on the line from the robot's current position to
    (x, y), pulled back by *stop_distance*, and the base is left facing the
    target — so it approaches a person without driving onto them. If already
    within *stop_distance*, it only rotates to face the point (no forward
    creep). False when odometry has no fix (current_pose's zeros fallback would
    send the base to an arbitrary goal, so it's not used here) or nav fails.

    With *blocking* False the base is commanded but not waited on
    (``nav.go_to(blocking=False)``) and control returns after *settle* seconds
    — so a follow loop can keep re-targeting a moving person every tick instead
    of committing a full blocking drive to an already-stale position.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] approach_point: odometry unavailable ({exc})")
        return False
    if not pose:
        return False
    rx, ry = pose["x"], pose["y"]
    dx, dy = x - rx, y - ry
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    if dist <= stop_distance:
        gx, gy = rx, ry  # already close enough: rotate in place to face them
    else:
        ux, uy = dx / dist, dy / dist
        gx, gy = x - ux * stop_distance, y - uy * stop_distance
    if blocking:
        return ctx.goto(gx, gy, heading)
    try:
        ctx.walkie.nav.go_to(gx, gy, heading, blocking=False)
    except Exception as exc:
        print(f"[skills] approach_point: non-blocking go_to failed ({exc})")
        return False
    if settle > 0:
        time.sleep(settle)  # let the base make progress before re-targeting
    return True


def rotate_by(ctx: TaskContext, delta_rad: float, *, blocking: bool = True) -> bool:
    """Rotate the base by *delta_rad* relative to its current heading.

    Reads the real odometry heading (not ``current_pose``'s zeros fallback,
    which would make a relative turn meaningless) and no-ops without a fix.
    Positive = CCW / left, matching the frame-offset convention used elsewhere
    (a target left of center needs a positive turn to re-center it).

    With *blocking* False the turn is commanded but not waited on
    (``nav.go_to(blocking=False)``, the goal preempting any in-flight one), so a
    fast control loop can re-issue a fresh correction every tick — closed-loop,
    since the new target heading is recomputed from the latest odometry each
    time. Default True keeps the one-shot "look toward" behaviour.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] rotate_by: odometry unavailable ({exc})")
        return False
    if not pose:
        return False
    target = pose["heading"] + delta_rad
    if blocking:
        return ctx.rotate_to(target)
    try:
        ctx.walkie.nav.go_to(pose["x"], pose["y"], target, blocking=False)
    except Exception as exc:
        print(f"[skills] rotate_by: non-blocking go_to failed ({exc})")
        return False
    return True


class MotionPredictor:
    """Estimate a tracked person's map-frame velocity and extrapolate where they
    went when they briefly leave the camera frame.

    Sightings are ``(t, x, y)`` map-frame points (``lift_bbox_world_xy`` output).
    Because they live in the fixed map frame, differencing them gives the
    person's true ground velocity — the robot's own motion doesn't pollute it.
    :meth:`velocity` is a least-squares linear fit over a sliding time window
    (robust to the per-frame depth-lift jitter a two-point difference would
    amplify); :meth:`predict` extrapolates from the last sighting, clamped to a
    human-plausible speed and a max horizon + distance so a noisy fit can't
    fling the goal across the room.

    :meth:`predict` returns ``None`` — telling the follow loop to fall back to a
    rotate-search — when there is too little history, the person was essentially
    stationary (no direction to extrapolate), or the last sighting is too stale
    to trust. All tunables come from ``HRI_FOLLOW_PREDICT_*`` env vars.
    """

    def __init__(self) -> None:
        self.window_sec = float(os.getenv("HRI_FOLLOW_PREDICT_WINDOW_SEC", "4.0"))
        self.min_samples = int(os.getenv("HRI_FOLLOW_PREDICT_MIN_SAMPLES", "3"))
        self.horizon_sec = float(os.getenv("HRI_FOLLOW_PREDICT_HORIZON_SEC", "3.0"))
        self.max_speed = float(os.getenv("HRI_FOLLOW_PREDICT_MAX_SPEED", "1.5"))
        self.min_speed = float(os.getenv("HRI_FOLLOW_PREDICT_MIN_SPEED", "0.15"))
        self.max_dist = float(os.getenv("HRI_FOLLOW_PREDICT_MAX_DIST", "3.0"))
        self.gap_reset_sec = float(os.getenv("HRI_FOLLOW_PREDICT_GAP_RESET_SEC", "3.0"))
        self._hist: list[tuple[float, float, float]] = []

    def update(self, t: float, xy: tuple[float, float]) -> None:
        """Record a fresh map-frame sighting at monotonic time *t*."""
        if self._hist and t - self._hist[-1][0] > self.gap_reset_sec:
            self._hist.clear()  # long blackout: the old trajectory is stale
        self._hist.append((t, xy[0], xy[1]))
        cutoff = t - self.window_sec
        self._hist = [s for s in self._hist if s[0] >= cutoff]

    def velocity(self) -> tuple[float, float] | None:
        """Least-squares ``(vx, vy)`` over the window, or None if degenerate."""
        if len(self._hist) < self.min_samples:
            return None
        ts = [s[0] for s in self._hist]
        t0 = ts[0]
        tu = [t - t0 for t in ts]
        n = len(tu)
        mean_t = sum(tu) / n
        denom = sum((t - mean_t) ** 2 for t in tu)
        if denom <= 1e-6:  # all samples ~same timestamp: no usable slope
            return None

        def slope(vals: list[float]) -> float:
            mean_v = sum(vals) / n
            return sum((tu[i] - mean_t) * (vals[i] - mean_v) for i in range(n)) / denom

        return slope([s[1] for s in self._hist]), slope([s[2] for s in self._hist])

    def predict(self, t: float) -> tuple[float, float] | None:
        """Extrapolated map-frame point at time *t*, or None to fall back to search."""
        if not self._hist:
            return None
        v = self.velocity()
        if v is None:
            return None
        vx, vy = v
        speed = math.hypot(vx, vy)
        if speed < self.min_speed:
            return None  # essentially stationary — no heading to extrapolate
        if speed > self.max_speed:  # clamp to a human-plausible walking speed
            vx, vy = vx / speed * self.max_speed, vy / speed * self.max_speed
        lt, lx, ly = self._hist[-1]
        dt = t - lt
        if dt > self.horizon_sec:
            return None  # last sighting too old to trust the extrapolation
        dt = max(0.0, dt)
        px, py = lx + vx * dt, ly + vy * dt
        ddx, ddy = px - lx, py - ly
        dist = math.hypot(ddx, ddy)
        if dist > self.max_dist:  # cap displacement from the last real sighting
            px, py = lx + ddx / dist * self.max_dist, ly + ddy / dist * self.max_dist
        return px, py

    def reset(self) -> None:
        self._hist.clear()


def _draw_follow_viz(img, box, *, color, label, save_path) -> None:
    """Best-effort: draw the tracked person's box + a status banner on *img*,
    then write it to *save_path* (overwritten each call).

    Used by :func:`follow_person` when ``HRI_FOLLOW_VIZ=1`` so you can watch which
    body the robot is locked onto — *box* is the tracked person on the live frame
    (which already shows everyone else), *color* encodes the loop state, and
    *label* is the one-line status. Never raises: a viz glitch must not disturb
    the follow loop.
    """
    try:
        from PIL import ImageDraw

        vis = img.convert("RGB").copy()
        draw = ImageDraw.Draw(vis)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.rectangle((0, 0, vis.width, 22), fill=(0, 0, 0))
        draw.text((6, 6), label, fill=color)
        vis.save(save_path, "JPEG", quality=70)
    except Exception as exc:
        print(f"[skills] follow viz failed ({exc})")


def follow_person(
    ctx: TaskContext,
    select,
    *,
    stopper=None,
    on_warmup=None,
    on_lost=None,
    on_stopped=None,
    follow_distance: float | None = None,
    timeout: float | None = None,
    nav_period: float | None = None,
    search_step_deg: float | None = None,
    max_lost: int | None = None,
    lead_gain: float | None = None,
    predict_on: bool | None = None,
) -> str:
    """Follow a person by re-targeting nav at their (predicted) position each tick.

    A dead-simple synchronous loop — no background thread. Every tick it
    snapshots the camera, asks *select* for the person's box, lifts it to a
    map-frame point, and drives ``nav.go_to`` there (led forward by the target's
    velocity × *lead_gain* seconds), stopping *follow_distance* m short. Because
    nav orients the base along its travel direction, driving to the target also
    faces it — there is no separate look-at/centering step, so the faster the
    loop runs the tighter it tracks. A :class:`MotionPredictor` keeps a running
    velocity estimate; when a tick yields no usable point (no detection, or the
    depth lift failed) the loop COASTS on the predictor's extrapolation, and only
    once that is exhausted does it rotate-search toward the last-seen side.
    Returns the exit reason: ``'stopped'`` | ``'lost'`` | ``'timeout'``.

    *select* is a callable ``(ctx, snap) -> bbox_xyxy | None`` — e.g.
    :func:`select_largest_person` (follow whoever's closest, for testing) or
    ``identity.select_person_to_follow`` (match an enrolled person by face first,
    attire fallback). *stopper*, when given, is a context manager carrying a ``.placed``
    :class:`threading.Event` (a :class:`CommandListener`); it is entered AFTER
    *on_warmup* — so a background mic listener never transcribes the warmup
    speech — and its event ends the loop the instant it is set. *on_warmup* runs
    once before the loop (e.g. a spoken ack); *on_lost* runs once the first time
    the target is lost; *on_stopped* runs after a *stopper*-triggered exit, before
    teardown, so a closing speech overlaps the join cost. Every tunable defaults
    from the ``HRI_FOLLOW_*`` env vars.
    """
    if follow_distance is None:
        follow_distance = float(os.getenv("HRI_FOLLOW_DISTANCE_M", "1.0"))
    if timeout is None:
        timeout = float(os.getenv("HRI_FOLLOW_TIMEOUT_SEC", "90"))
    if nav_period is None:
        nav_period = float(os.getenv("HRI_FOLLOW_NAV_PERIOD_SEC", "0"))
    if search_step_deg is None:
        search_step_deg = float(os.getenv("HRI_FOLLOW_SEARCH_STEP_DEG", "10"))
    search_step = math.radians(search_step_deg)
    if max_lost is None:
        max_lost = int(os.getenv("HRI_FOLLOW_MAX_LOST", "8"))
    if lead_gain is None:
        lead_gain = float(os.getenv("HRI_FOLLOW_LEAD_GAIN", "0.5"))
    if predict_on is None:
        predict_on = os.getenv("HRI_FOLLOW_PREDICT", "1").lower() in ("1", "true", "yes")
    # Keep driving to the last seen point for this long (s) when a tick yields no
    # detection — a dropped pose frame or a briefly-stationary person must NOT
    # fall straight into the rotate-search (which would otherwise stutter / stall).
    lost_grace = float(os.getenv("HRI_FOLLOW_LOST_GRACE_SEC", "1.5"))
    debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
    # Visualization: annotate each frame with the tracked person's box + status
    # and write it to HRI_FOLLOW_VIZ_PATH (overwritten every tick) so you can
    # watch who the robot is locked onto. Separate flag from the timing debug.
    viz = os.getenv("HRI_FOLLOW_VIZ", "0").lower() in ("1", "true", "yes")
    viz_path = os.getenv("HRI_FOLLOW_VIZ_PATH", "follow_viz.jpg")

    predictor = MotionPredictor() if predict_on else None
    idle = threading.Event()  # never set — an interruptible sleep when no stopper
    lost = 0
    last_dir = 1.0  # default search direction (left) until first seen
    last_xy: tuple[float, float] | None = None  # last good lifted point, for the grace hold
    last_seen_t: float | None = None
    last_box: BBox | None = None  # last selected box, drawn dim while coasting (viz)
    deadline = time.monotonic() + timeout
    reason = "timeout"
    n, t_log = 0, time.monotonic()
    if on_warmup is not None:
        on_warmup()  # speak the ack BEFORE the stopper starts (so the mic skips it)
    with (stopper if stopper is not None else nullcontext()) as listener:
        placed = getattr(listener, "placed", None)
        while time.monotonic() < deadline and not (placed is not None and placed.is_set()):
            now = time.monotonic()
            snap = ctx.snapshot()
            t_snap = time.monotonic()
            box = select(ctx, snap) if snap is not None else None
            t_sel = time.monotonic()
            # In view: lift the box and lead it by the fitted velocity. The lift
            # feeds the predictor (raw), so it can coast when a tick comes up empty.
            side = None
            target = None
            if box is not None:
                side = (box[0] + box[2]) / 2 / snap.img.width - 0.5
                last_dir = 1.0 if side < 0 else -1.0  # left of center -> turn left (+)
                xy = lift_bbox_world_xy(ctx, snap, box, use_edge_filter=False)
                if xy is not None:
                    last_xy, last_seen_t, last_box = xy, now, box
                    if predictor is not None:
                        predictor.update(now, xy)
                        vel = predictor.velocity()
                        if vel is not None and lead_gain:
                            xy = (xy[0] + lead_gain * vel[0], xy[1] + lead_gain * vel[1])
                    target = xy
            if target is None:
                # No fresh point this tick — COAST instead of searching: use the
                # predictor's extrapolation if it has one (target moving), else
                # hold the last seen point for the grace window (target stationary,
                # or a single dropped detection frame). Only past the grace do we
                # actually search.
                pred = predictor.predict(now) if predictor is not None else None
                if pred is not None:
                    target = pred
                elif (
                    last_xy is not None
                    and last_seen_t is not None
                    and (now - last_seen_t) <= lost_grace
                ):
                    target = last_xy
            if target is not None:
                lost = 0
                approach_point(ctx, *target, stop_distance=follow_distance, blocking=False)
            else:
                # Truly lost (no detection, no prediction, grace expired): turn
                # toward the last-seen side until the target reappears. NON-blocking
                # so a single search turn can't freeze the loop (a blocking go_to to
                # the robot's own pose can stall on a nav recovery for seconds).
                lost += 1
                if lost == 1 and on_lost is not None:
                    on_lost()  # nudge once, then keep searching
                if lost >= max_lost:
                    print("[skills] follow_person: lost the target past the search budget")
                    reason = "lost"
                    break
                rotate_by(ctx, last_dir * search_step, blocking=False)
            if viz and snap is not None:
                # Green = tracking a fresh detection; yellow = coasting on the last
                # seen box (prediction/grace hold); red = searching, nothing to draw.
                if box is not None:
                    draw_box, color, state = box, (0, 255, 0), "TRACK"
                elif target is not None:
                    draw_box, color, state = last_box, (255, 200, 0), "COAST"
                else:
                    draw_box, color, state = None, (255, 60, 60), "SEARCH"
                tx = f"({target[0]:.2f},{target[1]:.2f})" if target is not None else "-"
                sd = f"{side:+.2f}" if side is not None else "-"
                _draw_follow_viz(
                    snap.img, draw_box, color=color, save_path=viz_path,
                    label=f"{state} side={sd} target={tx} lost={lost}",
                )
            if debug:
                n += 1
                state = "drive" if target is not None else f"search(lost={lost})"
                print(f"[follow_person] {state} box={'Y' if box is not None else 'N'} "
                      f"snap={1e3 * (t_snap - now):.0f}ms select={1e3 * (t_sel - t_snap):.0f}ms "
                      f"lift+nav={1e3 * (time.monotonic() - t_sel):.0f}ms")
                if time.monotonic() - t_log >= 3.0:
                    print(f"[follow_person] {n / (time.monotonic() - t_log):.1f} Hz")
                    n, t_log = 0, time.monotonic()
            # Optional extra inter-command delay; 0 = flat out (snapshot+lift paces).
            (placed or idle).wait(nav_period)
        if placed is not None and placed.is_set():
            reason = "stopped"
            if on_stopped is not None:
                on_stopped()  # speak while the threads wind down
    return reason


def person_bboxes(ctx: TaskContext, img: Image.Image) -> list[BBox]:
    """All detected person boxes (xyxy) in *img* via pose estimation; [] on failure.

    This is the *fast*, real-time path: pose estimation only, with NO face or
    attire recognition. Those per-person embeddings (run by ``locate_people``)
    are the expensive part — keeping them out of the per-tick loop is what lets
    the tracker sample at pose-estimation rate.
    """
    try:
        persons = ctx.walkieAI.pose_estimation.estimate(img)
    except Exception as exc:
        print(f"[skills] person_bboxes: pose estimation failed ({exc})")
        return []
    return [cxcywh_to_xyxy(p.bbox) for p in persons]


def nearest_person_bbox(ctx: TaskContext, img: Image.Image) -> BBox | None:
    """The largest (so nearest) detected person box in *img*, or None.

    Fallback target for following when no enrolled identity is matched — the
    biggest body in view is the one the robot is closest to.
    """
    boxes = person_bboxes(ctx, img)
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def select_largest_person(ctx: TaskContext, snap) -> BBox | None:
    """:func:`follow_person` selector: the largest (nearest) person box, no identity.

    Pose estimation only — picks the biggest body in view, the one the robot is
    closest to. Use to exercise the follow loop without enrolling anyone. *snap*
    is the current CameraSnapshot (the loop lifts the returned box against it);
    ``None`` snap or no person detected → ``None``.
    """
    if snap is None:
        return None
    return nearest_person_bbox(ctx, snap.img)


def classify_host_command(ctx: TaskContext, heard: str) -> str:
    """LLM intent of a heard utterance: ``'follow'``, ``'place'``, or ``'other'``.

    Filters genuine host instructions from the party-crowd chatter the mic
    picks up during the bag handover. An empty/garbled transcript or an
    extraction failure maps to ``'other'`` so the robot never acts on noise.
    """
    if not heard.strip():
        return "other"
    cmd = ctx.extract(
        prompts.HostCommand, prompts.CLASSIFY_HOST_COMMAND_INSTRUCTIONS, heard
    )
    return cmd.intent if cmd is not None else "other"


class CommandListener:
    """Background mic loop that catches a host command without ever going deaf.

    While the robot drives after the host it must still hear "put the bag
    here". Doing record -> transcribe -> classify serially on the nav thread
    would leave the microphone dark during every drive step AND during every
    STT + LLM round-trip — precisely when the host is most likely to speak. So
    this splits the work across two daemon threads:

    * a *recorder* that loops ``record_until_silence`` and re-arms IMMEDIATELY,
      pushing each captured clip onto a queue without waiting for STT or the
      LLM (so the mic is re-listening within milliseconds of a phrase ending);
    * a *worker* that drains the queue, transcribes + classifies each clip
      (:func:`classify_host_command`) and sets :attr:`placed` when the host
      says to put the bag down.

    The follow loop polls :attr:`placed` instead of calling ``ctx.listen``.
    Only this listener may touch the microphone while it runs — one
    ``sd.InputStream`` can be open at a time, so the loop must not call
    ``ctx.listen`` concurrently. Best-effort: every mic / STT / model failure
    is swallowed so a glitch never crashes the step. Use as a context manager::

        with CommandListener(ctx) as listener:
            while ...:
                ... drive toward the host ...
                if listener.placed.is_set():
                    break
    """

    def __init__(self, ctx: TaskContext, *, record_timeout: float | None = None) -> None:
        self.ctx = ctx
        self.placed = threading.Event()      # set once a 'place' command is heard
        self.last_text = ""                  # most recent transcript (debug/inspection)
        self.last_intent: str | None = None  # most recent classified intent
        self.record_timeout = (
            record_timeout
            if record_timeout is not None
            else float(os.getenv("HRI_FOLLOW_RECORD_TIMEOUT_SEC", "5"))
        )
        self._stop = threading.Event()
        self._queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._threads: list[threading.Thread] = []

    def start(self) -> "CommandListener":
        if self._threads:
            return self
        self._stop.clear()
        self.placed.clear()
        self._threads = [
            threading.Thread(target=self._record_loop, daemon=True),
            threading.Thread(target=self._work_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)  # wake the worker so it can see the stop flag
        for t in self._threads:
            # The recorder may be mid-capture (blocked up to record_timeout);
            # give it that long plus a margin to wind down its InputStream.
            t.join(timeout=max(2.0, self.record_timeout + 1.0))
        self._threads = []

    def __enter__(self) -> "CommandListener":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _record_loop(self) -> None:
        # Dev/CI path (DISABLE_LISTENING): no mic — read typed lines and feed
        # them straight to the classifier (no latency to hide, so no queue).
        if self.ctx.disable_listening:
            while not self._stop.is_set():
                try:
                    line = input("[listen] > ")
                except EOFError:
                    return
                self._handle_text(line)
            return
        while not self._stop.is_set():
            try:
                audio = self.ctx.walkie.microphone.record_until_silence(
                    timeout=self.record_timeout
                )
            except Exception as exc:
                print(f"[skills] CommandListener record failed ({exc})")
                self._stop.wait(0.2)
                continue
            if audio and not self._stop.is_set():
                self._queue.put(audio)  # hand off; re-arm the mic right away

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            audio = self._queue.get()
            if audio is None:  # stop sentinel
                return
            try:
                text = self.ctx.walkieAI.stt.transcribe(audio)
            except Exception as exc:
                print(f"[skills] CommandListener STT failed ({exc})")
                continue
            self._handle_text(text)

    def _handle_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        print(f"[heard] {text}")
        intent = classify_host_command(self.ctx, text)
        self.last_text, self.last_intent = text, intent
        if intent == "place":
            self.placed.set()


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
        if seat.parts:
            # A sofa: report each cushion so the model can offer a free one even
            # when someone else is on it. The whole-sofa status is "free" as long
            # as any cushion is open.
            free_parts = [p.label for p in seat.parts if not p.occupied]
            who = f" — {_person_label(occupants[i], person_names)} is on it" if i in occupants else ""
            status = ("OCCUPIED (all cushions taken)" if seat.occupied
                      else f"has free cushion(s): {', '.join(free_parts)}") + who
            cushions = "; ".join(
                f"{p.label} cushion (center x={p.center_px[0]:.0f}px) "
                f"{'taken' if p.occupied else 'FREE'}"
                for p in seat.parts
            )
            status += f" [{cushions}]"
        elif i in occupants:
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
) -> tuple[SeatCandidate | None, SeatPart | None, str | None]:
    """Let the LLM choose which seat to offer and word the spoken offer.

    Returns (seat, part, announcement). *part* is the chosen sofa cushion
    (LEFT/MIDDLE/RIGHT) when the seat is a sofa, else None — the caller faces and
    announces that cushion rather than the whole sofa. The model sees the whole
    frame (seats, each sofa's free cushions, who is recognized where, the host,
    the other guest's seat) so it can offer a free cushion next to the host and
    refer to the host in the announcement. A null announcement means "use the
    default line". An explicit null seat from the model means "nothing suitable";
    an extraction failure or out-of-range index degrades to the deterministic
    pick_free_seat (whose first free cushion is then used for a sofa).
    """
    if not seats:
        return None, None, None
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
        seat = pick_free_seat(seats)
        return seat, resolve_free_part(seat), None
    if choice.seat_index is None:
        print(f"[skills] LLM declined to pick a seat ({choice.reason or 'no reason given'})")
        return None, None, None
    if not 0 <= choice.seat_index < len(seats):
        print(f"[skills] LLM seat index {choice.seat_index} out of range; using heuristic pick")
        seat = pick_free_seat(seats)
        return seat, resolve_free_part(seat), None
    seat = seats[choice.seat_index]
    part = resolve_free_part(seat, getattr(choice, "seat_part", None))
    print(f"[skills] LLM picked seat [{choice.seat_index}] {seat.class_name}"
          f"{f' ({part.label} cushion)' if part else ''}"
          f" ({choice.reason or 'no reason given'})")
    return seat, part, (choice.announcement or "").strip() or None


# ---------------------------------------------------------------------------
# Face presence & tracking
# ---------------------------------------------------------------------------
# The head servo only tilts, so "face the person" is a small in-place BASE
# rotation (nav.go_to, like sweep_snapshots/face_point). We track the single
# biggest face in view — the person standing right in front — and ignore any
# face whose bbox area is below HRI_FACE_MIN_AREA_PX, so a distant bystander
# never pulls the robot.


def biggest_face(
    ctx: TaskContext, img: Image.Image | None = None, *, min_area: float = 0.0
) -> FaceEmbedding | None:
    """The largest detected face in *img* (or a fresh capture) above *min_area* px².

    Returns the nearest (largest-bbox) face, or None when capture/detection
    fails or no face clears the area floor. Best-effort: never raises.
    """
    if img is None:
        img = ctx.capture()
    if img is None:
        return None
    try:
        faces = ctx.walkieAI.face_recognition.embed(img)
    except Exception as exc:
        print(f"[skills] face detection failed ({exc})")
        return None
    faces = [f for f in faces if f.area() >= min_area]
    if not faces:
        return None
    return max(faces, key=lambda f: f.area())


def wait_for_person(
    ctx: TaskContext,
    *,
    min_area: float | None = None,
    timeout: float | None = None,
    poll: float | None = None,
) -> bool:
    """Block until a face bigger than *min_area* px² is in view, or *timeout* s.

    Polls the camera every *poll* seconds. Returns True as soon as someone is
    standing in front, False if the timeout elapses first (the caller proceeds
    anyway so a no-show can't stall the run). Params default from the
    ``HRI_FACE_*`` env vars.
    """
    if min_area is None:
        min_area = float(os.getenv("HRI_FACE_MIN_AREA_PX", "10000"))
    if timeout is None:
        timeout = float(os.getenv("HRI_FACE_WAIT_TIMEOUT_SEC", "30"))
    if poll is None:
        poll = float(os.getenv("HRI_FACE_WAIT_POLL_SEC", "0.5"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if biggest_face(ctx, min_area=min_area) is not None:
            return True
        time.sleep(poll)
    return False


class FaceTracker:
    """Background loop that keeps the base pointed at the biggest face in view.

    Each tick captures a frame, picks the largest face above
    ``HRI_FACE_MIN_AREA_PX`` (the person right in front), and rotates the base
    by the face's measured angular offset from the optical axis (scaled by
    ``HRI_FACE_TRACK_GAIN`` and capped at ``HRI_FACE_TRACK_MAX_STEP_DEG``) so it
    re-centers. A dead-band (``HRI_FACE_TRACK_DEADBAND_PX``) suppresses jitter
    when the face is already roughly centered. Runs in its own daemon thread so
    the calling step can ask questions / caption meanwhile; every camera /
    detector / nav failure is swallowed so a tracking glitch never crashes the
    step.

    The head servo only tilts, so this rotates the BASE (nav.go_to) — only use
    it while nothing else is driving the base. Use as a context manager::

        with FaceTracker(ctx):
            ... ask the guest their name, caption their appearance ...
    """

    def __init__(self, ctx: TaskContext) -> None:
        self.ctx = ctx
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.min_area = float(os.getenv("HRI_FACE_MIN_AREA_PX", "10000"))
        self.interval = float(os.getenv("HRI_FACE_TRACK_INTERVAL_SEC", "0.6"))
        self.deadband_px = float(os.getenv("HRI_FACE_TRACK_DEADBAND_PX", "80"))
        self.max_step = math.radians(float(os.getenv("HRI_FACE_TRACK_MAX_STEP_DEG", "20")))
        self.gain = float(os.getenv("HRI_FACE_TRACK_GAIN", "0.8"))
        self.hfov = math.radians(float(os.getenv("HRI_CAMERA_HFOV_DEG", "110")))

    def start(self) -> "FaceTracker":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3))
            self._thread = None

    def __enter__(self) -> "FaceTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                print(f"[skills] face tracker tick failed ({exc})")
            self._stop.wait(self.interval)

    def _tick(self) -> None:
        img = self.ctx.capture()
        if img is None:
            return
        face = biggest_face(self.ctx, img, min_area=self.min_area)
        if face is None:
            return
        x1, _y1, x2, _y2 = face.bbox_xyxy
        face_cx = (x1 + x2) / 2
        # Pixel offset of the face from frame center (positive = face is left of
        # center, i.e. the robot must turn left / CCW / +heading to re-center).
        offset_px = img.width / 2 - face_cx
        if abs(offset_px) <= self.deadband_px:
            return
        delta = (offset_px / img.width) * self.hfov * self.gain
        delta = max(-self.max_step, min(self.max_step, delta))
        pose = self.ctx.current_pose()
        self.ctx.rotate_to(pose["heading"] + delta)


def side_relative_to_listener(
    ctx: TaskContext,
    listener_xy: tuple[float, float],
    subject_xy: tuple[float, float],
) -> str | None:
    """Which side ("left"/"right") *subject_xy* sits on from the listener's own
    point of view, ASSUMING the listener faces the robot (people turn toward
    whoever is speaking to them). None when odometry has no fix.

    The listener's facing is taken as robot - listener; the 2D cross product of
    that facing with the listener->subject vector is positive when the subject
    is counter-clockwise of the facing direction, i.e. on the listener's left.
    """
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] side_relative_to_listener: odometry unavailable ({exc})")
        return None
    if not pose:
        return None
    fx, fy = pose["x"] - listener_xy[0], pose["y"] - listener_xy[1]
    sx, sy = subject_xy[0] - listener_xy[0], subject_xy[1] - listener_xy[1]
    cross = fx * sy - fy * sx
    if cross == 0:
        return None
    return "left" if cross > 0 else "right"


def _guest_intro_fallback(
    listener_name: str | None,
    subject_name: str | None,
    subject_drink: str | None,
    side: str | None,
) -> str:
    """Template introduction spoken to one guest about the other beside them."""
    who = f"the person on your {side}" if side else "the guest next to you"
    name = subject_name or prompts.GENERIC_OTHER_GUEST
    lead = f"{listener_name}, " if listener_name else ""
    line = f"{lead}{who} is {name}."
    if subject_drink:
        line += f" Their favorite drink is {subject_drink}."
    return line


def llm_guest_intro_speeches(ctx: TaskContext, acts: list[dict]) -> dict[int, str]:
    """Word both guest-to-guest introductions in ONE LLM call: {listener: text}.

    *acts* is one dict per spoken line — the robot FACES that listener and
    presents the other guest beside them::

        {"listener": 1|2, "listener_name": str|None,
         "subject_name": str|None, "subject_drink": str|None,
         "side": "left"|"right"|None}

    Returns a speech keyed by listener number, each falling back to a template
    on extraction failure.
    """
    fallback = {
        a["listener"]: _guest_intro_fallback(
            a["listener_name"], a["subject_name"], a["subject_drink"], a["side"]
        )
        for a in acts
    }
    lines = []
    for a in acts:
        lines.append(
            f"While facing guest {a['listener']} (name="
            f"{a['listener_name'] or 'unknown'}), present the OTHER guest "
            f"(name={a['subject_name'] or 'unknown'}; favorite drink="
            f"{a['subject_drink'] or 'unknown'}), who is on guest "
            f"{a['listener']}'s {a['side'] or 'unknown'} side."
        )
    speeches = ctx.extract(
        prompts.GuestIntroSpeeches, prompts.GUEST_INTRO_INSTRUCTIONS, "\n".join(lines)
    )
    if speeches is None:
        print("[skills] guest intro speech extraction failed; using template lines")
        return fallback
    by_listener = {
        1: (speeches.facing_guest_1 or "").strip(),
        2: (speeches.facing_guest_2 or "").strip(),
    }
    return {a["listener"]: by_listener.get(a["listener"], "") or fallback[a["listener"]] for a in acts}
