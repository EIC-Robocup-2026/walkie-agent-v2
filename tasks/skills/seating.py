"""Seat / sofa detection, cushion occupancy, and people<->seat matching.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

import math
import os
import time

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from client import PersonPose
from tasks.base import TaskContext

from .geometry import BBox, _point_in_box, cxcywh_to_xyxy, overlap_fraction, person_seat_anchor
from .lift import lift_bbox_world_xy


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
    conf_thresh: float,
) -> set[int]:
    """Which of *region_boxes* a person is sitting on (one region per person).

    Everyone in view is assumed SEATED — during a seat scan the only standing
    person (the newly arrived guest) is next to the robot, outside the seating
    area. So each person simply claims the single region containing their
    seating anchor (:func:`person_seat_anchor`: hips when detected, else the
    bbox's lower-centre). A person whose anchor lies on none of these regions is
    seated on something else (another seat, or one the detector missed) and
    occupies none of them.
    """
    occupied: set[int] = set()
    for p in persons:
        anchor = person_seat_anchor(p, conf_thresh)
        for i, rb in enumerate(region_boxes):
            if _point_in_box(anchor, rb):
                occupied.add(i)
                break  # a person sits on one cushion
    return occupied


def parse_sofa_parts(
    sofa_bbox: BBox,
    persons: list[PersonPose],
    *,
    has_middle: bool = True,
    conf_thresh: float = 0.3,
) -> list[SeatPart]:
    """Split a sofa into cushions and mark each one occupied independently.

    A cushion is occupied when a person's seating anchor falls on it (see
    :func:`_occupied_region_indices` — everyone in view is assumed seated). The
    result lets the picker offer a free cushion of a sofa that already has
    someone on it, instead of writing the whole sofa off as taken.
    """
    regions = split_seat_regions(sofa_bbox, has_middle=has_middle)
    occ = _occupied_region_indices(
        persons, [rb for _label, rb in regions], conf_thresh=conf_thresh,
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
        return ctx.walkieAI.image.estimate_poses(img)
    except Exception as exc:
        print(f"[skills] pose estimation failed ({exc})")
        return []


def scan_seats(
    ctx: TaskContext,
    timings: dict | None = None,
    detect_per_class: bool | None = None,
):
    """One frame: detect seats (open-vocab) + people, mark occupancy.

    Returns (seats, persons, snap) where ``snap`` is the CameraSnapshot the
    frame came from — its frozen depth/pose geometry lifts seat/person bboxes
    to map-frame points later (lift_bbox_world_xy), immune to the LLM/detection
    latency that follows. ([], [], None) on capture failure and ([], [], snap)
    on detection failure. A whole sofa counts as one seat for now. Seats,
    persons, and snap.img all come from the same capture, so pixel coordinates
    are comparable and crops line up.

    Per-stage wall-clock in seconds (``snapshot`` / ``detect`` / ``pose`` /
    ``assemble`` / ``total``) is written into *timings* when a dict is passed,
    and printed when ``HRI_SCAN_TIMING=1`` — so a caller can benchmark which
    stage of the scan dominates. A stage that didn't run (e.g. detect after a
    capture failure) is simply absent from the dict.

    *detect_per_class* (defaults to ``HRI_SCAN_DETECT_PER_CLASS``) splits the
    single batched detect into one open-vocab call per seat class, timing each
    into ``detect:<class>`` keys (the plain ``detect`` key still holds the wall
    time of the whole detect stage). This is a benchmark aid — N round-trips are
    slower than the one batched call and detections from different prompts are
    just concatenated (a real object may appear under more than one class), so
    leave it off for the production path.

    Detection and pose estimation are independent remote calls (each AI sub-client
    has its own HTTP session), so by default (``HRI_SCAN_PARALLEL=1``) they run
    concurrently and assembly waits for both — the pose round-trip overlaps detect.
    With parallelism on, ``detect`` and ``pose`` are each true per-call wall-times
    but ``total ≈ snapshot + max(detect, pose) + assemble``, so ``detect + pose``
    no longer sums to ``total``; that gap is the saved time. ``HRI_SCAN_PARALLEL=0``
    restores serial execution.

    Seat-detection frames can be downscaled before sending via
    ``HRI_SEAT_DETECT_MAX_SIZE`` (empty = full resolution); the client scales the
    returned bbox back to input-image coords, so geometry is unchanged.
    """
    if detect_per_class is None:
        detect_per_class = os.getenv("HRI_SCAN_DETECT_PER_CLASS", "0").lower() in ("1", "true", "yes")
    parallel = os.getenv("HRI_SCAN_PARALLEL", "1").lower() in ("1", "true", "yes")
    _sm = os.getenv("HRI_SEAT_DETECT_MAX_SIZE", "").strip()
    seat_max_size = int(_sm) if _sm else None
    stages: dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        t0 = time.perf_counter()
        snap = ctx.snapshot()
        stages["snapshot"] = time.perf_counter() - t0
        if snap is None:
            return [], [], None
        img = snap.img
        seat_classes = [
            c.strip()
            for c in os.getenv("HRI_SEAT_CLASSES", "chair,sofa,armchair,stool").split(",")
            if c.strip()
        ]
        # Occupancy assumes everyone in view is SEATED: each person claims the
        # seat/cushion under their seating anchor (hips when detected, else the
        # bbox's lower-centre). The newly arrived guest stands next to the
        # robot, outside the scanned seating area.
        conf_thresh = float(os.getenv("HRI_POSE_KP_CONF", "0.3"))
        sofa_has_middle = os.getenv("HRI_SOFA_HAS_MIDDLE", "1").lower() in ("1", "true", "yes")

        # Detect + pose are independent; each closure owns its own perf_counter span
        # and writes DISJOINT keys into ``stages`` (detect → detect/detect:*, pose →
        # pose), so they can run on two threads with no lock. A detect hard-fail is
        # signalled back via the flag (not a bare return from the worker thread),
        # so the calling thread still does the ([], [], snap) early-out below.
        def _run_detect():
            t_d = time.perf_counter()
            if detect_per_class:
                # Benchmark mode: one call per class, each timed into detect:<class>.
                # Best-effort per class so a single failing prompt can't abort the scan.
                dets = []
                for cls in seat_classes:
                    tc = time.perf_counter()
                    try:
                        found = ctx.walkieAI.image.detect(
                            img, prompts=[cls], max_size=seat_max_size)
                    except Exception as exc:
                        print(f"[skills] seat detection failed for {cls!r} ({exc})")
                        found = []
                    stages[f"detect:{cls}"] = time.perf_counter() - tc
                    dets.extend(found)
                stages["detect"] = time.perf_counter() - t_d
                return dets, False
            try:
                dets = ctx.walkieAI.image.detect(
                    img, prompts=seat_classes, max_size=seat_max_size)
            except Exception as exc:
                print(f"[skills] seat detection failed ({exc})")
                stages["detect"] = time.perf_counter() - t_d
                return None, True
            stages["detect"] = time.perf_counter() - t_d
            return dets, False

        def _run_pose():
            t_p = time.perf_counter()
            try:
                ppl = ctx.walkieAI.image.estimate_poses(img)
            except Exception as exc:
                print(f"[skills] pose estimation failed ({exc}); assuming all seats free")
                ppl = []
            finally:
                stages["pose"] = time.perf_counter() - t_p
            return ppl

        if parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_detect = ex.submit(_run_detect)
                fut_pose = ex.submit(_run_pose)
                detections, detect_failed = fut_detect.result()
                persons = fut_pose.result()
        else:
            detections, detect_failed = _run_detect()
            persons = _run_pose()
        if detect_failed:
            return [], [], snap

        t0 = time.perf_counter()
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
                    bbox, persons, has_middle=sofa_has_middle, conf_thresh=conf_thresh,
                )
                occupied = bool(parts) and all(p.occupied for p in parts)
            else:
                parts = None
                occupied = bool(_occupied_region_indices(
                    persons, [bbox], conf_thresh=conf_thresh,
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
        stages["assemble"] = time.perf_counter() - t0
        return seats, persons, snap
    finally:
        stages["total"] = time.perf_counter() - t_total
        if timings is not None:
            timings.update(stages)
        if os.getenv("HRI_SCAN_TIMING", "0").lower() in ("1", "true", "yes"):
            order = ("snapshot", "detect", "pose", "assemble", "total")
            print("[skills] scan_seats timing: " + "  ".join(
                f"{k}={stages[k] * 1000:.0f}ms" for k in order if k in stages))
            per_class = {k: v for k, v in stages.items() if k.startswith("detect:")}
            if per_class:
                print("[skills]   detect breakdown: " + "  ".join(
                    f"{k.split(':', 1)[1]}={v * 1000:.0f}ms" for k, v in per_class.items()))


@dataclass
class SeatSweep:
    """A multi-heading seat scan merged into one picture (see scan_seats_sweep).

    ``seats`` is the cross-view de-duplicated list the picker chooses from;
    ``seat_frames``/``seat_world`` are parallel to it (which snapshot each seat
    was seen in, and its lifted map-frame centre when the lift succeeded).
    ``seats_by_frame``/``persons_by_frame`` keep each frame's RAW scan output —
    pixel coordinates are only comparable within one frame, so per-frame
    consumers (find_seated_person_bbox, match_people_to_seats) read these.
    """

    seats: list[SeatCandidate] = field(default_factory=list)
    seat_frames: list[int] = field(default_factory=list)
    seat_world: list[tuple[float, float] | None] = field(default_factory=list)
    seats_by_frame: list[list[SeatCandidate]] = field(default_factory=list)
    persons_by_frame: list[list[PersonPose]] = field(default_factory=list)
    snaps: list = field(default_factory=list)
    offsets_deg: list[float] = field(default_factory=list)

    @property
    def frame_labels(self) -> list[str]:
        """Human-readable view name per frame ("CENTER view", "20° LEFT view")."""
        return [
            "CENTER view" if abs(o) < 1e-6
            else f"{abs(o):.0f}° {'LEFT' if o > 0 else 'RIGHT'} view"
            for o in self.offsets_deg
        ]

    @property
    def center_index(self) -> int:
        """Index of the most forward-facing frame (smallest |heading offset|)."""
        if not self.offsets_deg:
            return 0
        return min(range(len(self.offsets_deg)), key=lambda i: abs(self.offsets_deg[i]))


def scan_seats_sweep(
    ctx: TaskContext,
    offsets_deg: Sequence[float] | None = None,
) -> SeatSweep:
    """Scan for seats at several base headings and merge into one SeatSweep.

    A single forward frame can miss seats (and seated people) off to the side,
    so — like the introduction's sweep_snapshots — the base rotates to each
    heading offset (degrees relative to the heading at entry, positive = left/
    CCW), runs a full :func:`scan_seats` there, and faces forward again. Each
    seat keeps the CameraSnapshot of the very frame it was detected in, so its
    bbox lifts to the correct map-frame point no matter where the robot was
    pointing.

    The views overlap, so the same physical seat is detected more than once;
    every seat's bbox centre is lifted to a map-frame point and instances of the
    same single/multi-seat kind within ``HRI_SEAT_SWEEP_DEDUP_M`` (sofas:
    ``HRI_SEAT_SWEEP_DEDUP_SOFA_M`` — their clipped-view centres wander more)
    are merged. The MOST CENTRAL view's instance is kept: a side view can clip a
    sofa at the frame edge and misjudge its free cushions, so the deliberate
    head-on look is trusted. A seat whose lift failed is kept as-is (it just
    can't dedupe).

    *offsets_deg* defaults to ``(D, 0, -D)`` with ``D = HRI_SEAT_SWEEP_DEG``
    (left, center, right); ``HRI_SEAT_SWEEP_DEG=0`` degrades to one forward
    scan, as does odometry being unavailable (a rotation would aim arbitrarily).
    Best-effort throughout: a failed capture drops that frame, never raises.
    """
    if offsets_deg is None:
        sweep_deg = float(os.getenv("HRI_SEAT_SWEEP_DEG", "20"))
        offsets_deg = (sweep_deg, 0.0, -sweep_deg) if sweep_deg > 0 else (0.0,)
    try:
        pose = ctx.walkie.status.get_position()
    except Exception as exc:
        print(f"[skills] scan_seats_sweep: odometry unavailable ({exc}); single scan")
        pose = None
    if not pose or all(abs(o) < 1e-6 for o in offsets_deg):
        seats, persons, snap = scan_seats(ctx)
        if snap is None:
            return SeatSweep()
        world = [lift_bbox_world_xy(ctx, snap, s.bbox_xyxy) for s in seats]
        return SeatSweep(
            seats=seats, seat_frames=[0] * len(seats), seat_world=world,
            seats_by_frame=[seats], persons_by_frame=[persons],
            snaps=[snap], offsets_deg=[0.0],
        )

    center = pose["heading"]
    settle = float(os.getenv("HRI_SWEEP_SETTLE_SEC", "1.0"))
    snaps: list = []
    kept_offsets: list[float] = []
    seats_by_frame: list[list[SeatCandidate]] = []
    persons_by_frame: list[list[PersonPose]] = []
    for off in offsets_deg:
        ctx.rotate_to(center + math.radians(off))
        if settle > 0:
            time.sleep(settle)  # let the base + depth settle before capturing
        frame_seats, frame_persons, snap = scan_seats(ctx)
        if snap is None:
            print(f"[skills] scan_seats_sweep: capture failed at {off:+.0f}°; skipping")
            continue
        snaps.append(snap)
        kept_offsets.append(off)
        seats_by_frame.append(frame_seats)
        persons_by_frame.append(frame_persons)
    ctx.rotate_to(center)  # leave the robot facing forward again

    # Lift every seat, then de-dup across views, most central frame first so the
    # kept instance carries the best (head-on) geometry and occupancy call.
    dedup_m = float(os.getenv("HRI_SEAT_SWEEP_DEDUP_M", "0.5"))
    sofa_dedup_m = float(os.getenv("HRI_SEAT_SWEEP_DEDUP_SOFA_M", "1.0"))
    entries: list[tuple[int, SeatCandidate, tuple[float, float] | None]] = [
        (fi, seat, lift_bbox_world_xy(ctx, snaps[fi], seat.bbox_xyxy))
        for fi in range(len(snaps))
        for seat in seats_by_frame[fi]
    ]
    kept: list[tuple[int, SeatCandidate, tuple[float, float] | None]] = []
    for idx in sorted(range(len(entries)), key=lambda i: abs(kept_offsets[entries[i][0]])):
        fi, seat, xy = entries[idx]
        sofa = is_sofa_class(seat.class_name)
        thresh = sofa_dedup_m if sofa else dedup_m
        dup = xy is not None and any(
            kxy is not None
            and is_sofa_class(kseat.class_name) == sofa
            and math.hypot(xy[0] - kxy[0], xy[1] - kxy[1]) < thresh
            for _kfi, kseat, kxy in kept
        )
        if not dup:
            kept.append((fi, seat, xy))
    kept.sort(key=lambda e: (e[0], e[1].center_px[0]))  # stable left→right reading order
    n_raw = sum(len(s) for s in seats_by_frame)
    if n_raw != len(kept):
        print(f"[skills] scan_seats_sweep: {n_raw} detections across "
              f"{len(snaps)} views -> {len(kept)} distinct seats")
    return SeatSweep(
        seats=[s for _fi, s, _xy in kept],
        seat_frames=[fi for fi, _s, _xy in kept],
        seat_world=[xy for _fi, _s, xy in kept],
        seats_by_frame=seats_by_frame,
        persons_by_frame=persons_by_frame,
        snaps=snaps,
        offsets_deg=kept_offsets,
    )


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


def match_people_to_seats(
    located: dict[str, tuple[int, BBox]],
    seats: list[SeatCandidate],
    *,
    seat_frames: list[int] | None = None,
    frame_index: int = 0,
    min_overlap: float | None = None,
) -> tuple[dict[int, str], dict[str, tuple[int, BBox]]]:
    """Tie each recognized person to the detected seat they occupy.

    *located* is :func:`identity.locate_people`'s output
    (``{id: (frame_index, person_bbox)}``). Pixel overlap is only meaningful
    within one frame, so a person is compared against the seats detected in the
    frame THEY were recognized in: *seat_frames* (parallel to *seats*, e.g.
    ``SeatSweep.seat_frames``) says which frame each seat came from; without it
    every seat is assumed to be from *frame_index*. A person claims the
    same-frame seat their box overlaps most (at least *min_overlap*, default
    ``HRI_SEAT_OCCUPIED_OVERLAP``), strongest overlap first so a confident match
    isn't blocked by a weak one; one person per seat.

    Returns ``(seat_occupants, seatless)``: *seat_occupants* maps a seat index
    to the id sitting in it; *seatless* maps the id of every recognized person
    who claimed NO seat to their ``(frame_index, box)`` — the detector missed
    their chair, they're on something off-vocabulary (a couch arm, a stool, the
    floor), or their seat was seen (and de-duplicated away) in a different sweep
    view. The caller still knows those people are present and roughly where, so
    the spot isn't wrongly offered as free.
    """
    if min_overlap is None:
        min_overlap = float(os.getenv("HRI_SEAT_OCCUPIED_OVERLAP", "0.25"))
    if seat_frames is None:
        seat_frames = [frame_index] * len(seats)
    pairs = [
        (overlap_fraction(box, seat.bbox_xyxy), pid, i)
        for pid, (fi, box) in located.items()
        for i, seat in enumerate(seats)
        if seat_frames[i] == fi
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
    seatless = {
        pid: (fi, box) for pid, (fi, box) in located.items()
        if pid not in claimed_pids
    }
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
