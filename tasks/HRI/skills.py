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


@dataclass
class HostEstimate:
    """A control-loop's-eye view of the host, computed at read time.

    *in_view*: a sighting landed within the view-grace window (a brief detector
    dropout does not flip this false). *xy*: the freshly lifted map-frame point
    when in view, LED forward by the host's velocity × the read's ``lead_gain``
    so the robot drives to where they will be (the goal nav faces, so it tracks
    them naturally); else None. *predicted*: the MotionPredictor's extrapolation
    for when the host is out of view (None if not trustworthy). *side*: the last
    horizontal frame offset, for the rotate-search fallback direction.
    """

    in_view: bool
    xy: tuple[float, float] | None
    predicted: tuple[float, float] | None
    side: float | None


class HostTracker:
    """Background perception loop that samples the host's map-frame position as
    fast as the camera + detector allow, feeding a :class:`MotionPredictor` — so
    the control loop can issue navigation goals at its OWN, slower cadence off a
    dense, always-fresh estimate.

    This decouples the *sampling rate* from the *nav-command rate*. The detector
    runs flat out in a daemon thread (or every ``HRI_FOLLOW_SAMPLE_INTERVAL_SEC``
    if a floor is set); the caller polls :meth:`read` only when it is ready to
    command the base — a much longer period, because re-issuing nav goals too
    fast just thrashes the stack and the base can't react that quickly anyway.
    A short detection dropout does not read as "lost": :attr:`HostEstimate.in_view`
    stays true for ``HRI_FOLLOW_VIEW_GRACE_SEC`` after the last sighting, which
    also absorbs single-frame detector flicker.

    *locate* is a callable ``(ctx) -> (world_xy | None, side | None)`` (the
    subtask's sampler). Set ``HRI_FOLLOW_TRACK_DEBUG=1`` to log the achieved
    sample rate every few seconds. Thread-safe — all predictor + shared-state
    access is under one lock; every camera/detector failure is swallowed so a
    glitch never crashes the step. Context manager, like :class:`FaceTracker`.
    """

    def __init__(self, ctx: TaskContext, locate, *, predict_on: bool = True) -> None:
        self.ctx = ctx
        self._locate = locate
        self.predictor = MotionPredictor() if predict_on else None
        self.interval = float(os.getenv("HRI_FOLLOW_SAMPLE_INTERVAL_SEC", "0.0"))
        self.view_grace = float(os.getenv("HRI_FOLLOW_VIEW_GRACE_SEC", "0.6"))
        self.debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_xy: tuple[float, float] | None = None
        self._last_lift_t: float | None = None
        self._last_seen_t: float | None = None
        self._last_side: float | None = None

    def start(self) -> "HostTracker":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval + 1.0))
            self._thread = None

    def __enter__(self) -> "HostTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        n, t_log = 0, time.monotonic()
        while not self._stop.is_set():
            try:
                world_xy, side = self._locate(self.ctx)
            except Exception as exc:
                print(f"[skills] HostTracker sample failed ({exc})")
                world_xy, side = None, None
            now = time.monotonic()
            if self.debug:
                n += 1
                if now - t_log >= 3.0:
                    print(f"[HostTracker] sampling at {n / (now - t_log):.1f} Hz")
                    n, t_log = 0, now
            with self._lock:
                if side is not None:  # a person box was found this sample
                    self._last_seen_t = now
                    self._last_side = side
                    if world_xy is not None:  # ...and it lifted to a map point
                        self._last_xy = world_xy
                        self._last_lift_t = now
                        if self.predictor is not None:
                            self.predictor.update(now, world_xy)
            if self.interval > 0:
                self._stop.wait(self.interval)  # else loop flat out (detector paces)

    def read(self, now: float | None = None, *, lead_gain: float = 0.0) -> HostEstimate:
        """Best current view of the host: in-view flag, LED point, prediction.

        *lead_gain* (seconds) extrapolates the in-view point forward along the
        host's fitted velocity, so the loop drives to where they are heading
        rather than where they were — 0 aims at the current point. The
        out-of-view *predicted* point is unaffected (it already extrapolates).
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            seen_t, lift_t = self._last_seen_t, self._last_lift_t
            xy, side = self._last_xy, self._last_side
            vel = self.predictor.velocity() if self.predictor is not None else None
            predicted = self.predictor.predict(now) if self.predictor is not None else None
        in_view = seen_t is not None and (now - seen_t) <= self.view_grace
        fresh_xy = xy if (lift_t is not None and now - lift_t <= self.view_grace) else None
        if fresh_xy is not None and vel is not None and lead_gain:
            fresh_xy = (fresh_xy[0] + lead_gain * vel[0], fresh_xy[1] + lead_gain * vel[1])
        return HostEstimate(in_view=in_view, xy=fresh_xy, predicted=predicted, side=side)


def follow_target(
    ctx: TaskContext,
    sample,
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
    predict_on: bool | None = None,
) -> str:
    """Drive the base after a moving person until stopped, lost, or timed out.

    The reusable core of the follow behaviour. A :class:`HostTracker` samples
    the target's map-frame position flat out in its own thread (feeding a
    :class:`MotionPredictor` a dense trajectory); this control loop reads the
    freshest estimate and issues a nav goal every *nav_period* s.

    The control law is just "drive to where the target is going". Every tick it
    re-targets ``nav.go_to`` at the target's lifted map point LED forward by their
    velocity × *lead_gain* (``HRI_FOLLOW_LEAD_GAIN`` seconds), stopping
    *follow_distance* m short. Because nav orients the base along its travel
    direction, aiming at the target IS facing the target — no separate
    look-at/centering step — so the faster the tracker samples, the tighter it
    holds them. When they slip out of frame the predictor's extrapolation keeps
    the goal moving the way they were heading; only with no usable point does it
    fall back to rotating toward the last-seen side. Returns the exit reason:
    ``'stopped'`` | ``'lost'`` | ``'timeout'``.

    *sample* is a ``(ctx) -> (world_xy | None, side | None)`` callable the
    tracker polls each tick — e.g. ``_HostSampler().sample`` for identity-based
    host following, or :func:`nearest_person_sample` for an identity-free
    "follow whoever's closest". *stopper*, when given, is a context manager
    carrying a ``.placed`` :class:`threading.Event` (a :class:`CommandListener`);
    it is entered AFTER *on_warmup* — so a background mic listener never
    transcribes the warmup speech — and its event ends the loop the instant it
    is set. *on_warmup* runs once right after the tracker starts (e.g. a spoken
    ack, so the tracker warms up during the TTS); *on_lost* runs once the first
    time the target drops out of view; *on_stopped* runs after a
    *stopper*-triggered exit, before the threads are torn down, so a closing
    speech overlaps the join cost instead of adding a silent gap. Every tunable
    defaults from the ``HRI_FOLLOW_*`` env vars.
    """
    if follow_distance is None:
        follow_distance = float(os.getenv("HRI_FOLLOW_DISTANCE_M", "1.0"))
    if timeout is None:
        timeout = float(os.getenv("HRI_FOLLOW_TIMEOUT_SEC", "90"))
    if nav_period is None:
        nav_period = float(os.getenv("HRI_FOLLOW_NAV_PERIOD_SEC", "1.0"))
    if search_step_deg is None:
        search_step_deg = float(os.getenv("HRI_FOLLOW_SEARCH_STEP_DEG", "10"))
    search_step = math.radians(search_step_deg)
    if max_lost is None:
        max_lost = int(os.getenv("HRI_FOLLOW_MAX_LOST", "8"))
    if predict_on is None:
        predict_on = os.getenv("HRI_FOLLOW_PREDICT", "1").lower() in ("1", "true", "yes")

    # How far ahead (s) to aim along the target's velocity, so the goal leads
    # them instead of trailing. 0 = drive to where they are right now.
    lead_gain = float(os.getenv("HRI_FOLLOW_LEAD_GAIN", "0.5"))

    idle = threading.Event()  # never set — an interruptible sleep when no stopper
    lost = 0
    last_dir = 1.0  # default search direction (left) until first seen
    deadline = time.monotonic() + timeout
    reason = "timeout"
    with HostTracker(ctx, sample, predict_on=predict_on) as tracker:
        if on_warmup is not None:
            on_warmup()  # tracker warms up during this (e.g. a TTS ack)
        with (stopper if stopper is not None else nullcontext()) as listener:
            placed = getattr(listener, "placed", None)
            while time.monotonic() < deadline and not (placed is not None and placed.is_set()):
                est = tracker.read(lead_gain=lead_gain)
                if est.side is not None:
                    # Left of center (side < 0) -> turn left (+) to re-center.
                    last_dir = 1.0 if est.side < 0 else -1.0
                # Drive to where the target is going (in view: lifted point led by
                # velocity; just out of view: the predictor's extrapolation). nav
                # faces the travel direction, so this also faces the target.
                target = est.xy or est.predicted
                if target is not None:
                    lost = 0
                    approach_point(ctx, *target, stop_distance=follow_distance,
                                   blocking=False)
                else:
                    # No usable point at all: turn toward the last-seen side and
                    # keep turning until the target comes back into view.
                    lost += 1
                    if lost == 1 and on_lost is not None:
                        on_lost()  # nudge once, then keep searching
                    if lost >= max_lost:
                        print("[skills] follow_target: lost the target past the search budget")
                        reason = "lost"
                        break
                    rotate_by(ctx, last_dir * search_step)
                # Nav cadence — the "higher delay" between commands. The event
                # wait returns the instant the stopper flags a place command.
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


def nearest_person_sample(ctx: TaskContext) -> tuple[tuple[float, float] | None, float | None]:
    """A :func:`follow_target`/:class:`HostTracker` sampler that follows whoever
    is closest — pose only, NO identity or appearance recognition.

    Returns ``(world_xy | None, side | None)`` exactly like ``_HostSampler.sample``
    but with none of its two-tier recognition: just the biggest (so nearest)
    person box this frame, lifted to a map-frame point. *side* is the box's
    horizontal frame offset (``cx/width - 0.5``, negative = left), ``None`` only
    when no person was detected; *world_xy* may be ``None`` if the depth lift
    failed. Use to exercise the follow loop without enrolled identities.

    Set ``HRI_FOLLOW_TRACK_DEBUG=1`` to log the per-stage cost (snapshot / pose /
    lift) every sample, so you can see what is capping the tracker's frame rate.
    """
    debug = os.getenv("HRI_FOLLOW_TRACK_DEBUG", "0").lower() in ("1", "true", "yes")
    t0 = time.monotonic()
    snap = ctx.snapshot()
    t_snap = time.monotonic()
    if snap is None:
        return None, None
    img = snap.img
    box = nearest_person_bbox(ctx, img)
    t_pose = time.monotonic()
    if box is None:
        if debug:
            print(f"[nearest_person_sample] no-person "
                  f"snap={1e3 * (t_snap - t0):.0f}ms pose={1e3 * (t_pose - t_snap):.0f}ms")
        return None, None
    side = (box[0] + box[2]) / 2 / img.width - 0.5
    # Lift every tick (the follow loop drives to this point each cycle); skip the
    # full-frame edge filter to keep the lift cheap enough to run flat out.
    xy = lift_bbox_world_xy(ctx, snap, box, use_edge_filter=False)
    if debug:
        print(f"[nearest_person_sample] snap={1e3 * (t_snap - t0):.0f}ms "
              f"pose={1e3 * (t_pose - t_snap):.0f}ms side={side:+.2f} "
              f"lift={1e3 * (time.monotonic() - t_pose):.0f}ms")
    return xy, side


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
