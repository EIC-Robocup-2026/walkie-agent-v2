"""Person / face detection primitives and trackers.

Moved out of tasks/HRI/skills.py into the shared tasks.skills package.
"""

from __future__ import annotations

import math
import os
import threading
import time

from PIL import Image

from client import FaceEmbedding, PersonPose
from tasks.base import TaskContext

from .geometry import BBox, cxcywh_to_xyxy


def person_bboxes(ctx: TaskContext, img: Image.Image) -> list[BBox]:
    """All detected person boxes (xyxy) in *img* via pose estimation; [] on failure.

    This is the *fast*, real-time path: pose estimation only, with NO face or
    attire recognition. Those per-person embeddings (run by ``locate_people``)
    are the expensive part — keeping them out of the per-tick loop is what lets
    the tracker sample at pose-estimation rate.
    """
    try:
        persons = ctx.walkieAI.image.estimate_poses(img)
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
        faces = ctx.walkieAI.image.faces(img)
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


def _wrap(angle: float) -> float:
    """Wrap radians to ``(-pi, pi]`` (shortest signed heading difference)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class FaceTracker:
    """Background base-rotation loop that keeps Walkie facing the nearest face.

    Runs TWO daemon threads (started on ``start`` / ``__enter__``, joined on
    ``stop`` / ``__exit__``):

    * a **detection thread** (producer) — captures frames back-to-back (the
      detector round-trip paces it; there is no fixed tick interval), finds the
      biggest face above ``HRI_FACE_MIN_AREA_PX``, and turns it into an ABSOLUTE
      map-frame target heading (a setpoint). Crucially, the robot heading is
      sampled at the *capture instant* — before the slow detector round-trip — so
      the setpoint isn't skewed by whatever rotation happened during detection.
    * a **control thread** (consumer) — a fast proportional loop at
      ``HRI_FACE_TRACK_HZ`` that reads the LIVE heading, computes the error to the
      latest setpoint, and publishes a yaw velocity via ``nav.set_velocity``
      (``wz = kp * error``, clamped to ``[MIN_WZ, MAX_WZ]``, zero inside
      ``DEADBAND_DEG``).

    Because the setpoint is an absolute heading and the loop closes on live
    odometry, it is self-correcting — as the base turns toward it the error
    shrinks to zero, so there's no overshoot from a stale pixel offset — and the
    base keeps getting commands fast enough not to trip the ``cmd_vel`` watchdog
    during a slow detection. A stale setpoint (no face for
    ``HRI_FACE_TRACK_LOST_SEC``) stops the base rather than spinning blind.

    The setpoint is derived depth-free from the face's pixel column + capture
    heading + the camera HFOV (``HRI_CAMERA_HFOV_DEG``). Set
    ``HRI_FACE_TRACK_LIFT=1`` to instead deproject the face bbox to a map point
    and aim at that (more accurate bearing, at the cost of a full depth+TF
    snapshot per detection).

    Only rotates the BASE (the head servo just tilts), so use it only while
    nothing else drives the base. Every camera / detector / nav fault is
    swallowed so a glitch never crashes the calling step. Use as a context
    manager::

        with FaceTracker(ctx):
            ... ask the guest their name, caption their appearance ...
    """

    def __init__(self, ctx: TaskContext) -> None:
        self.ctx = ctx
        self._stop = threading.Event()
        self._detect_thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._target: tuple[float, float] | None = None  # (bearing_rad, monotonic_ts)

        self.min_area = float(os.getenv("HRI_FACE_MIN_AREA_PX", "10000"))
        self.hfov = math.radians(float(os.getenv("HRI_CAMERA_HFOV_DEG", "110")))
        self.lift = os.getenv("HRI_FACE_TRACK_LIFT", "0").strip().lower() in ("1", "true", "yes")
        # Control loop: proportional yaw with a deadband and a min/max speed clamp
        # (the min floor overcomes base stiction so a tiny error still centers).
        self.hz = max(1.0, float(os.getenv("HRI_FACE_TRACK_HZ", "15")))
        self.kp = float(os.getenv("HRI_FACE_TRACK_KP", "1.2"))
        self.max_wz = float(os.getenv("HRI_FACE_TRACK_MAX_WZ", "0.6"))
        self.min_wz = float(os.getenv("HRI_FACE_TRACK_MIN_WZ", "0.15"))
        self.deadband = math.radians(float(os.getenv("HRI_FACE_TRACK_DEADBAND_DEG", "6")))
        self.lost_sec = float(os.getenv("HRI_FACE_TRACK_LOST_SEC", "1.0"))
        # Brief idle when a detection cycle finds nothing (avoids a hot spin); NOT a
        # tracking interval — a successful detect loops straight back with no delay.
        self.detect_idle = float(os.getenv("HRI_FACE_TRACK_DETECT_IDLE_SEC", "0.1"))

    # -- lifecycle -------------------------------------------------------
    def start(self) -> "FaceTracker":
        nav = getattr(self.ctx.walkie, "nav", None)
        if nav is None or not hasattr(nav, "set_velocity"):
            print("[skills] FaceTracker: nav.set_velocity unavailable; base tracking disabled")
            return self
        if self._detect_thread is None:
            self._stop.clear()
            self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._detect_thread.start()
            self._control_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in (self._detect_thread, self._control_thread):
            if t is not None:
                t.join(timeout=3.0)
        self._detect_thread = self._control_thread = None

    def __enter__(self) -> "FaceTracker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- detection thread (producer) -------------------------------------
    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                bearing = self._detect_target_bearing()
            except Exception as exc:  # noqa: BLE001 — a glitch must not kill the loop
                print(f"[skills] face tracker detect failed ({exc})")
                bearing = None
            if bearing is not None:
                with self._lock:
                    self._target = (bearing, time.monotonic())
                continue  # the detector round-trip IS the pacing — no artificial delay
            self._stop.wait(self.detect_idle)  # nothing found: brief idle, don't hot-spin

    def _detect_target_bearing(self) -> float | None:
        """Absolute map-frame heading toward the biggest face, or None.

        Samples the robot heading at the CAPTURE instant (before the detector
        round-trip) so the setpoint isn't skewed by rotation during detection.
        """
        if self.lift:
            snap = self.ctx.snapshot()
            if snap is None:
                return None
            face = biggest_face(self.ctx, snap.img, min_area=self.min_area)
            if face is None:
                return None
            return self._bearing_from_snapshot(snap, face)
        # Light (default) path: grab the frame + heading back-to-back (both fast
        # local reads), THEN run the slow detector — so `heading` is the capture pose.
        img = self.ctx.capture()
        if img is None:
            return None
        heading = self.ctx.current_pose()["heading"]
        face = biggest_face(self.ctx, img, min_area=self.min_area)
        if face is None:
            return None
        return self._bearing_from_pixel(face, img.width, heading)

    def _bearing_from_pixel(self, face, width: int, heading: float) -> float:
        """Absolute heading to a face at its bbox column, from the capture heading
        and the camera HFOV (depth-free).

        A column LEFT of centre (``face_cx < width/2``) needs a LARGER, CCW heading
        to face it, so ``bearing = heading + (½ - face_cx/width) * hfov`` — the same
        sign convention the old pixel-offset tick used.
        """
        x1, _y1, x2, _y2 = face.bbox_xyxy
        face_cx = (x1 + x2) / 2.0
        off = (width / 2.0 - face_cx) / max(1.0, float(width)) * self.hfov
        return _wrap(heading + off)

    def _bearing_from_snapshot(self, snap, face) -> float | None:
        """Deproject the face bbox to a map point and return the heading to it;
        fall back to the pixel bearing when depth/pose is unavailable."""
        rp = getattr(snap, "robot_pose", None) or {}
        try:
            xy = snap.bbox_world_xy(face.bbox_xyxy)
        except Exception:  # noqa: BLE001 — degrade to the pixel bearing below
            xy = None
        if xy is not None and "x" in rp and "y" in rp:
            return _wrap(math.atan2(xy[1] - rp["y"], xy[0] - rp["x"]))
        heading = rp.get("heading")
        if heading is None:
            return None
        return self._bearing_from_pixel(face, snap.img.width, heading)

    # -- control thread (consumer) ---------------------------------------
    def _control_loop(self) -> None:
        nav = self.ctx.walkie.nav
        dt = 1.0 / self.hz
        try:
            while not self._stop.is_set():
                try:
                    nav.set_velocity(0.0, 0.0, self._yaw_command())
                except Exception as exc:  # noqa: BLE001
                    print(f"[skills] face tracker set_velocity failed ({exc})")
                self._stop.wait(dt)
        finally:
            for _ in range(3):  # guaranteed zero-velocity stop on any exit
                try:
                    nav.set_velocity(0.0, 0.0, 0.0)
                except Exception:  # noqa: BLE001
                    break
            try:
                nav.stop()
            except Exception:  # noqa: BLE001
                pass

    def _yaw_command(self) -> float:
        """Proportional yaw rate toward the latest setpoint; 0 when centered/stale.

        Closes on the LIVE heading, so as the base turns the error shrinks and the
        command self-terminates inside the deadband — no stale-offset overshoot."""
        with self._lock:
            target = self._target
        if target is None or time.monotonic() - target[1] > self.lost_sec:
            return 0.0  # no fresh face: hold still rather than spin blind
        error = _wrap(target[0] - self.ctx.current_pose()["heading"])
        if abs(error) <= self.deadband:
            return 0.0
        mag = min(self.max_wz, max(self.min_wz, abs(self.kp * error)))
        return math.copysign(mag, error)


# COCO keypoint indices (same convention as agents/vision_agent/tools.py).
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6


_LEFT_WRIST, _RIGHT_WRIST = 9, 10


def is_calling_gesture(pose: PersonPose, conf_thresh: float = 0.3) -> bool:
    """True when either wrist is raised above the same-side shoulder.

    Pure function (no ctx/network) so it is unit-testable. Image y grows
    downward, so "raised" means ``wrist.y < shoulder.y``. Mirrors the
    arm-raised heuristic in agents/vision_agent/tools.py.
    """
    kpts = {kp.index: kp for kp in pose.keypoints}
    ls, lw = kpts.get(_LEFT_SHOULDER), kpts.get(_LEFT_WRIST)
    rs, rw = kpts.get(_RIGHT_SHOULDER), kpts.get(_RIGHT_WRIST)
    left = ls and lw and lw.confidence > conf_thresh and ls.confidence > conf_thresh and lw.y < ls.y
    right = rs and rw and rw.confidence > conf_thresh and rs.confidence > conf_thresh and rw.y < rs.y
    return bool(left or right)
