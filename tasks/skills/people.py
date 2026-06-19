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
