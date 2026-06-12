"""Capture — one frame's detections, lifted segments, and pose correction.

The capture is the pipeline's first-class unit (ConceptGraphs-style): a
:class:`~services.walkie_graphs.camera_snapshot.CameraSnapshot` plus the
detector's segmentations, lifted into per-detection world-frame point
**segments** and a classless **background remainder** (every valid-depth pixel
under no mask). Pose error is a property of the capture — the robot's pose
estimate at that instant — not of individual objects, so registration happens
here, once per capture: :func:`register_capture` solves ONE rigid correction
against the existing map (anchored by the background) and applies it to every
segment. This replaces the old per-object ICP inside merges, whose flat-object
"slide" degeneracy needed a translation-cap workaround; with walls and floor
anchoring the solve, the same caps now just bound genuine pose error.

Object point clouds are then *derived from* segments: each kept detection's
segment persists in the :class:`CaptureStore` (one ``.npz`` per capture,
refcounted by the nodes that reference it), and a node's cloud is the fusion
of its segments — rebuildable at any time.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

try:  # optional, like geometry.py — only needed to resize/dilate masks
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from . import pcd_ops
from .geometry import deproject_mask


@dataclass(frozen=True)
class Segment:
    """One detection's lifted points, owned by one capture."""

    capture_id: str
    det_idx: int
    points: np.ndarray  # (N, 3) float32, world frame, post-correction

    @property
    def ref(self) -> str:
        return f"{self.capture_id}:{self.det_idx}"


def parse_ref(ref: str) -> tuple[str, int]:
    """Split a segment ref back into ``(capture_id, det_idx)``."""
    cid, idx = ref.rsplit(":", 1)
    return cid, int(idx)


@dataclass
class Capture:
    """One processed frame: segments + background + the solved pose correction."""

    id: str
    ts: float
    cam: object  # CameraPose | None — capture-time camera pose (provenance)
    segments: list[Segment] = field(default_factory=list)
    background: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float32)
    )
    correction: np.ndarray = field(default_factory=lambda: np.eye(4))
    icp_fitness: float = 0.0
    icp_accepted: bool = False


def new_capture_id(ts: float | None = None) -> str:
    return f"c{int(ts if ts is not None else time.time())}-{uuid.uuid4().hex[:6]}"


def _union_mask_at(masks, shape_hw) -> np.ndarray:
    """OR all detection masks together at the depth resolution."""
    h, w = shape_hw
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is None:
            continue
        mm = np.asarray(m)
        if mm.shape[:2] != (h, w):
            if cv2 is None:  # pragma: no cover — cv2 is a hard dep in practice
                continue
            mm = cv2.resize(
                mm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        union |= mm.astype(bool).astype(np.uint8)
    return union


def lift_capture(
    frame,
    masks,
    kept_points,
    *,
    capture_id: str | None = None,
    edge_mask: np.ndarray | None = None,
    bg_voxel_m: float = 0.05,
    bg_max_points: int = 60_000,
    bg_dilate_px: int = 4,
    bg_max_depth_m: float = 0.0,
) -> Capture:
    """Build a Capture from a frame's already-lifted detections + the remainder.

    ``masks`` is every detection's mask — kept, dropped, and **masking-only**
    (excluded classes like ``person``, detected purely so their pixels are
    carved OUT of the background instead of polluting it as ghost geometry).
    ``kept_points`` is ``[(det_idx, world_points)]`` for the graph-bound subset,
    lifted by the caller (no second deprojection here).

    The background remainder is the inverse of the dilated mask union: the
    dilation (``bg_dilate_px``) widens every mask because its rim is exactly
    where depth is least reliable, and a person's mask tends to underestimate
    their silhouette. Lifted at the coarse ``bg_voxel_m`` with the frame's
    flying-pixel edge filter; empty when the frame has no geometry.
    """
    cid = capture_id or new_capture_id(frame.ts)
    segments = [
        Segment(capture_id=cid, det_idx=int(i), points=np.asarray(pts, dtype=np.float32))
        for i, pts in kept_points
    ]
    background = np.zeros((0, 3), dtype=np.float32)
    if getattr(frame, "has_geometry", False):
        union = _union_mask_at(masks, frame.depth.shape[:2])
        if bg_dilate_px > 0 and cv2 is not None and union.any():
            union = cv2.dilate(
                union, np.ones((3, 3), np.uint8), iterations=int(bg_dilate_px)
            )
        background = deproject_mask(
            union == 0,
            frame.depth,
            frame.intr,
            frame.cam,
            voxel=bg_voxel_m,
            max_points=bg_max_points,
            erode_px=0,
            edge_mask=edge_mask,
            max_depth=bg_max_depth_m,
        )
    return Capture(id=cid, ts=frame.ts, cam=frame.cam, segments=segments, background=background)


def register_capture(
    capture: Capture,
    target_points: np.ndarray | None,
    *,
    max_corr_dist: float,
    min_fitness: float = 0.5,
    max_trans_m: float = 0.3,
    max_rot_deg: float = 5.0,
    src_budget: int = 20_000,
    min_points: int = 500,
) -> Capture:
    """Solve ONE rigid correction for the whole capture against the map.

    Source = background + all segments (the background dominates and anchors
    the solve — walls pin the translations a lone flat object can't). The
    correction is accepted only when ``fitness >= min_fitness`` AND it moves
    the capture by at most ``max_trans_m`` / ``max_rot_deg`` — anything larger
    than plausible pose error is a degenerate solve (e.g. a corridor sliding
    along itself), and the capture ingests raw, exactly as if ICP were off.
    On accept, every segment and the background are transformed in place.
    """
    if max_corr_dist <= 0:
        return capture
    parts = [capture.background] + [s.points for s in capture.segments]
    parts = [p for p in parts if len(p)]
    if not parts:
        return capture
    src = np.vstack(parts)
    if len(src) < min_points or target_points is None or len(target_points) < min_points:
        return capture
    T, fitness = pcd_ops.icp(
        pcd_ops.subsample(src, src_budget), target_points, max_corr_dist
    )
    capture.icp_fitness = fitness
    if fitness < min_fitness:
        return capture
    centroid = src.mean(axis=0, dtype=np.float64)
    shift = float(
        np.linalg.norm(pcd_ops.apply_transform(centroid[None, :], T)[0] - centroid)
    )
    if shift > max_trans_m or pcd_ops.rotation_angle_deg(T) > max_rot_deg:
        return capture
    capture.segments = [
        Segment(s.capture_id, s.det_idx, pcd_ops.apply_transform(s.points, T))
        for s in capture.segments
    ]
    capture.background = pcd_ops.apply_transform(capture.background, T)
    capture.correction = T @ capture.correction
    capture.icp_accepted = True
    return capture


class CaptureStore:
    """Per-capture segment persistence: one npz per capture, refcounted.

    Segments are written **post-correction** (world frame), so rebuilding an
    object's cloud is a plain concat — no transforms to replay. Writes are
    deferred (queued by :meth:`save`, written by :meth:`flush` on the service's
    flush cadence, mirroring ``GraphMemory.flush_pcds``); a small write-through
    cache serves reads for pending and recently-used segments. Nodes
    :meth:`retain`/:meth:`release` the refs they hold; :meth:`gc` unlinks
    capture files no live segment references (crash orphans included, once the
    loaded graph has re-retained its refs at startup).
    """

    _CACHE_MAX = 1024  # segments; bounds worst-case cache at ~25 MB

    def __init__(self, captures_dir: str) -> None:
        self.dir = captures_dir
        os.makedirs(self.dir, exist_ok=True)
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, np.ndarray]] = {}  # capture_id -> arrays
        self._cache: dict[str, np.ndarray] = {}  # segment ref -> points (FIFO)
        self._refs: dict[str, int] = {}  # capture_id -> live segment count

    @classmethod
    def from_env(cls) -> "CaptureStore":
        return cls(os.getenv("WALKIE_GRAPHS_CAPTURES_DIR", "graph_captures"))

    def _path(self, capture_id: str) -> str:
        return os.path.join(self.dir, f"{capture_id}.npz")

    def _cache_put(self, ref: str, points: np.ndarray) -> None:
        self._cache[ref] = points
        while len(self._cache) > self._CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))

    def save(self, capture: Capture) -> None:
        """Queue a capture's segments for the next flush (skips empty captures)."""
        if not capture.segments:
            return
        arrays: dict[str, np.ndarray] = {
            f"seg_{s.det_idx}": np.asarray(s.points, dtype=np.float32)
            for s in capture.segments
        }
        arrays["T"] = np.asarray(capture.correction, dtype=np.float64)
        arrays["fitness"] = np.float64(capture.icp_fitness)
        with self._lock:
            self._pending[capture.id] = arrays
            for s in capture.segments:
                self._cache_put(s.ref, arrays[f"seg_{s.det_idx}"])

    def flush(self) -> int:
        """Write all queued captures to disk (atomic per file); returns the count."""
        with self._lock:
            items = list(self._pending.items())
            self._pending = {}
        for cid, arrays in items:
            path = self._path(cid)
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
        return len(items)

    def load_segment(self, ref: str) -> np.ndarray | None:
        """A segment's points, from cache, the pending queue, or disk."""
        cid, idx = parse_ref(ref)
        with self._lock:
            hit = self._cache.get(ref)
            if hit is not None:
                return hit
            pending = self._pending.get(cid)
            if pending is not None:
                return pending.get(f"seg_{idx}")
        try:
            with np.load(self._path(cid)) as data:
                pts = np.asarray(data[f"seg_{idx}"], dtype=np.float32)
        except Exception:  # noqa: BLE001 — missing file/key = segment gone
            return None
        with self._lock:
            self._cache_put(ref, pts)
        return pts

    # ------------------------------------------------------------------
    # Reference counting + GC
    # ------------------------------------------------------------------
    def retain(self, ref: str | None) -> None:
        if not ref:
            return
        cid, _ = parse_ref(ref)
        with self._lock:
            self._refs[cid] = self._refs.get(cid, 0) + 1

    def release(self, ref: str | None) -> None:
        if not ref:
            return
        cid, _ = parse_ref(ref)
        with self._lock:
            n = self._refs.get(cid, 0) - 1
            if n > 0:
                self._refs[cid] = n
            else:
                self._refs.pop(cid, None)

    def gc(self) -> int:
        """Unlink capture files (and pending writes) with no live references."""
        removed = 0
        with self._lock:
            live = {cid for cid, n in self._refs.items() if n > 0}
            self._pending = {c: a for c, a in self._pending.items() if c in live}
            try:
                names = [n for n in os.listdir(self.dir) if n.endswith(".npz")]
            except OSError:
                names = []
            dead = [n[: -len(".npz")] for n in names if n[: -len(".npz")] not in live]
            for cid in dead:
                for ref in [r for r in self._cache if r.startswith(f"{cid}:")]:
                    self._cache.pop(ref, None)
        for cid in dead:
            try:
                os.remove(self._path(cid))
                removed += 1
            except OSError:
                pass
        return removed
