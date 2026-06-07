"""WalkieGraphsService — the background thread that grows the scene graph.

Mirrors :class:`services.perception.PerceptionService`: a daemon thread that, each
tick, reads the robot pose + lift + head tilt, captures an RGB-D frame, runs
masked object detection (scoped to the *interested classes*), lifts each mask to a
3D world point cloud, embeds + captions the crop via the AI client, and upserts it
into :class:`~walkie_graphs.memory.GraphMemory`. Every few ticks it recomputes the
geometric relations, prunes, and pushes to the visualizer.

Detection/caption/embedding are all direct ``walkieAI`` client calls — there is no
local model here.
"""

from __future__ import annotations

import os
import threading
import time

from .geometry import Intrinsics, compute_camera_pose, deproject_mask
from .memory import Detection3D, GraphMemory

_CROP_MARGIN_PX = 10


def _csv(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _opt_float(name: str):
    v = os.getenv(name, "")
    return float(v) if v.strip() else None


def _vec3(value: str) -> tuple[float, float, float]:
    parts = [float(x) for x in value.split(",")]
    return (parts[0], parts[1], parts[2])


def _crop_pil(img, bbox):
    """Crop a PIL image to ``bbox`` (x1,y1,x2,y2) with a small margin, clamped."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - _CROP_MARGIN_PX)
    y1 = max(0, int(y1) - _CROP_MARGIN_PX)
    x2 = min(w, int(x2) + _CROP_MARGIN_PX)
    y2 = min(h, int(y2) + _CROP_MARGIN_PX)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


class WalkieGraphsService(threading.Thread):
    """Background observer that drives ``GraphMemory`` from live RGB-D frames."""

    def __init__(
        self,
        walkieAI,
        walkie,
        memory: GraphMemory,
        *,
        model=None,
        viz=None,
        verbose: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="WalkieGraphsService")
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.memory = memory
        self.model = model  # currently unused (server captions + geometric edges)
        self.viz = viz
        self.verbose = verbose
        self._stop_event = threading.Event()

        self.interval = float(os.getenv("WALKIE_GRAPHS_INTERVAL_SEC", "3.0"))
        self.interested = _csv(os.getenv("WALKIE_GRAPHS_INTERESTED_CLASSES", ""))
        self._interested_lower = {c.lower() for c in self.interested}
        self.exclude = {c.lower() for c in _csv(os.getenv("WALKIE_GRAPHS_EXCLUDE_CLASSES", "person"))}
        # Empty caption list => fall back to the interested list; empty both => caption all.
        cap = _csv(os.getenv("WALKIE_GRAPHS_CAPTION_CLASSES", "")) or self.interested
        self.caption_classes = {c.lower() for c in cap}

        self.min_points = int(os.getenv("WALKIE_GRAPHS_MIN_POINTS", "30"))
        self.voxel_m = float(os.getenv("WALKIE_GRAPHS_VOXEL_M", "0.02"))
        self.max_points = int(os.getenv("WALKIE_GRAPHS_MAX_POINTS_PER_OBJ", "2000"))
        self.relation_every_n = int(os.getenv("WALKIE_GRAPHS_RELATION_EVERY_N", "5"))

        self._hfov = float(os.getenv("WALKIE_GRAPHS_HFOV_DEG", "110"))
        self._fx, self._fy = _opt_float("WALKIE_GRAPHS_FX"), _opt_float("WALKIE_GRAPHS_FY")
        self._cx, self._cy = _opt_float("WALKIE_GRAPHS_CX"), _opt_float("WALKIE_GRAPHS_CY")
        self._lift_to_head = _vec3(os.getenv("WALKIE_GRAPHS_LIFT_TO_HEAD", "0.265,0.0,0.422"))
        self._pivot_to_optic = _vec3(os.getenv("WALKIE_GRAPHS_PIVOT_TO_OPTIC", "0.065,0.0,0.0"))
        # head.get_angle() is radians; geometry wants "positive = camera tilts down".
        # If the joint-state sign is inverted vs that, set SIGN=-1; OFFSET corrects a
        # non-level zero. effective_tilt = sign * get_angle() + offset.
        self._tilt_sign = float(os.getenv("WALKIE_GRAPHS_HEAD_TILT_SIGN", "1"))
        self._tilt_offset = float(os.getenv("WALKIE_GRAPHS_HEAD_TILT_OFFSET_RAD", "0"))
        self._intr_cache: dict[tuple[int, int], Intrinsics] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def stop_and_join(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[graphs] {msg}")

    def run(self) -> None:
        self._log(f"started (interval={self.interval}s)")
        tick = 0
        while not self._stop_event.is_set():
            try:
                touched = self._observe_once()
                tick += 1
                if self.relation_every_n > 0 and tick % self.relation_every_n == 0:
                    self.memory.derive_relations()
                    self.memory.prune()
                if self.viz is not None and touched is not None:
                    try:
                        self.viz.update(self.memory, robot_pose=self.walkie.status.get_position())
                    except Exception as e:  # noqa: BLE001
                        self._log(f"viz update failed: {e}")
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill the thread
                self._log(f"tick error: {e}")
            self._stop_event.wait(self.interval)
        self._log("stopped.")

    # ------------------------------------------------------------------
    # One observation
    # ------------------------------------------------------------------
    def _observe_once(self) -> list:
        """Capture one RGB-D frame and fold every kept detection into the graph."""
        depth = self._depth()
        if depth is None:
            return []
        try:
            img = self.walkie.camera.capture_pil()  # PIL RGB
        except Exception as e:  # noqa: BLE001
            self._log(f"capture failed: {e}")
            return []

        intr = self._intrinsics(depth.shape[1], depth.shape[0])
        pose = self.walkie.status.get_position() or {"x": 0.0, "y": 0.0, "heading": 0.0}
        cam = compute_camera_pose(
            float(pose.get("x", 0.0)),
            float(pose.get("y", 0.0)),
            float(pose.get("heading", 0.0)),
            self._lift_cm(),
            self._tilt_rad(),
            lift_to_head=self._lift_to_head,
            pivot_to_optic=self._pivot_to_optic,
        )

        detections = self.walkieAI.object_detection.detect(
            img, prompts=self.interested or None, return_mask=True
        )

        pending = []  # (detected, points, crop)
        for d in detections:
            if not self._keep(d.class_name or ""):
                continue
            if d.mask is None:
                continue
            pts = deproject_mask(
                d.mask, depth, intr, cam, voxel=self.voxel_m, max_points=self.max_points
            )
            if len(pts) < self.min_points:
                continue
            pending.append((d, pts, _crop_pil(img, d.bbox)))

        captions = self._caption([p[0] for p in pending], [p[2] for p in pending])

        touched = []
        for i, (d, pts, crop) in enumerate(pending):
            det = Detection3D(
                class_name=d.class_name or "object",
                class_id=d.class_id,
                confidence=float(d.confidence or 0.0),
                bbox_xyxy=tuple(int(v) for v in d.bbox),
                points_world=pts,
                clip_emb=self._embed(crop),
                caption=captions.get(i, ""),
                ts=time.time(),
                crop=crop,
            )
            touched.append(self.memory.upsert(det))
        return touched

    # ------------------------------------------------------------------
    # Per-class scoping
    # ------------------------------------------------------------------
    def _keep(self, class_name: str) -> bool:
        c = class_name.lower()
        if c in self.exclude:
            return False
        if self._interested_lower and c not in self._interested_lower:
            return False
        return True

    def _should_caption(self, class_name: str) -> bool:
        return not self.caption_classes or class_name.lower() in self.caption_classes

    def _caption(self, detected: list, crops: list) -> dict[int, str]:
        idx = [i for i, d in enumerate(detected) if self._should_caption(d.class_name or "")]
        if not idx:
            return {}
        imgs = [crops[i] for i in idx]
        prompts = [f"Describe the {detected[i].class_name}." for i in idx]
        try:
            out = self.walkieAI.image_caption.caption_batch(imgs, prompts=prompts)
        except Exception as e:  # noqa: BLE001
            self._log(f"caption failed: {e}")
            return {}
        return {i: (c or "") for i, c in zip(idx, out)}

    def _embed(self, crop) -> list[float]:
        try:
            return list(self.walkieAI.image_embed.embed_image(crop))
        except Exception as e:  # noqa: BLE001 — embed route may be disabled server-side
            self._log(f"embed failed (degrading to spatial dedup): {e}")
            return []

    # ------------------------------------------------------------------
    # Sensor reads
    # ------------------------------------------------------------------
    def _intrinsics(self, width: int, height: int) -> Intrinsics:
        key = (width, height)
        if key not in self._intr_cache:
            self._intr_cache[key] = Intrinsics.from_hfov(
                width, height, self._hfov, fx=self._fx, fy=self._fy, cx=self._cx, cy=self._cy
            )
        return self._intr_cache[key]

    def _depth(self):
        try:
            return self.walkie.robot.camera.get_depth()
        except Exception as e:  # noqa: BLE001
            self._log(f"depth unavailable: {e}")
            return None

    def _lift_cm(self) -> float:
        try:
            v = self.walkie.robot.lift.get(norm_pos=False)
            return float(v) if v is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _tilt_rad(self) -> float:
        try:
            v = self.walkie.robot.head.get_angle()
            v = float(v) if v is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0
        return self._tilt_sign * v + self._tilt_offset
