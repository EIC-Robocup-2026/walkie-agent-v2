"""WalkieGraphsService — the robot's perception loop and scene-graph builder.

A daemon thread that, each tick, captures an RGB-D frame *with* its pose and
intrinsics (one :class:`CameraSnapshot`), runs masked object detection, and folds
the frame into the graph **as one capture**:

1. every kept mask lifts to a world-frame point *segment*, and the unmasked
   remainder lifts to a classless *background* cloud (walls, floor — carved
   around ALL masks, including masking-only excluded classes like ``person``);
2. the whole capture is registered against the existing map with ONE rigid ICP
   correction (``capture.register_capture``) — pose error belongs to the
   capture, not to individual objects;
3. corrected segments are captioned/embedded via the AI client and upserted
   into :class:`~services.walkie_graphs.memory.GraphMemory` (which persists the
   segments through the :class:`~services.walkie_graphs.capture.CaptureStore`),
   and the background remainder accumulates in the ``BackgroundStore``.

When a ``snapshot_path`` is configured the live ``perception.json`` view of the
frame is then written (the agents' "what's in front of me now" context). Every
few ticks it recomputes the geometric relations, prunes, and pushes to the
visualizer.

Detection/caption/embedding are all direct ``walkieAI`` client calls — there is no
local model here. Pose/people detection is intentionally not part of this loop.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from interfaces.devices.camera import CameraSnapshot, camera_pose

from .capture import lift_capture, register_capture
from .fusion import subtract_contained_masks
from .geometry import depth_discontinuity_mask, deproject_mask
from .memory import Detection3D, GraphMemory
from .snapshot import build_object_records, write_atomic

# Back-compat: the per-tick frame snapshot is now the shared CameraSnapshot
# (interfaces.devices.camera); old name kept for existing callers.
FrameSnapshot = CameraSnapshot


def _csv(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def _crop_pil(img, bbox, margin: int = 20):
    """Crop a PIL image to ``bbox`` (x1,y1,x2,y2) with a clamped margin.

    The margin gives CLIP/captioning some surrounding context (ConceptGraphs pads
    its feature crops by 20 px for the same reason).
    """
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(w, int(x2) + margin)
    y2 = min(h, int(y2) + margin)
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
        snapshot_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="WalkieGraphsService")
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.memory = memory
        self.model = model  # currently unused (server captions + geometric edges)
        self.viz = viz
        # Where the loop writes the live perception.json snapshot each tick (the agents'
        # "what's in front of me now" view). None => don't write a snapshot — keeps the
        # manual observe() path side-effect-free; main.py passes the real path in production.
        self.snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self.verbose = verbose
        self._stop_event = threading.Event()

        self.interval = float(os.getenv("WALKIE_GRAPHS_INTERVAL_SEC", "3.0"))
        self.interested = _csv(os.getenv("WALKIE_GRAPHS_INTERESTED_CLASSES", ""))
        self._interested_lower = {c.lower() for c in self.interested}
        self.exclude = {c.lower() for c in _csv(os.getenv("WALKIE_GRAPHS_EXCLUDE_CLASSES", "person"))}
        # Excluded classes are still PROMPTED for, in a masking-only role: a person
        # we'd never map must still get a mask, so their pixels are carved out of the
        # background cloud instead of lingering as ghost geometry. Only meaningful
        # when detection is scoped (an empty interested list already detects all).
        extra = sorted(c for c in self.exclude if c not in self._interested_lower)
        self.detect_prompts = (self.interested + extra) if self.interested else []
        # Empty caption list => fall back to the interested list; empty both => caption all.
        cap = _csv(os.getenv("WALKIE_GRAPHS_CAPTION_CLASSES", "")) or self.interested
        self.caption_classes = {c.lower() for c in cap}

        self.min_confidence = float(os.getenv("WALKIE_GRAPHS_CONFIDENCE_THRESHOLD", "0.0"))
        self.min_points = int(os.getenv("WALKIE_GRAPHS_MIN_POINTS", "30"))
        self.voxel_m = float(os.getenv("WALKIE_GRAPHS_VOXEL_M", "0.02"))
        self.max_points = int(os.getenv("WALKIE_GRAPHS_MAX_POINTS_PER_OBJ", "2000"))
        self.relation_every_n = int(os.getenv("WALKIE_GRAPHS_RELATION_EVERY_N", "5"))
        # Depth "flying pixel" cleanup (the shadow trailing off an object's silhouette):
        # erode each mask inward by this many pixels, and drop pixels sitting on a depth
        # jump larger than this many metres. 0 / 0.0 disable each.
        self.mask_erode_px = int(os.getenv("WALKIE_GRAPHS_MASK_ERODE_PX", "2"))
        self.depth_edge_thresh_m = float(os.getenv("WALKIE_GRAPHS_DEPTH_EDGE_THRESH_M", "0.05"))
        # Make the depth-edge threshold grow with distance (thresh + rel * depth): a fixed
        # threshold erases grazing surfaces (a bed/table seen edge-on keeps only its near
        # corner). 0.0 = original fixed behaviour. See depth_discontinuity_mask.
        self.depth_edge_rel = float(os.getenv("WALKIE_GRAPHS_DEPTH_EDGE_REL", "0.0"))
        # Statistical outlier removal on each lifted cloud (Open3D remove_statistical_outlier):
        # a 3D density filter that strips flying pixels the per-pixel edge filter can't, and
        # unlike a fixed depth-edge threshold never erases grazing surfaces. 0 disables.
        # Default off in code (tests), on via config.toml — the usual split.
        self.sor_k = int(os.getenv("WALKIE_GRAPHS_SOR_K", "0"))
        self.sor_std_ratio = float(os.getenv("WALKIE_GRAPHS_SOR_STD_RATIO", "2.0"))
        # Trusted sensor range (ZED2i): depth error grows ~quadratically with
        # distance, so pixels beyond this are dropped at deprojection — far-object
        # artifacts and silhouette bleeding never enter the map. 0 = unbounded.
        self.max_depth_m = float(os.getenv("WALKIE_GRAPHS_MAX_DEPTH_M", "0"))
        self.bg_max_depth_m = float(
            os.getenv("WALKIE_GRAPHS_BG_MAX_DEPTH_M", str(self.max_depth_m))
        )
        # Detection-time filters (ConceptGraphs filter_gobs): reject whole-frame /
        # background boxes and degenerate masks before they cost a deproject. 1.0 / 0
        # are no-ops (keep everything); config.toml tightens them.
        self.max_bbox_area_ratio = float(os.getenv("WALKIE_GRAPHS_MAX_BBOX_AREA_RATIO", "1.0"))
        self.min_mask_area_px = int(os.getenv("WALKIE_GRAPHS_MIN_MASK_AREA_PX", "0"))
        # Subtract a contained detection's mask from its container's (CG
        # mask_subtract_contained): keeps the mug's pixels out of the table's cloud/crop.
        self.mask_subtract = os.getenv("WALKIE_GRAPHS_MASK_SUBTRACT", "1").strip().lower() in (
            "1", "true", "yes",
        )
        # Context margin around the bbox for the CLIP/caption crop (CG pads 20 px).
        self.crop_margin_px = int(os.getenv("WALKIE_GRAPHS_CROP_MARGIN_PX", "20"))
        # Capture-level registration: ONE rigid ICP correction per frame against the
        # map (background + nearby objects), replacing all per-object ICP. 0 = off
        # (code default, so unit tests never need open3d); config.toml enables it.
        # The trans/rot caps bound a correction to plausible pose error — anything
        # bigger is a degenerate solve (corridor sliding along itself) and the frame
        # ingests raw instead.
        self.capture_icp_max_corr_m = float(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_MAX_CORR_M", "0"))
        self.capture_icp_min_fitness = float(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_MIN_FITNESS", "0.5"))
        self.capture_icp_max_trans_m = float(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_MAX_TRANS_M", "0.3"))
        self.capture_icp_max_rot_deg = float(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_MAX_ROT_DEG", "5"))
        self.capture_icp_src_budget = int(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_SRC_BUDGET", "20000"))
        self.capture_icp_tgt_budget = int(os.getenv("WALKIE_GRAPHS_CAPTURE_ICP_TGT_BUDGET", "40000"))
        # Background remainder: lift resolution, mask-union dilation (mask rims are
        # exactly where depth is least reliable), and the store's save cadence.
        self.bg_voxel_m = float(os.getenv("WALKIE_GRAPHS_BG_VOXEL_M", "0.05"))
        self.bg_mask_dilate_px = int(os.getenv("WALKIE_GRAPHS_BG_MASK_DILATE_PX", "4"))
        self.bg_save_every_n = int(os.getenv("WALKIE_GRAPHS_BG_SAVE_EVERY_N", "20"))
        # Concurrency for the per-object CLIP embed calls (no batch endpoint exists, so
        # fan the I/O-bound round-trips out across a thread pool). 1 = sequential.
        self._embed_workers = int(os.getenv("WALKIE_GRAPHS_EMBED_WORKERS", "8"))
        # Periodic-maintenance cadences (in ingest ticks), staggered so two heavy
        # passes never land on the same tick. 0 disables a pass.
        self.denoise_every_n = int(os.getenv("WALKIE_GRAPHS_DENOISE_EVERY_N", "20"))
        self.merge_every_n = int(os.getenv("WALKIE_GRAPHS_MERGE_EVERY_N", "20"))
        self.ghost_every_n = int(os.getenv("WALKIE_GRAPHS_GHOST_EVERY_N", "20"))
        # Deferred point-cloud persistence: flush pending .npz writes every N ticks
        # (memory.defer_pcd_writes batches them; reads always come from the cache).
        self.pcd_flush_every_n = int(os.getenv("WALKIE_GRAPHS_PCD_FLUSH_EVERY_N", "5"))
        # Per-object ICP refine pass: co-register a node's segments to fix smear from
        # captures whose registration was rejected (code default 0 = off; config.toml
        # enables it). refine_limit caps how many nodes are processed per tick.
        self.refine_every_n = int(os.getenv("WALKIE_GRAPHS_REFINE_EVERY_N", "0"))
        self.refine_limit = int(os.getenv("WALKIE_GRAPHS_REFINE_LIMIT", "3"))
        # Free-space carving: remove map geometry a trusted capture sees straight
        # through (moved-object ghosts, lateral edge-shadow trails). 0 = off (code
        # default). Only runs on an accepted/registration-off capture (the gate in
        # _maybe_tick) — never a rejected/cold solve, which would carve a mis-posed hole.
        self.carve_every_n = int(os.getenv("WALKIE_GRAPHS_CARVE_EVERY_N", "0"))
        self.carve_margin_base = float(os.getenv("WALKIE_GRAPHS_CARVE_MARGIN_BASE_M", "0.05"))
        self.carve_margin_rel = float(os.getenv("WALKIE_GRAPHS_CARVE_MARGIN_REL", "0.02"))
        self.carve_z_min = float(os.getenv("WALKIE_GRAPHS_CARVE_Z_MIN_M", "0.05"))
        self.carve_max_z = float(
            os.getenv("WALKIE_GRAPHS_CARVE_MAX_Z_M", str(self.max_depth_m or 4.0))
        )
        self.carve_evict_min_points = int(os.getenv("WALKIE_GRAPHS_CARVE_EVICT_MIN_POINTS", "20"))
        self.carve_refine_frac = float(os.getenv("WALKIE_GRAPHS_CARVE_REFINE_FRAC", "0.5"))
        # Tier 3 (optional LLM): caption refinement + LLM edge inference. 0 = off, and
        # both require self.model. They only ever run on these cadences when enabled.
        self.caption_refine_every_n = int(os.getenv("WALKIE_GRAPHS_CAPTION_REFINE_EVERY_N", "0"))
        self.caption_refine_limit = int(os.getenv("WALKIE_GRAPHS_CAPTION_REFINE_LIMIT", "8"))
        self.caption_refine_use_images = os.getenv(
            "WALKIE_GRAPHS_CAPTION_REFINE_USE_IMAGES", "0"
        ).strip().lower() in ("1", "true", "yes")
        self.llm_edges_every_n = int(os.getenv("WALKIE_GRAPHS_LLM_EDGES_EVERY_N", "0"))

        # Camera calibration + pose come from interfaces.devices.camera (CameraSnapshot.capture
        # reads them straight from the walkie-sdk, configured by WALKIE_GRAPHS_TF_*).
        # Motion gate: a frame captured while the robot/head moves carries a smeared,
        # mis-posed cloud (the streamed depth + TF can't be perfectly synchronized), so
        # don't fold it into the graph. The camera pose is sampled again after detection
        # and compared with the capture-time pose — the detection round-trip provides the
        # bracketing window for free. 0 disables either bound.
        self.motion_max_trans_m = float(os.getenv("WALKIE_GRAPHS_MOTION_MAX_TRANS_M", "0.0"))
        self.motion_max_rot_deg = float(os.getenv("WALKIE_GRAPHS_MOTION_MAX_ROT_DEG", "0.0"))
        self._debug = os.getenv("WALKIE_GRAPHS_DEBUG", "0").lower() in ("1", "true", "yes")
        # Per-stage timing breakdown printed each tick (set WALKIE_GRAPHS_PERF=0 to mute).
        self._perf = os.getenv("WALKIE_GRAPHS_PERF", "1").strip().lower() in ("1", "true", "yes")
        self._last_cam = None  # latest CameraPose, for the visualizer
        self._tick = 0  # ingest_frame call counter, drives the relation/prune/viz cadence
        self._last_touched: list = []  # nodes upserted by the last ingest_frame (for observe())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def stop_and_join(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout)
        try:  # persist any deferred clouds/captures/background before the process exits
            self.memory.flush_pcds()
            if getattr(self.memory, "capture_store", None) is not None:
                self.memory.capture_store.flush()
            if getattr(self.memory, "background", None) is not None:
                self.memory.background.save()
        except Exception as e:  # noqa: BLE001 — shutdown must not raise
            self._log(f"final flush failed: {e}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[graphs] {msg}")

    def _perf_log(self, msg: str) -> None:
        if self._perf:
            print(f"==========WALKIE-GRAPH========== {msg}")

    def _wait_after(self, elapsed: float) -> float:
        """Seconds to sleep for **fixed-rate** scheduling (one observation per interval).

        Returns the leftover of the interval after a cycle that took ``elapsed`` seconds,
        or 0.0 when the cycle already overran — i.e. observe again immediately. So a 1 s
        observation with a 3 s interval waits 2 s, and an observation slower than the
        interval waits not at all.
        """
        return max(0.0, self.interval - elapsed)

    def run(self) -> None:
        self._log(f"started (interval={self.interval}s)")
        self._warmup_open3d()
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                # _observe_once -> ingest_frame(tick=True) handles the relation/prune/viz
                # cadence, so the standalone thread and perception's driver share one path.
                self._observe_once()
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill the thread
                self._log(f"tick error: {e}")
            # Fixed rate: subtract the time the cycle took from the interval. If it ran
            # longer than the interval, don't wait — start the next observation now.
            elapsed = time.perf_counter() - start
            remaining = self._wait_after(elapsed)
            if remaining > 0:
                self._stop_event.wait(remaining)
            elif elapsed > self.interval:
                self._perf_log(
                    f"cycle took {elapsed:.2f}s > interval {self.interval:.1f}s — "
                    f"observing immediately (no wait)"
                )
        self._log("stopped.")

    def _warmup_open3d(self) -> None:
        """Pay Open3D's lazy ``import`` (~0.8s) + first-solve JIT/CUDA context up front.

        Without this the whole cost lands inside the first real perception tick
        (observed as a ~0.5s spike). Doing it here on the background thread before
        the loop moves that one-off off the hot path. Best-effort: a missing wheel
        or any failure is non-fatal — registration just stays a no-op as before.
        """
        try:
            from . import pcd_ops

            t0 = time.perf_counter()
            device = pcd_ops.warmup()
            self._perf_log(
                f"open3d warmup ({device}): {(time.perf_counter() - t0) * 1000:.0f}ms"
            )
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            self._log(f"open3d warmup skipped: {e}")

    # ------------------------------------------------------------------
    # One observation
    # ------------------------------------------------------------------
    def _observe_once(self) -> list:
        """Capture one RGB-D frame *with* its pose/intrinsics in one atomic read, fold every
        kept detection into the graph, and (when a ``snapshot_path`` is set) write the live
        ``perception.json`` view of the frame.

        This is *the* production perception loop: the background thread runs it every tick.
        The full sensor + pose :class:`FrameSnapshot` is taken up front, *before* the slow
        detection round-trip, so the camera pose that lifts the depth (and the robot heading
        in the snapshot) matches where the robot was when the frame was taken — the robot
        keeps moving while detection runs. Detection is scoped to the interested classes
        plus the excluded ones in a masking-only role (``self.detect_prompts``) — the
        graph and the snapshot still only carry interested classes; excluded masks just
        carve the background.
        """
        frame = self._capture_frame()
        if frame is None:
            return []
        t_det = time.perf_counter()
        detections = self.walkieAI.object_detection.detect(
            frame.img, prompts=self.detect_prompts or None, return_mask=True
        )
        d_det = time.perf_counter() - t_det
        self._perf_log(f"detect: {d_det * 1000:.0f}ms ({len(detections)} detections)")
        if self._moved_during(frame):
            # The robot/head moved across the capture→now window: the frame's cloud
            # would land mis-posed and smear the graph. Blank the geometry so the
            # existing no-geometry path reports unknown positions and upserts nothing
            # (the snapshot still gets written; the next still frame is one tick away).
            frame = replace(frame, cam=None)
        result = self.ingest_frame(frame, detections, tick=True)
        if self.snapshot_path is not None:
            self._write_snapshot(frame, detections, result)
        return self._last_touched

    def _moved_during(self, frame: FrameSnapshot) -> bool:
        """Did the camera move between the frame's capture and now (post-detection)?

        Compares the capture-time camera pose with a fresh lookup. The detection
        round-trip (~0.5 s) provides the bracketing window: if the two poses agree
        within ``motion_max_trans_m`` / ``motion_max_rot_deg``, the robot was still for
        the whole window covering the frame's exposure, so the cloud is trustworthy.
        Over-rejects a frame where motion started only after capture — acceptable, the
        next clean frame is one tick away. Gate disabled (False) when both bounds are 0
        or either pose is unavailable.
        """
        if self.motion_max_trans_m <= 0 and self.motion_max_rot_deg <= 0:
            return False
        if frame.cam is None:
            return False  # no geometry anyway; ingest_frame handles it
        cam_now = camera_pose(self.walkie, log=self._log)
        if cam_now is None:
            return True  # pose just vanished mid-tick: don't trust the frame
        d_trans = float(np.linalg.norm(cam_now.t - frame.cam.t))
        # Rotation angle between the two orientations: angle(Ra^T @ Rb).
        cos_ang = (np.trace(frame.cam.R.T @ cam_now.R) - 1.0) / 2.0
        d_rot_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))
        moved = (self.motion_max_trans_m > 0 and d_trans > self.motion_max_trans_m) or (
            self.motion_max_rot_deg > 0 and d_rot_deg > self.motion_max_rot_deg
        )
        if moved:
            self._perf_log(
                f"motion gate: moved {d_trans * 100:.1f}cm / {d_rot_deg:.1f}deg during "
                f"capture window — skipping graph ingest for this frame"
            )
        return moved

    def _capture_frame(self) -> FrameSnapshot | None:
        """Read depth, RGB, camera pose, intrinsics, and robot pose as one atomic frame.

        Delegates to :meth:`WalkieInterface.capture_snapshot` — everything the rest of
        the tick consumes is read back-to-back and *before* detection, so it all
        describes the same instant. Returns ``None`` (skip the tick) when depth or the image is
        unavailable; ``cam``/``intr`` may be ``None`` and are handled downstream as
        "no geometry".
        """
        t0 = time.perf_counter()
        frame = self.walkie.capture_snapshot(log=self._log)
        self._perf_log(
            f"capture stage: {(time.perf_counter() - t0) * 1000:.0f}ms"
            + ("" if frame is not None else " (no frame — skipping tick)")
        )
        return frame

    def _write_snapshot(
        self, frame: FrameSnapshot, detections, ingest_result: dict[int, dict]
    ) -> None:
        """Write the live perception.json from one frame's snapshot + detections + ingest result.

        Reuses the centroids/captions the ingest pass already produced (no extra detection
        or caption work) and the robot heading captured *with* the frame, so the per-object
        headings match where the robot was at capture time. Masking-only detections
        (excluded classes, prompted purely to carve the background) are dropped — the
        live view stays scoped to the interested classes, as before. A graph hiccup or
        a build error must never take the loop down, so the whole write is best-effort.
        """
        try:
            pose = frame.robot_pose or {"heading": 0.0}
            robot_heading = float(pose.get("heading", 0.0))
            shown = [
                i
                for i, d in enumerate(detections)
                if (d.class_name or "").lower() not in self.exclude
            ]
            objs = build_object_records(
                [detections[i] for i in shown],
                {j: ingest_result[i] for j, i in enumerate(shown)},
                frame.img.size,
                robot_heading,
            )
            write_atomic(self.snapshot_path, {"ts": frame.ts, "objects": objs})
        except Exception as e:  # noqa: BLE001
            self._log(f"snapshot write failed: {e}")

    def ingest_frame(
        self, frame: FrameSnapshot, detections, *, tick: bool = True
    ) -> dict[int, dict]:
        """Fold the kept subset of one captured frame's ``detections`` into the graph.

        ``frame`` is the atomic :class:`FrameSnapshot` from :meth:`_capture_frame` — its
        ``img``/``depth``/``cam``/``intr``/``robot_pose`` were all read at the *same* instant
        the frame was taken, so the camera pose that lifts the depth matches the image
        instead of trailing the detection round-trip. ``detections`` is the list of
        ``DetectedObject`` for that frame (masks required for 3D — ``return_mask=True``).

        Returns a per-detection dict keyed by index into ``detections``::

            {i: {"centroid": (x, y, z) | None, "caption": str}}

        A centroid is returned for **every** detection that deprojects, so the caller
        (perception) can fill ``position_3d`` for the whole live view; ``None`` when the
        detection has no mask, depth/pose is missing, or no masked pixel has valid depth.
        Only detections passing ``_keep`` + ``min_confidence`` + ``min_points`` are
        upserted into :class:`GraphMemory` — ``min_points`` gates graph entry only, not the
        returned centroid (a sparse cloud still gives a usable position but is too thin to
        fuse durably). When ``tick`` is true this advances the relation/prune/viz cadence.
        """
        t_frame = time.perf_counter()
        img, depth, cam, intr = frame.img, frame.depth, frame.cam, frame.intr
        # Default: unknown position, no caption, for every detection.
        result: dict[int, dict] = {
            i: {"centroid": None, "caption": ""} for i in range(len(detections))
        }

        if cam is None or intr is None:
            # No geometry this frame (pose or camera-info unavailable at capture): report
            # unknown positions, upsert nothing, but still advance the cadence so periodic
            # maintenance keeps ticking.
            self._last_touched = []
            self._maybe_tick(tick, robot_pose=frame.robot_pose)
            self._perf_log("ingest aborted (no geometry): pose/intrinsics unavailable at capture")
            return result
        self._last_cam = cam

        # Deproject every masked detection once: centroid for the live view, and (when
        # dense enough + a kept class) the full cloud for graph upsert.
        try:
            img_area = int(img.size[0]) * int(img.size[1])
        except Exception:  # noqa: BLE001
            img_area = 0

        # CG mask_subtract_contained: remove each contained detection's pixels from its
        # container's mask, so a table's cloud doesn't absorb the mug sitting on it.
        t = time.perf_counter()
        masks = [d.mask for d in detections]
        if self.mask_subtract and sum(m is not None for m in masks) >= 2:
            try:
                masks = subtract_contained_masks([d.bbox for d in detections], masks)
            except Exception as e:  # noqa: BLE001 — never let mask cleanup kill the tick
                self._log(f"mask subtract failed: {e}")
                masks = [d.mask for d in detections]
        d_masksub = time.perf_counter() - t

        # Depth discontinuity map, computed once per frame (shared by all detections):
        # drops "flying pixels" at silhouettes where depth mixes foreground+background.
        t = time.perf_counter()
        edge_mask = depth_discontinuity_mask(
            depth, self.depth_edge_thresh_m, rel_thresh=self.depth_edge_rel
        )
        d_edge = time.perf_counter() - t

        t = time.perf_counter()
        pending = []  # (orig_index, detected, points, crop)
        for i, d in enumerate(detections):
            mask = masks[i]
            if mask is None or not mask.any():
                continue
            if (d.class_name or "").lower() in self.exclude:
                # Masking-only detection: its mask carves the background union below,
                # but it gets no centroid (not shown in the live view) and no segment.
                continue
            if not self._passes_size_filters(d, img_area):
                continue
            pts = deproject_mask(
                mask,
                depth,
                intr,
                cam,
                voxel=self.voxel_m,
                max_points=self.max_points,
                erode_px=self.mask_erode_px,
                edge_mask=edge_mask,
                max_depth=self.max_depth_m,
                sor_k=self.sor_k,
                sor_std_ratio=self.sor_std_ratio,
            )
            if len(pts) == 0:
                continue
            c = pts.mean(axis=0)
            result[i]["centroid"] = (float(c[0]), float(c[1]), float(c[2]))
            if (
                self._keep(d.class_name or "")
                and float(d.confidence or 0.0) >= self.min_confidence
                and len(pts) >= self.min_points
            ):
                pending.append((i, d, pts, _crop_pil(img, d.bbox, self.crop_margin_px)))
        d_deproject = time.perf_counter() - t

        # Fold the frame into ONE capture: the kept segments plus the background
        # remainder (every valid-depth pixel under no mask — the dilated union
        # includes dropped and masking-only masks, so e.g. people are carved out).
        t = time.perf_counter()
        capture = lift_capture(
            frame,
            masks,
            [(i, pts) for i, _d, pts, _crop in pending],
            edge_mask=edge_mask,
            bg_voxel_m=self.bg_voxel_m,
            bg_dilate_px=self.bg_mask_dilate_px,
            bg_max_depth_m=self.bg_max_depth_m,
        )
        d_bg = time.perf_counter() - t

        # Register the capture against the map: one rigid correction for the whole
        # frame, anchored by the background. On accept, every segment (and the
        # remainder) is transformed; on reject/skip the frame ingests raw.
        t = time.perf_counter()
        capture = self._register_capture(capture)
        if capture.segments:
            corrected = {s.det_idx: s for s in capture.segments}
            pending = [(i, d, corrected[i].points, crop) for i, d, _pts, crop in pending]
            for i, d, pts, _crop in pending:
                c = pts.mean(axis=0)
                result[i]["centroid"] = (float(c[0]), float(c[1]), float(c[2]))
        d_icp = time.perf_counter() - t

        # Single caption pass over the captionable kept subset (reuses _caption's policy),
        # mapped back to original detection indices.
        t = time.perf_counter()
        captions = self._caption([p[1] for p in pending], [p[3] for p in pending])
        for local, cap in captions.items():
            result[pending[local][0]]["caption"] = cap
        d_caption = time.perf_counter() - t

        # Embed the kept crops — one network round-trip PER object, run concurrently so
        # the per-frame embed cost is the slowest single call, not their sum.
        t = time.perf_counter()
        embeddings = self._embed_batch([p[3] for p in pending])
        d_embed = time.perf_counter() - t

        # Persist the capture's segments (deferred write) BEFORE upserting, so every
        # segment_ref the nodes take is already readable from the store.
        if getattr(self.memory, "capture_store", None) is not None:
            self.memory.capture_store.save(capture)

        # Upsert the kept subset, batching the per-object Chroma writes into one flush.
        t = time.perf_counter()
        touched = []
        seg_refs = {s.det_idx: s.ref for s in capture.segments}
        with self.memory.batch_writes():
            for (i, d, pts, crop), emb in zip(pending, embeddings):
                det = Detection3D(
                    class_name=d.class_name or "object",
                    class_id=d.class_id,
                    confidence=float(d.confidence or 0.0),
                    bbox_xyxy=tuple(int(v) for v in d.bbox),
                    points_world=pts,
                    clip_emb=emb,
                    caption=result[i]["caption"],
                    ts=time.time(),
                    crop=crop,
                    segment_ref=seg_refs.get(i),
                )
                touched.append(self.memory.upsert(det))
        d_upsert = time.perf_counter() - t
        self._last_touched = touched

        # When capture registration was rejected (raw pose error still in the segments),
        # flag the touched nodes so the refine pass can co-align their segments later.
        if (
            self.capture_icp_max_corr_m > 0
            and not capture.icp_accepted
            and touched
            and hasattr(self.memory, "flag_for_refine")
        ):
            self.memory.flag_for_refine([n.id for n in touched])

        # The classless remainder joins the world background (viz + future ICP anchor).
        if getattr(self.memory, "background", None) is not None and len(capture.background):
            self.memory.background.add(capture.background)

        t = time.perf_counter()
        self._maybe_tick(
            tick, touched=touched, robot_pose=frame.robot_pose, frame=frame, capture=capture
        )
        d_tick = time.perf_counter() - t

        total = time.perf_counter() - t_frame
        n_cap = len(captions)
        stats = self.memory.pop_perf_stats()
        breakdown = " ".join(f"{k}={v * 1000:.0f}ms" for k, v in sorted(stats.items()))
        icp_state = (
            f"fit={capture.icp_fitness:.2f}"
            + ("" if capture.icp_accepted else " rejected")
            if capture.icp_fitness > 0
            else "skipped"
        )
        self._perf_log(
            f"ingest tick #{self._tick}: TOTAL={total:.2f}s | "
            f"masksub={d_masksub * 1000:.0f}ms edge={d_edge * 1000:.0f}ms "
            f"deproject={d_deproject * 1000:.0f}ms ({len(detections)} det) | "
            f"bg={d_bg * 1000:.0f}ms ({len(capture.background)} pts) "
            f"capture_icp={d_icp * 1000:.0f}ms ({icp_state}) | "
            f"caption={d_caption * 1000:.0f}ms ({n_cap}) "
            f"embed={d_embed * 1000:.0f}ms ({len(pending)}) "
            f"upsert={d_upsert * 1000:.0f}ms [{breakdown}] | "
            f"maint+viz={d_tick * 1000:.0f}ms"
        )
        return result

    def _register_capture(self, capture):
        """Solve the capture's pose correction against the map (skip when starved).

        Skips (identity) when disabled (``capture_icp_max_corr_m <= 0``), when the
        capture has nothing to align, or when the map target around the capture is
        still too thin to anchor a solve (cold start) — ``register_capture`` itself
        enforces the fitness and translation/rotation caps.
        """
        if self.capture_icp_max_corr_m <= 0:
            return capture
        parts = [capture.background] + [s.points for s in capture.segments]
        parts = [p for p in parts if len(p)]
        if not parts:
            return capture
        mins = np.min([p.min(axis=0) for p in parts], axis=0)
        maxs = np.max([p.max(axis=0) for p in parts], axis=0)
        target = self.memory.icp_target_near(
            tuple(mins), tuple(maxs), pad=0.5, budget=self.capture_icp_tgt_budget
        )
        return register_capture(
            capture,
            target,
            max_corr_dist=self.capture_icp_max_corr_m,
            min_fitness=self.capture_icp_min_fitness,
            max_trans_m=self.capture_icp_max_trans_m,
            max_rot_deg=self.capture_icp_max_rot_deg,
            src_budget=self.capture_icp_src_budget,
        )

    def _passes_size_filters(self, d, img_area: int) -> bool:
        """Drop degenerate masks and whole-frame (background) boxes — CG ``filter_gobs``."""
        if self.min_mask_area_px > 0 and d.mask is not None:
            try:
                if int(d.mask.sum()) < self.min_mask_area_px:
                    return False
            except Exception:  # noqa: BLE001
                pass
        if self.max_bbox_area_ratio < 1.0 and img_area > 0:
            x1, y1, x2, y2 = d.bbox
            if (x2 - x1) * (y2 - y1) > self.max_bbox_area_ratio * img_area:
                return False
        return True

    def _maybe_tick(
        self,
        tick: bool,
        *,
        touched: list | None = None,
        robot_pose=None,
        frame=None,
        capture=None,
    ) -> None:
        """Advance the cadence counter; run relations/prune + periodic maintenance + viz.

        ``robot_pose`` is the frame-time robot pose (from the snapshot), used for the viz
        marker so it matches the objects just ingested. ``frame``/``capture`` (the lifted,
        registered capture this tick) feed free-space carving — omitted on the no-geometry
        path, where carving has no depth image to project against.
        """
        if not tick:
            return
        self._tick += 1
        t = self._tick
        ran: list[str] = []  # (name=ms) for whatever periodic work actually fired

        def _timed(name, fn):
            t0 = time.perf_counter()
            fn()
            ran.append(f"{name}={ (time.perf_counter() - t0) * 1000:.0f}ms")

        if self.relation_every_n > 0 and t % self.relation_every_n == 0:
            _timed("relations", self.memory.derive_relations)
            _timed("prune", self.memory.prune)
        # Staggered ConceptGraphs post-processing — offsets 0/1/2 so no two collide,
        # and only after a full interval has elapsed (no churn on a near-empty graph).
        if self.denoise_every_n > 0 and t >= self.denoise_every_n and t % self.denoise_every_n == 0:
            _timed("denoise", self.memory.denoise_nodes)
        if self.merge_every_n > 0 and t >= self.merge_every_n and t % self.merge_every_n == 1:
            _timed("merge", self.memory.merge_overlapping_nodes)
        if self.ghost_every_n > 0 and t >= self.ghost_every_n and t % self.ghost_every_n == 2:
            _timed("evict", lambda: self.memory.evict_stale_provisional(time.time()))
        # Free-space carving (offset 3). Gated on a TRUSTED pose: registration off
        # (correction is identity) or an accepted solve — never a rejected/cold one.
        if (
            self.carve_every_n > 0
            and t >= self.carve_every_n
            and t % self.carve_every_n == 3
            and frame is not None
            and capture is not None
            and getattr(frame, "depth", None) is not None
            and getattr(frame, "intr", None) is not None
            and getattr(frame, "cam", None) is not None
            and hasattr(self.memory, "carve_free_space")
            and (self.capture_icp_max_corr_m <= 0 or capture.icp_accepted)
        ):
            from .carve import corrected_pose

            pose = corrected_pose(capture.cam, capture.correction)
            _timed(
                "carve",
                lambda: self.memory.carve_free_space(
                    frame.depth, frame.intr, pose,
                    margin_base=self.carve_margin_base,
                    margin_rel=self.carve_margin_rel,
                    z_min=self.carve_z_min,
                    max_z=self.carve_max_z,
                    evict_min_points=self.carve_evict_min_points,
                    refine_frac=self.carve_refine_frac,
                ),
            )
        if (
            self.refine_every_n > 0
            and t >= self.refine_every_n
            and t % self.refine_every_n == 6
            and hasattr(self.memory, "refine_nodes")
        ):
            _timed("refine_nodes", lambda: self.memory.refine_nodes(self.refine_limit))
        if self.pcd_flush_every_n > 0 and t % self.pcd_flush_every_n == 0:
            _timed("pcd_flush", self.memory.flush_pcds)
            if getattr(self.memory, "capture_store", None) is not None:
                _timed("cap_flush", self.memory.capture_store.flush)
                _timed("cap_gc", self.memory.capture_store.gc)
        # Background persistence on its own (offset) cadence — losing a few ticks of
        # wall points to a crash is harmless, they re-observe on the next pass.
        if (
            getattr(self.memory, "background", None) is not None
            and self.bg_save_every_n > 0
            and t % self.bg_save_every_n == self.bg_save_every_n - 1
        ):
            _timed("bg_save", self.memory.background.save)
        # Tier 3 LLM passes (only when a model is wired and the cadence is enabled).
        if (
            self.model is not None
            and self.caption_refine_every_n > 0
            and t >= self.caption_refine_every_n
            and t % self.caption_refine_every_n == 3
        ):
            _timed(
                "refine",
                lambda: self.memory.refine_captions(
                    self.model,
                    limit=self.caption_refine_limit,
                    use_images=self.caption_refine_use_images,
                ),
            )
        if (
            self.model is not None
            and self.llm_edges_every_n > 0
            and t >= self.llm_edges_every_n
            and t % self.llm_edges_every_n == 4
        ):
            _timed("llm_edges", lambda: self.memory.infer_edges_llm(self.model))
        if self.viz is not None and touched is not None:
            t0 = time.perf_counter()
            # Frame-time robot pose so the viz marker matches the objects just ingested.
            try:
                self.viz.update(self.memory, robot_pose=robot_pose, cam_pose=self._last_cam)
            except Exception as e:  # noqa: BLE001
                self._log(f"viz update failed: {e}")
            ran.append(f"viz={(time.perf_counter() - t0) * 1000:.0f}ms")
        if ran:
            self._perf_log(f"  maintenance: {' '.join(ran)}")

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

    def _embed_batch(self, crops: list) -> list[list[float]]:
        """Embed many crops concurrently — there's no batch endpoint, so fan the
        per-crop network calls out across a small thread pool (they're I/O-bound)."""
        if not crops:
            return []
        workers = min(len(crops), max(1, self._embed_workers))
        if workers <= 1:
            return [self._embed(c) for c in crops]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self._embed, crops))

