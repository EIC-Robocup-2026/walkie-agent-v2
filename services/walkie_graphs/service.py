"""WalkieGraphsService — the robot's perception loop and scene-graph builder.

A daemon thread that, each tick, resolves the camera's pose (the optical frame's pose in
the map frame from the SDK ``transform.lookup``) and intrinsics (``camera.get_intrinsics``),
captures an RGB-D frame, runs masked object detection (scoped to the *interested classes*),
lifts each mask to a 3D world point cloud, embeds + captions the crop via the AI client, and
upserts it into :class:`~services.walkie_graphs.memory.GraphMemory`. When a ``snapshot_path``
is configured it then writes the live ``perception.json`` view of the frame (the agents'
"what's in front of me now" context). Every few ticks it recomputes the geometric relations,
prunes, and pushes to the visualizer.

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

from .fusion import subtract_contained_masks
from .geometry import (
    CameraPose,
    Intrinsics,
    depth_discontinuity_mask,
    deproject_mask,
)
from .memory import Detection3D, GraphMemory
from .snapshot import build_object_records, write_atomic


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


@dataclass
class FrameSnapshot:
    """Every sensor + pose read for one tick, taken together at the capture instant.

    The detection round-trip that follows is slow and the robot keeps moving through it,
    so reading the camera/robot pose *after* detection would describe a different instant
    than the depth/image it lifts. Capturing it all here pins the lift pose, intrinsics,
    and snapshot heading to the moment the frame was actually taken. ``cam``/``intr`` may
    be ``None`` (pose or camera-info unavailable) and are handled downstream as "no
    geometry this frame"; ``depth`` and ``img`` are always present (a missing one skips
    the tick before a snapshot is built).
    """

    ts: float
    img: object  # PIL RGB frame
    depth: np.ndarray  # aligned depth, H×W metres (NaN invalid)
    cam: CameraPose | None  # camera optical-frame pose in the map frame
    intr: Intrinsics | None  # pinhole intrinsics at the depth resolution
    robot_pose: dict | None  # status.get_position() — snapshot heading + viz marker


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
        # Tier 3 (optional LLM): caption refinement + LLM edge inference. 0 = off, and
        # both require self.model. They only ever run on these cadences when enabled.
        self.caption_refine_every_n = int(os.getenv("WALKIE_GRAPHS_CAPTION_REFINE_EVERY_N", "0"))
        self.caption_refine_limit = int(os.getenv("WALKIE_GRAPHS_CAPTION_REFINE_LIMIT", "8"))
        self.caption_refine_use_images = os.getenv(
            "WALKIE_GRAPHS_CAPTION_REFINE_USE_IMAGES", "0"
        ).strip().lower() in ("1", "true", "yes")
        self.llm_edges_every_n = int(os.getenv("WALKIE_GRAPHS_LLM_EDGES_EVERY_N", "0"))

        # Camera calibration + pose come straight from the walkie-sdk: real pinhole
        # intrinsics from camera.get_intrinsics(), and the camera OPTICAL frame's pose
        # in the map frame from transform.lookup(MAP_FRAME, CAMERA_FRAME) — the optical
        # frame already bakes in lift, head tilt, and every mount offset, and its axes
        # match the pinhole math, so no manual composition or axis conversion is needed.
        self._tf_map_frame = os.getenv("WALKIE_GRAPHS_TF_MAP_FRAME", "map")
        self._tf_cam_frame = os.getenv(
            "WALKIE_GRAPHS_TF_CAMERA_FRAME", "zed_head_left_camera_frame_optical"
        )
        self._tf_timeout = float(os.getenv("WALKIE_GRAPHS_TF_TIMEOUT_SEC", "1.0"))
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
        self._intr_cache: dict[tuple[int, int], Intrinsics] = {}
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
        try:  # persist any deferred point clouds before the process exits
            self.memory.flush_pcds()
        except Exception as e:  # noqa: BLE001 — shutdown must not raise
            self._log(f"final pcd flush failed: {e}")

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
        (``prompts=self.interested``), so both the graph and the snapshot only carry those.
        """
        frame = self._capture_frame()
        if frame is None:
            return []
        t_det = time.perf_counter()
        detections = self.walkieAI.object_detection.detect(
            frame.img, prompts=self.interested or None, return_mask=True
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
        cam_now = self._camera_pose()
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

        Everything the rest of the tick consumes is read here, back-to-back and *before*
        detection, so it all describes the same instant — the camera pose that lifts the
        depth isn't skewed by the detection round-trip the robot drives through. Returns
        ``None`` (skip the tick) when depth or the image is unavailable (both mandatory);
        ``cam``/``intr`` may be ``None`` and are handled downstream as "no geometry".
        """
        t_depth = time.perf_counter()
        depth = self._depth()
        d_depth = time.perf_counter() - t_depth
        if depth is None:
            self._perf_log(f"depth=None ({d_depth * 1000:.0f}ms) — skipping tick")
            return None
        try:
            t_cap = time.perf_counter()
            img = self.walkie.camera.capture_pil()  # PIL RGB
            d_cap = time.perf_counter() - t_cap
        except Exception as e:  # noqa: BLE001
            self._log(f"capture failed: {e}")
            return None
        # Pose, intrinsics, and robot pose: read NOW, still at the capture instant, so the
        # lift pose matches the depth/image instead of trailing the detection round-trip.
        ts = time.time()
        t_pose = time.perf_counter()
        cam = self._camera_pose()
        d_pose = time.perf_counter() - t_pose
        t_intr = time.perf_counter()
        intr = self._intrinsics(depth.shape[1], depth.shape[0]) if cam is not None else None
        d_intr = time.perf_counter() - t_intr
        robot_pose = self._robot_pose()
        self._perf_log(
            f"capture stage: depth={d_depth * 1000:.0f}ms capture={d_cap * 1000:.0f}ms "
            f"pose={d_pose * 1000:.0f}ms intr={d_intr * 1000:.0f}ms"
        )
        return FrameSnapshot(
            ts=ts, img=img, depth=depth, cam=cam, intr=intr, robot_pose=robot_pose
        )

    def _write_snapshot(
        self, frame: FrameSnapshot, detections, ingest_result: dict[int, dict]
    ) -> None:
        """Write the live perception.json from one frame's snapshot + detections + ingest result.

        Reuses the centroids/captions the ingest pass already produced (no extra detection
        or caption work) and the robot heading captured *with* the frame, so the per-object
        headings match where the robot was at capture time. A graph hiccup or a build error
        must never take the loop down, so the whole write is best-effort.
        """
        try:
            pose = frame.robot_pose or {"heading": 0.0}
            robot_heading = float(pose.get("heading", 0.0))
            objs = build_object_records(detections, ingest_result, frame.img.size, robot_heading)
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

        # Upsert the kept subset, batching the per-object Chroma writes into one flush.
        t = time.perf_counter()
        touched = []
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
                )
                touched.append(self.memory.upsert(det))
        d_upsert = time.perf_counter() - t
        self._last_touched = touched

        t = time.perf_counter()
        self._maybe_tick(tick, touched=touched, robot_pose=frame.robot_pose)
        d_tick = time.perf_counter() - t

        total = time.perf_counter() - t_frame
        n_cap = len(captions)
        stats = self.memory.pop_perf_stats()
        breakdown = " ".join(f"{k}={v * 1000:.0f}ms" for k, v in sorted(stats.items()))
        self._perf_log(
            f"ingest tick #{self._tick}: TOTAL={total:.2f}s | "
            f"masksub={d_masksub * 1000:.0f}ms edge={d_edge * 1000:.0f}ms "
            f"deproject={d_deproject * 1000:.0f}ms ({len(detections)} det) | "
            f"caption={d_caption * 1000:.0f}ms ({n_cap}) "
            f"embed={d_embed * 1000:.0f}ms ({len(pending)}) "
            f"upsert={d_upsert * 1000:.0f}ms [{breakdown}] | "
            f"maint+viz={d_tick * 1000:.0f}ms"
        )
        return result

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
        self, tick: bool, *, touched: list | None = None, robot_pose=None
    ) -> None:
        """Advance the cadence counter; run relations/prune + periodic maintenance + viz.

        ``robot_pose`` is the frame-time robot pose (from the snapshot), used for the viz
        marker so it matches the objects just ingested.
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
        if self.pcd_flush_every_n > 0 and t % self.pcd_flush_every_n == 0:
            _timed("pcd_flush", self.memory.flush_pcds)
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

    # ------------------------------------------------------------------
    # Sensor reads
    # ------------------------------------------------------------------
    def _intrinsics(self, width: int, height: int):
        """Real pinhole intrinsics from the SDK, scaled to the depth resolution.

        ``camera.get_intrinsics()`` reads the ZED's ``CameraInfo`` (cached by the SDK —
        intrinsics are static) and applies to the registered depth too. Returns
        ``None`` (skip the tick) when no camera info is available yet."""
        key = (width, height)
        cached = self._intr_cache.get(key)
        if cached is not None:
            return cached
        try:
            raw = self.walkie.robot.camera.get_intrinsics()
        except Exception as e:  # noqa: BLE001
            self._log(f"intrinsics unavailable: {e}")
            return None
        if not raw:
            self._log("intrinsics unavailable (camera_info not published yet)")
            return None
        intr = Intrinsics(
            fx=float(raw["fx"]),
            fy=float(raw["fy"]),
            cx=float(raw["cx"]),
            cy=float(raw["cy"]),
            width=int(raw.get("width") or width),
            height=int(raw.get("height") or height),
        ).scaled_to(width, height)
        self._intr_cache[key] = intr
        return intr

    def _camera_pose(self):
        """Camera optical-frame pose in the map frame, from the SDK transform tree.

        ``transform.lookup(MAP_FRAME, CAMERA_FRAME)`` with the camera *optical* frame
        returns a pose whose rotation maps camera-optical points straight into the map
        (lift, head tilt, and mount offsets already baked in), so deprojection needs no
        further composition. Returns ``None`` (skip the tick) when the lookup fails."""
        try:
            tf = self.walkie.robot.transform.lookup(
                self._tf_map_frame, self._tf_cam_frame, timeout=self._tf_timeout
            )
        except Exception as e:  # noqa: BLE001
            self._log(
                f"transform lookup error ({self._tf_map_frame} -> {self._tf_cam_frame}): {e}"
            )
            return None
        if tf is None:
            self._log(
                f"transform lookup returned None ({self._tf_map_frame} -> "
                f"{self._tf_cam_frame}); skipping tick"
            )
            return None

        from walkie_sdk.utils.converters import quaternion_to_matrix

        q, p = tf["quaternion"], tf["position"]
        R = quaternion_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        t = np.array([float(p["x"]), float(p["y"]), float(p["z"])])
        if self._debug:
            self._log(
                f"pose(optical tf {self._tf_cam_frame}) "
                f"cam=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})m"
            )
        return CameraPose(R=R, t=t)

    def _depth(self):
        try:
            return self.walkie.robot.camera.get_depth()
        except Exception as e:  # noqa: BLE001
            self._log(f"depth unavailable: {e}")
            return None

    def _robot_pose(self):
        """Robot base pose dict (``status.get_position()``) — snapshot heading + viz marker.

        Read as part of the frame snapshot so the snapshot's per-object headings use the
        pose from capture time, not a later read. Best-effort: returns ``None`` on failure
        (the snapshot then degrades the heading to 0.0 and the viz marker is omitted)."""
        try:
            return self.walkie.status.get_position()
        except Exception as e:  # noqa: BLE001
            self._log(f"robot pose unavailable: {e}")
            return None
