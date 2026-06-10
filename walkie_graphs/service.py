"""WalkieGraphsService — the background thread that grows the scene graph.

Mirrors :class:`services.perception.PerceptionService`: a daemon thread that, each
tick, resolves the camera's pose (the optical frame's pose in the map frame from the
SDK ``transform.lookup``) and intrinsics (``camera.get_intrinsics``), captures an
RGB-D frame, runs masked object detection (scoped to the *interested classes*), lifts
each mask to a 3D world point cloud, embeds + captions the crop via the AI client, and
upserts it into :class:`~walkie_graphs.memory.GraphMemory`. Every few ticks it
recomputes the geometric relations, prunes, and pushes to the visualizer.

Detection/caption/embedding are all direct ``walkieAI`` client calls — there is no
local model here.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np

from .fusion import subtract_contained_masks
from .geometry import CameraPose, Intrinsics, deproject_mask
from .memory import Detection3D, GraphMemory


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

        self.min_confidence = float(os.getenv("WALKIE_GRAPHS_CONFIDENCE_THRESHOLD", "0.0"))
        self.min_points = int(os.getenv("WALKIE_GRAPHS_MIN_POINTS", "30"))
        self.voxel_m = float(os.getenv("WALKIE_GRAPHS_VOXEL_M", "0.02"))
        self.max_points = int(os.getenv("WALKIE_GRAPHS_MAX_POINTS_PER_OBJ", "2000"))
        self.relation_every_n = int(os.getenv("WALKIE_GRAPHS_RELATION_EVERY_N", "5"))
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
        # Periodic-maintenance cadences (in ingest ticks), staggered so two heavy
        # passes never land on the same tick. 0 disables a pass.
        self.denoise_every_n = int(os.getenv("WALKIE_GRAPHS_DENOISE_EVERY_N", "20"))
        self.merge_every_n = int(os.getenv("WALKIE_GRAPHS_MERGE_EVERY_N", "20"))
        self.ghost_every_n = int(os.getenv("WALKIE_GRAPHS_GHOST_EVERY_N", "20"))
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
        self._debug = os.getenv("WALKIE_GRAPHS_DEBUG", "0").lower() in ("1", "true", "yes")
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

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[graphs] {msg}")

    def run(self) -> None:
        self._log(f"started (interval={self.interval}s)")
        while not self._stop_event.is_set():
            try:
                # _observe_once -> ingest_frame(tick=True) handles the relation/prune/viz
                # cadence, so the standalone thread and perception's driver share one path.
                self._observe_once()
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill the thread
                self._log(f"tick error: {e}")
            self._stop_event.wait(self.interval)
        self._log("stopped.")

    # ------------------------------------------------------------------
    # One observation
    # ------------------------------------------------------------------
    def _observe_once(self) -> list:
        """Capture one RGB-D frame and fold every kept detection into the graph.

        Standalone path — used by the background thread and the ``observe()`` facade.
        In production, perception captures + detects once per frame and calls
        :meth:`ingest_frame` directly, so the detector never runs twice on one frame.
        """
        depth = self._depth()
        if depth is None:
            return []
        try:
            img = self.walkie.camera.capture_pil()  # PIL RGB
        except Exception as e:  # noqa: BLE001
            self._log(f"capture failed: {e}")
            return []
        detections = self.walkieAI.object_detection.detect(
            img, prompts=self.interested or None, return_mask=True
        )
        self.ingest_frame(img, detections, depth, tick=True)
        return self._last_touched

    def ingest_frame(self, img, detections, depth, *, tick: bool = True) -> dict[int, dict]:
        """Fold the kept subset of one captured frame's ``detections`` into the graph.

        ``img`` is the PIL RGB frame the detections came from; ``detections`` is a list
        of ``DetectedObject`` (masks required for 3D — request ``return_mask=True``);
        ``depth`` is the aligned depth array (H×W float32 metres, NaN invalid) or None.

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
        # Default: unknown position, no caption, for every detection.
        result: dict[int, dict] = {
            i: {"centroid": None, "caption": ""} for i in range(len(detections))
        }

        cam = self._camera_pose() if depth is not None else None
        intr = self._intrinsics(depth.shape[1], depth.shape[0]) if cam is not None else None
        if cam is None or intr is None:
            # No geometry this frame (pose or intrinsics unavailable): report unknown
            # positions, upsert nothing, but still advance the cadence so periodic
            # maintenance keeps ticking.
            self._last_touched = []
            self._maybe_tick(tick)
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
        masks = [d.mask for d in detections]
        if self.mask_subtract and sum(m is not None for m in masks) >= 2:
            try:
                masks = subtract_contained_masks([d.bbox for d in detections], masks)
            except Exception as e:  # noqa: BLE001 — never let mask cleanup kill the tick
                self._log(f"mask subtract failed: {e}")
                masks = [d.mask for d in detections]

        pending = []  # (orig_index, detected, points, crop)
        for i, d in enumerate(detections):
            mask = masks[i]
            if mask is None or not mask.any():
                continue
            if not self._passes_size_filters(d, img_area):
                continue
            pts = deproject_mask(
                mask, depth, intr, cam, voxel=self.voxel_m, max_points=self.max_points
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

        # Single caption pass over the captionable kept subset (reuses _caption's policy),
        # mapped back to original detection indices.
        captions = self._caption([p[1] for p in pending], [p[3] for p in pending])
        for local, cap in captions.items():
            result[pending[local][0]]["caption"] = cap

        # Embed + upsert the kept subset.
        touched = []
        for i, d, pts, crop in pending:
            det = Detection3D(
                class_name=d.class_name or "object",
                class_id=d.class_id,
                confidence=float(d.confidence or 0.0),
                bbox_xyxy=tuple(int(v) for v in d.bbox),
                points_world=pts,
                clip_emb=self._embed(crop),
                caption=result[i]["caption"],
                ts=time.time(),
                crop=crop,
            )
            touched.append(self.memory.upsert(det))
        self._last_touched = touched
        self._maybe_tick(tick, touched=touched)
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

    def _maybe_tick(self, tick: bool, *, touched: list | None = None) -> None:
        """Advance the cadence counter; run relations/prune + periodic maintenance + viz."""
        if not tick:
            return
        self._tick += 1
        t = self._tick
        if self.relation_every_n > 0 and t % self.relation_every_n == 0:
            self.memory.derive_relations()
            self.memory.prune()
        # Staggered ConceptGraphs post-processing — offsets 0/1/2 so no two collide,
        # and only after a full interval has elapsed (no churn on a near-empty graph).
        if self.denoise_every_n > 0 and t >= self.denoise_every_n and t % self.denoise_every_n == 0:
            self.memory.denoise_nodes()
        if self.merge_every_n > 0 and t >= self.merge_every_n and t % self.merge_every_n == 1:
            self.memory.merge_overlapping_nodes()
        if self.ghost_every_n > 0 and t >= self.ghost_every_n and t % self.ghost_every_n == 2:
            self.memory.evict_stale_provisional(time.time())
        # Tier 3 LLM passes (only when a model is wired and the cadence is enabled).
        if (
            self.model is not None
            and self.caption_refine_every_n > 0
            and t >= self.caption_refine_every_n
            and t % self.caption_refine_every_n == 3
        ):
            self.memory.refine_captions(
                self.model,
                limit=self.caption_refine_limit,
                use_images=self.caption_refine_use_images,
            )
        if (
            self.model is not None
            and self.llm_edges_every_n > 0
            and t >= self.llm_edges_every_n
            and t % self.llm_edges_every_n == 4
        ):
            self.memory.infer_edges_llm(self.model)
        if self.viz is not None and touched is not None:
            try:
                self.viz.update(
                    self.memory,
                    robot_pose=self.walkie.status.get_position(),
                    cam_pose=self._last_cam,
                )
            except Exception as e:  # noqa: BLE001
                self._log(f"viz update failed: {e}")

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
