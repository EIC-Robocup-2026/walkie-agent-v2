"""realtime_explore — the perception PRODUCER (capture + batch build).

Renamed from ``walkie_graphs`` and split from the world model: this package only
*produces* observations and pushes them into a :class:`~walkie_world.world.WalkieWorld`
(the model now owns the scene store, relations and queries). Callers construct
``RealtimeExplore(model=, walkieAI=, walkie=, world=, snapshot_path=)`` and call
``start/stop/observe``. Query passthroughs delegate to the injected world (kept for
back-compat with the Database agent until it queries ``ctx.world`` directly).

Two loops, decoupled:

- **Capture thread** (cheap, ~real-time): every ``INTERVAL_SEC`` grab one synchronized
  RGB-D frame + one fused detect/caption/embed round-trip, write the live
  ``perception.json`` straight from the detections, and append a compact
  :class:`~.buffer.Snapshot` to the on-disk ring buffer. No ICP, no fusion.
- **Build worker** (batch, occasional, single-flight): every ``REBUILD_EVERY_N`` new
  snapshots build object observations over the recent window
  (:func:`~.builder.build_scene`) and hand them to ``world.observe_objects`` (merge →
  derive relations → install). Queries read the last installed scene.

If no ``world`` is injected the producer builds its own (objects only,
``enable_people=False``) — a transitional default; in the migrated pipeline exactly
one shared :class:`WalkieWorld` is constructed in run.py/main.py and passed in.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from walkie_world import WalkieWorld
from walkie_world.scene.store import ObjectNode, Relation

from .buffer import Detection, Snapshot, SnapshotBuffer
from .builder import build_scene
from .viz import build_scene_viz


def _envf(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _envi(name, default):
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _envb(name, default="0"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _classes(name):
    return {c.strip().lower() for c in os.getenv(name, "").split(",") if c.strip()}


class RealtimeExplore:
    """Perception producer: cheap capture into a ring buffer + occasional batch builds
    that feed a shared :class:`~walkie_world.world.WalkieWorld`."""

    def __init__(
        self,
        model=None,
        walkieAI=None,
        walkie=None,
        *,
        world: Optional[WalkieWorld] = None,
        viz=None,
        snapshot_path=None,
    ) -> None:
        self.model = model
        self.walkieAI = walkieAI
        self.walkie = walkie
        # Build (and thereby START) the shared Rerun session unless one is injected or
        # viz is disabled — constructing the producer is what kicks Rerun off in the
        # main agent flow, exactly as the old facade did.
        self.viz = viz if viz is not None else build_scene_viz()
        self.snapshot_path = snapshot_path
        self._last_structural = None  # last TSDF cloud (for the viz background)
        self._last_cam = None         # last capture's CameraPose (for the camera marker)

        # --- config (all WALKIE_EXPLORE_* env, defaulted in config.toml) ---
        self.interval = _envf("WALKIE_EXPLORE_INTERVAL_SEC", "3.0")
        self.interested = _classes("WALKIE_EXPLORE_INTERESTED_CLASSES")
        self.exclude = _classes("WALKIE_EXPLORE_EXCLUDE_CLASSES") or {"person"}
        self.caption_classes = _classes("WALKIE_EXPLORE_CAPTION_CLASSES")
        self.min_confidence = _envf("WALKIE_EXPLORE_CONFIDENCE_THRESHOLD", "0.3")
        self.crop_margin_px = _envi("WALKIE_EXPLORE_CROP_MARGIN_PX", "20")
        self.voxel_m = _envf("WALKIE_EXPLORE_VOXEL_M", "0.02")
        self.max_depth = _envf("WALKIE_EXPLORE_MAX_DEPTH_M", "4.0")
        self.min_points = _envi("WALKIE_EXPLORE_MIN_POINTS", "50")
        self.keep_rgb = _envb("WALKIE_EXPLORE_KEEP_RGB", "0")
        self.pose_mode = os.getenv("WALKIE_EXPLORE_POSE_MODE", "baseline").strip().lower()
        self.do_tsdf = _envb("WALKIE_EXPLORE_TSDF", "0")
        self.rebuild_every_n = _envi("WALKIE_EXPLORE_REBUILD_EVERY_N", "30")
        self.rebuild_min_interval = _envf("WALKIE_EXPLORE_REBUILD_MIN_INTERVAL_SEC", "20")
        self.build_window = _envi("WALKIE_EXPLORE_BUILD_WINDOW", "0")  # 0 = whole buffer
        self.assoc = dict(
            overlap_min=_envf("WALKIE_EXPLORE_ASSOC_OVERLAP_MIN", "0.2"),
            clip_min=_envf("WALKIE_EXPLORE_ASSOC_CLIP_MIN", "0.85"),
            cross_class_clip_min=_envf("WALKIE_EXPLORE_ASSOC_CROSS_CLASS_CLIP_MIN", "0.95"),
            max_dist_m=_envf("WALKIE_EXPLORE_ASSOC_MAX_DIST_M", "0.5"),
        )
        # Lift cleanup (forwarded to deproject_mask in the builder).
        self.lift = dict(
            erode_px=_envi("WALKIE_EXPLORE_MASK_ERODE_PX", "2"),
            edge_thresh=_envf("WALKIE_EXPLORE_DEPTH_EDGE_THRESH_M", "0.05"),
            edge_rel=_envf("WALKIE_EXPLORE_DEPTH_EDGE_REL", "0.0"),
            max_points=_envi("WALKIE_EXPLORE_MAX_POINTS_PER_OBJ", "2000"),
            sor_k=_envi("WALKIE_EXPLORE_SOR_K", "0"),
            sor_std_ratio=_envf("WALKIE_EXPLORE_SOR_STD_RATIO", "2.0"),
        )
        # Live scene feed: draw EACH captured frame's lifted detections to Rerun under
        # world/live (before the batch build), so you can watch the scene fill in live.
        self.viz_live = _envb("WALKIE_EXPLORE_VIZ_LIVE", "0")
        # On a cold start (empty store) build sooner so the Database agent isn't blind
        # for a full REBUILD_EVERY_N window; afterwards the normal cadence applies.
        self.first_build_n = max(1, min(self.rebuild_every_n,
                                        _envi("WALKIE_EXPLORE_FIRST_BUILD_N", "10")))
        # detection prompts: interested classes drive the open-vocab detector.
        self.detect_prompts = sorted(self.interested)

        buffer_dir = os.getenv("WALKIE_EXPLORE_BUFFER_DIR", "graph_buffer")
        snap_cap = _envi("WALKIE_EXPLORE_SNAPSHOT_CAP", "400")

        # The world model owns the scene store + relations + queries. Build one (objects
        # only) if none was injected, binding embed_text to the AI server for CLIP search.
        if world is None:
            embed_text = None
            if walkieAI is not None:
                def embed_text(q, _ai=walkieAI):
                    return _ai.image.embed_text(q)
            world = WalkieWorld(embed_text=embed_text, enable_people=False)
        self.world = world
        # Where to drop the TSDF map.npz (alongside the scene store's json/npy files).
        self._scene_dir = self.world.scene.store_dir

        self.buffer = SnapshotBuffer(buffer_dir, cap=snap_cap, keep_rgb=self.keep_rgb)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._build_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtx-build")
        self._build_future = None
        self._since_build = 0
        self._last_build_ts = 0.0

    # ------------------------------------------------------------------ logging
    def _log(self, msg: str) -> None:
        if _envb("WALKIE_EXPLORE_DEBUG_INGEST", "0") or _envb("WALKIE_EXPLORE_PERF", "0"):
            print(f"[explore] {msg}")

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Start the background capture thread (no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="RealtimeExplore", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread, drain the build worker, persist the world."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._build_exec.shutdown(wait=True, cancel_futures=False)
        try:
            self.world.persist()
        except Exception:  # noqa: BLE001
            pass

    def observe(self) -> list[ObjectNode]:
        """Manual path: capture one frame, build now (blocking), return the objects."""
        self._capture_once()
        self._build_once()
        return self.world.all_objects()

    # ------------------------------------------------------------------ capture loop
    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._capture_once()
            except Exception as e:  # noqa: BLE001 — one bad tick must not kill the thread
                self._log(f"capture tick failed: {e}")
            wait = max(0.0, self.interval - (time.monotonic() - t0))
            self._stop.wait(wait)

    def _capture_once(self) -> None:
        if self.walkie is None or self.walkieAI is None:
            return
        frame = self.walkie.capture_snapshot(log=self._log)
        if frame is None:
            return
        res = self.walkieAI.image.process(
            frame.img,
            detection={"prompts": self.detect_prompts or None, "return_mask": True},
            per_detection={
                "caption": {
                    "prompt_template": "Describe the {class_name}.",
                    "classes": sorted(self.caption_classes) or None,
                },
                "embed": True,
                "exclude_classes": sorted(self.exclude),
                "min_confidence": self.min_confidence,
                "crop_margin_px": self.crop_margin_px,
            },
        )
        detections = res.detection or []
        # Live perception.json — cheap per-detection centroid lift, decoupled from the graph.
        if self.snapshot_path is not None:
            self._write_perception(frame, detections)
        # Keep the live robot/camera markers fresh every tick (cheap), so they don't
        # freeze between the occasional batch builds; with VIZ_LIVE, also draw this
        # frame's lifted detections live (before the batch build folds them in).
        self._last_cam = frame.cam
        if self.viz is not None:
            try:
                self.viz.update_markers(robot_pose=frame.robot_pose, cam_pose=frame.cam)
                if self.viz_live:
                    self.viz.update_live(frame, detections, exclude=self.exclude)
            except Exception:  # noqa: BLE001 — viz is best-effort
                pass
        # Buffer the frame for the next batch build (only if liftable + has detections).
        if frame.has_geometry:
            self._buffer_frame(frame, detections)
            self._since_build += 1
            self._maybe_build()

    def _write_perception(self, frame, detections) -> None:
        from .snapshot import build_object_records, write_atomic
        try:
            pose = frame.robot_pose or {"heading": 0.0}
            robot_heading = float(pose.get("heading", 0.0))
            shown = [i for i, d in enumerate(detections)
                     if (d.class_name or "").lower() not in self.exclude]
            result = {}
            for j, i in enumerate(shown):
                d = detections[i]
                centroid = None
                if frame.has_geometry and getattr(d, "mask", None) is not None:
                    try:
                        pts = frame.mask_to_points(d.mask, max_points=400, sor_k=0)
                        if len(pts):
                            m = np.median(pts, axis=0)
                            centroid = (float(m[0]), float(m[1]), float(m[2]))
                    except Exception:  # noqa: BLE001
                        centroid = None
                result[j] = {"centroid": centroid, "caption": getattr(d, "caption", "") or ""}
            objs = build_object_records(
                [detections[i] for i in shown], result, frame.img.size, robot_heading
            )
            write_atomic(self.snapshot_path, {"ts": frame.ts, "objects": objs})
        except Exception as e:  # noqa: BLE001 — never take the loop down for a snapshot write
            self._log(f"perception.json write failed: {e}")

    def _buffer_frame(self, frame, detections) -> None:
        try:
            intr = frame.intr
            dets = []
            for d in detections:
                cls = (d.class_name or "").lower()
                if cls in self.exclude or getattr(d, "mask", None) is None:
                    continue
                # confidence is `float | None` on DetectedObject — coerce, then gate graph
                # entry on the threshold (the server's per_detection.min_confidence only
                # gates captioning/embedding, NOT which boxes the detector returns).
                conf = float(getattr(d, "confidence", 0.0) or 0.0)
                if conf < self.min_confidence:
                    continue
                dets.append(Detection(
                    class_name=d.class_name, class_id=getattr(d, "class_id", None),
                    conf=conf, bbox=tuple(int(v) for v in d.bbox),
                    caption=getattr(d, "caption", "") or "",
                    clip_emb=list(getattr(d, "embedding", None) or []),
                    mask=np.asarray(d.mask).astype(np.uint8),
                ))
            if not dets:
                return
            rgb = np.asarray(frame.img)[:, :, :3].astype(np.uint8) if self.keep_rgb else None
            # Coerce the pose dict to plain floats — the SDK may hand back numpy scalars,
            # which json.dump (the buffer index) can't serialize.
            robot_pose = (
                {k: float(v) for k, v in frame.robot_pose.items()} if frame.robot_pose else None
            )
            snap = Snapshot(
                ts=float(frame.ts), depth=np.asarray(frame.depth, np.float32),
                intr=(intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height),
                cam_R=np.asarray(frame.cam.R, float), cam_t=np.asarray(frame.cam.t, float),
                robot_pose=robot_pose, detections=dets, rgb=rgb,
            )
            self.buffer.append(snap)
        except Exception as e:  # noqa: BLE001
            self._log(f"buffer append failed: {e}")

    # ------------------------------------------------------------------ build worker
    def _maybe_build(self) -> None:
        # Build sooner on a cold start (empty store) so queries work within ~20 s.
        threshold = self.first_build_n if self.world.count() == 0 else self.rebuild_every_n
        if self._since_build < threshold:
            return
        if (time.monotonic() - self._last_build_ts) < self.rebuild_min_interval:
            return
        if self._build_future is not None and not self._build_future.done():
            return  # single-flight: a build is already running
        self._since_build = 0
        self._last_build_ts = time.monotonic()
        self._build_future = self._build_exec.submit(self._safe_build)

    def _safe_build(self) -> None:
        try:
            self._build_once()
        except Exception as e:  # noqa: BLE001
            self._log(f"build failed: {e}")

    def _build_once(self) -> None:
        if len(self.buffer) == 0:
            return
        with self.buffer.building():
            window = None if self.build_window <= 0 else self.build_window
            snaps = self.buffer.load_window(window)
        result = build_scene(
            snaps,
            pose_mode=self.pose_mode, do_tsdf=self.do_tsdf,
            voxel_m=self.voxel_m, max_depth=self.max_depth, min_points=self.min_points,
            **self.lift, **self.assoc, log=self._log,
        )
        # The world owns the merge -> derive relations -> install sequence (and its lock),
        # so a background build and a query caller can't tear the scene.
        self.world.observe_objects(result.observations)
        if result.structural_cloud is not None:
            self._last_structural = result.structural_cloud
            self._save_structural(result.structural_cloud)
        self._update_viz()

    def _save_structural(self, cloud) -> None:
        try:
            if self._scene_dir is not None:
                Path(self._scene_dir).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(Path(self._scene_dir) / "map.npz", points=np.asarray(cloud, np.float32))
        except Exception as e:  # noqa: BLE001
            self._log(f"map save failed: {e}")

    def _update_viz(self) -> None:
        """Redraw the whole scene graph after a build (rooms + objects + relations + markers)."""
        if self.viz is None:
            return
        try:
            robot_pose = None
            if self.walkie is not None:
                try:
                    robot_pose = self.walkie.status.get_position()
                except Exception:  # noqa: BLE001
                    robot_pose = None
            self.viz.update(self.world.scene, robot_pose=robot_pose,
                            cam_pose=self._last_cam, structural=self._last_structural,
                            rooms=self.world.rooms)
        except Exception:  # noqa: BLE001 — viz is best-effort
            pass

    # ------------------------------------------------------------------ query passthroughs
    # Delegate to the shared world so the Database agent (which still holds a producer
    # reference during migration) keeps working; removed once it queries ctx.world.
    def query_text(self, query, k=5, *, near=None, radius=None) -> list[ObjectNode]:
        return self.world.query_text(query, k, near=near, radius=radius)

    def query_near(self, center, radius) -> list[ObjectNode]:
        return self.world.query_near(center, radius)

    def recently_seen(self, limit=5) -> list[ObjectNode]:
        return self.world.recently_seen(limit)

    def all_objects(self) -> list[ObjectNode]:
        return self.world.all_objects()

    def get(self, node_id) -> Optional[ObjectNode]:
        return self.world.get(node_id)

    def relations_of(self, node_id) -> list[Relation]:
        return self.world.relations_of(node_id)

    def to_text_description(self) -> str:
        return self.world.to_text_description()

    # API-compat no-ops (the v1 facade exposed optional LLM refinement; unused here).
    def refine_captions(self, **_kw) -> int:
        return 0

    def infer_edges(self, **_kw) -> int:
        return 0

    def visualize(self) -> None:
        self._update_viz()
