from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

from interfaces.walkie_interface import WalkieInterface


_CAMERA_HFOV_DEG = 110.0


def _xyxy_to_cxcywh(bbox) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1)]


def _bbox_heading(bbox, image_width: int, robot_heading: float) -> float:
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    offset = (cx - image_width / 2.0) / (image_width / 2.0)
    angle_offset_rad = math.radians(offset * (_CAMERA_HFOV_DEG / 2.0))
    return robot_heading - angle_offset_rad


def _zone(offset: float, labels: tuple[str, str, str, str, str]) -> str:
    if offset < -0.6:
        return labels[0]
    if offset < -0.2:
        return labels[1]
    if offset < 0.2:
        return labels[2]
    if offset < 0.6:
        return labels[3]
    return labels[4]


_H_LABELS = ("left", "slightly left", "center", "slightly right", "right")
_V_LABELS = ("top", "slightly top", "center", "slightly bottom", "bottom")


def _frame_position(bbox, image_width: int, image_height: int) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h_off = (cx - image_width / 2.0) / (image_width / 2.0)
    v_off = (cy - image_height / 2.0) / (image_height / 2.0)
    return _zone(h_off, _H_LABELS), _zone(v_off, _V_LABELS)


# COCO indices used to summarize "arm raised"
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6
_LEFT_WRIST, _RIGHT_WRIST = 9, 10

_CAPTION_IMAGE_MARGIN = 10
def _crop_image(img, bbox):
    x1, y1, x2, y2 = bbox
    return img.crop((x1 - _CAPTION_IMAGE_MARGIN, y1 - _CAPTION_IMAGE_MARGIN, x2 + _CAPTION_IMAGE_MARGIN, y2 + _CAPTION_IMAGE_MARGIN))

def _summarize_pose(pose) -> str:
    kpts = {kp.index: kp for kp in pose.keypoints}
    flags = []
    ls, lw = kpts.get(_LEFT_SHOULDER), kpts.get(_LEFT_WRIST)
    rs, rw = kpts.get(_RIGHT_SHOULDER), kpts.get(_RIGHT_WRIST)
    if ls and lw and lw.confidence > 0.3 and ls.confidence > 0.3 and lw.y < ls.y:
        flags.append("left arm raised")
    if rs and rw and rw.confidence > 0.3 and rs.confidence > 0.3 and rw.y < rs.y:
        flags.append("right arm raised")
    return ", ".join(flags) if flags else "standing"


class PerceptionService(threading.Thread):
    """Background thread: writes the current world snapshot to perception.json.

    Runs only during the Ready stage. Each tick:
      - captures an image
      - runs object detection + image caption + pose estimation in sequence
        (the AI server may not parallelize, so keep it simple)
      - lifts bboxes to 3D positions
      - writes JSON atomically (.tmp → rename)

    The JSON shape:
        {
          "ts": <epoch>,
          "objects": [{"class","bbox","conf","position_3d","caption"}],
          "people":  [{"bbox","conf","pose_summary"}]
        }
    """

    def __init__(
        self,
        walkieAI,
        walkie: WalkieInterface,
        output_path: str | Path,
        *,
        interval: float = 2.0,
        graphs=None,
        caption_objects: bool = True,
        caption_filter: list[str] = [],
        position_timeout: float = 2.0,
        verbose: bool = True,
    ) -> None:
        super().__init__(daemon=True, name="PerceptionService")
        self.walkieAI = walkieAI
        self.walkie = walkie
        self.output_path = Path(output_path)
        self.interval = interval
        # The scene-graph facade. Perception runs the single (open-vocab, masked) detection
        # and hands each frame to graphs.ingest_frame, which owns geometry/caption/embed/
        # upsert and returns per-object world centroids + captions for the snapshot.
        self.graphs = graphs
        self._graphs_enabled = os.getenv("WALKIE_GRAPHS_ENABLED", "1").lower() in (
            "1",
            "true",
            "yes",
        )
        # Open-vocabulary prompt list shared with the graph (its interested classes). Masks
        # — needed for the 3D centroids — come from the prompted provider, so this is also
        # what makes position_3d/caption available. None => detector default vocabulary.
        self._prompts = graphs.detection_prompts() if graphs is not None else None
        # caption_objects/caption_filter/position_timeout are retained for signature
        # stability but unused: captioning + 3D both live in graphs.ingest_frame now.
        self.caption_objects = caption_objects
        self.caption_filter = caption_filter
        self.position_timeout = position_timeout
        self.verbose = verbose
        self._stop_event = threading.Event()

    def stop_and_join(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        self.join(timeout)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[perception] {msg}")

    def run(self) -> None:
        self._log(f"started (interval={self.interval}s, output={self.output_path})")
        while not self._stop_event.is_set():
            try:
                snap = self._snapshot()
                self._write_atomic(snap)
            except Exception as e:  # noqa: BLE001
                self._log(f"tick error: {e}")
            self._stop_event.wait(self.interval)
        self._log("stopped.")

    def _depth(self):
        """Latest aligned depth frame (H×W float32 metres, NaN invalid), or None.

        Captured back-to-back with the colour image so a detection's mask deprojects
        against the matching depth. None when depth is unavailable — positions then
        degrade to ``null`` while the rest of the snapshot still writes.
        """
        try:
            return self.walkie.robot.camera.get_depth()
        except Exception as e:  # noqa: BLE001
            self._log(f"depth unavailable: {e}")
            return None

    def _snapshot(self) -> dict:
        img = self.walkie.camera.capture_pil()
        depth = self._depth()  # back-to-back with the image so mask↔depth align
        image_width, image_height = img.size
        pose = self.walkie.status.get_position() or {"x": 0.0, "y": 0.0, "heading": 0.0}
        robot_heading = float(pose.get("heading", 0.0))

        # One open-vocabulary, masked detection per frame — the same call the scene graph
        # used to make on its own. Masks feed the 3D deprojection; prompts scope it to the
        # task vocabulary (people come from the separate pose_estimation path below).
        objects = self.walkieAI.object_detection.detect(
            img, prompts=self._prompts, return_mask=True
        )
        people = self.walkieAI.pose_estimation.estimate(img)

        # Hand the frame to the graph: it owns geometry/caption/embed/upsert and returns
        # per-object world centroids + captions (keyed by index into `objects`). Perception
        # just consumes that for the snapshot — so detection and captioning run once total.
        graph_result: dict[int, dict] = {}
        if self.graphs is not None and self._graphs_enabled:
            try:
                graph_result = self.graphs.ingest_frame(img, objects, depth) or {}
            except Exception as e:  # noqa: BLE001 — a graph hiccup must not stop the snapshot
                self._log(f"graphs ingest failed: {e}")
                graph_result = {}

        obj_records = []
        for i, o in enumerate(objects):
            r = graph_result.get(i) or {}
            pos = r.get("centroid")
            caption = r.get("caption", "")
            frame_h, frame_v = _frame_position(o.bbox, image_width, image_height)
            obj_records.append(
                {
                    "class": o.class_name,
                    "bbox": list(o.bbox),
                    "conf": o.confidence,
                    "position_3d": list(pos) if pos else None,
                    "heading": _bbox_heading(o.bbox, image_width, robot_heading),
                    "frame_h": frame_h,
                    "frame_v": frame_v,
                    "caption": caption,
                }
            )

        people_records = []
        for p in people:
            frame_h, frame_v = _frame_position(p.bbox, image_width, image_height)
            people_records.append(
                {
                    "bbox": list(p.bbox),
                    "conf": p.confidence,
                    "heading": _bbox_heading(p.bbox, image_width, robot_heading),
                    "frame_h": frame_h,
                    "frame_v": frame_v,
                    "pose_summary": _summarize_pose(p),
                }
            )

        return {
            "ts": time.time(),
            "objects": obj_records,
            "people": people_records,
        }

    def _write_atomic(self, snap: dict) -> None:
        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as f:
            json.dump(snap, f)
        os.replace(tmp, self.output_path)
