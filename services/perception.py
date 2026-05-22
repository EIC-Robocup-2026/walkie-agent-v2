from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from interfaces.walkie_interface import WalkieInterface


def _xyxy_to_cxcywh(bbox) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1)]


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
        self.caption_objects = caption_objects
        self.caption_filter = caption_filter
        self.position_timeout = position_timeout
        self.verbose = verbose
        self._stop = threading.Event()

    def stop_and_join(self, timeout: float | None = None) -> None:
        self._stop.set()
        self.join(timeout)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[perception] {msg}")

    def run(self) -> None:
        self._log(f"started (interval={self.interval}s, output={self.output_path})")
        while not self._stop.is_set():
            try:
                snap = self._snapshot()
                self._write_atomic(snap)
            except Exception as e:  # noqa: BLE001
                self._log(f"tick error: {e}")
            self._stop.wait(self.interval)
        self._log("stopped.")

    def _snapshot(self) -> dict:
        img = self.walkie.camera.capture_pil()
        objects = self.walkieAI.object_detection.detect(img)
        people = self.walkieAI.pose_estimation.estimate(img)

        # Lift bboxes to 3D in one batched call.
        positions: list = []
        if objects:
            try:
                # print(objects)
                positions = (
                    self.walkie.tools.bboxes_to_positions(
                        [_xyxy_to_cxcywh(o.bbox) for o in objects],
                        timeout=self.position_timeout,
                    )
                    or []
                )
                # print(positions)
            except Exception as e:  # noqa: BLE001
                self._log(f"bboxes_to_positions failed: {e}")
                positions = []

        obj_records = []
        for i, o in enumerate(objects):
            pos = positions[i] if i < len(positions) and positions[i] else None
            caption = ""
            if self.caption_objects:
                try:
                    if o.class_name not in self.caption_filter:
                        continue
                    cropped_img = _crop_image(img, o.bbox)
                    caption = self.walkieAI.image_caption.caption(
                        cropped_img, prompt=f"Describe the {o.class_name} in detail."
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"Captioning error: {e}")
                    caption = ""
            obj_records.append(
                {
                    "class": o.class_name,
                    "bbox": list(o.bbox),
                    "conf": o.confidence,
                    "position_3d": list(pos) if pos else None,
                    "caption": caption,
                }
            )

        people_records = []
        for p in people:
            people_records.append(
                {
                    "bbox": list(p.bbox),
                    "conf": p.confidence,
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
