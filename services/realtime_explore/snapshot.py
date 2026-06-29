"""Live perception snapshot — the ``perception.json`` the agents read each turn.

The perception loop (:class:`~services.realtime_explore.service.RealtimeExplore`) folds every frame
into the durable scene graph, then writes this small, ephemeral view of *what's in front of
the robot right now* for ``PerceptionContextMiddleware`` to inject into the prompt. Object
positions/captions come from the same ingest pass (no extra detection/caption work); these
helpers only turn the detections + ingest result into the on-disk JSON.

Snapshot shape::

    {"ts": <epoch>, "objects": [{"class","bbox","conf","position_3d","heading","frame_h","frame_v","caption"}]}

(People/pose are deliberately absent — moving classes don't belong in the spatial view, and
live pose lookups stay in the Vision agent.)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


_CAMERA_HFOV_DEG = 110.0


def _bbox_heading(bbox, image_width: int, robot_heading: float) -> float:
    """World heading (rad) toward a bbox centre, from the robot heading + horizontal FOV."""
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
    """Coarse in-frame placement ("left"/"center"/… , "top"/"center"/…) of a bbox centre."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h_off = (cx - image_width / 2.0) / (image_width / 2.0)
    v_off = (cy - image_height / 2.0) / (image_height / 2.0)
    return _zone(h_off, _H_LABELS), _zone(v_off, _V_LABELS)


def build_object_records(
    detections,
    ingest_result: dict[int, dict],
    image_size: tuple[int, int],
    robot_heading: float,
) -> list[dict]:
    """Assemble the snapshot's per-object records from one frame's detections + ingest result.

    ``detections`` is the detector output (already scoped to the interested classes, so the
    snapshot only ever lists those); ``ingest_result`` is the ``{i: {"centroid", "caption"}}``
    dict :meth:`WalkieGraphsService.ingest_frame` returns, keyed by index into ``detections``.
    A detection with no 3D centroid (sparse/distant/no depth) gets ``position_3d=None`` rather
    than being dropped.
    """
    image_width, image_height = image_size
    records = []
    for i, o in enumerate(detections):
        r = ingest_result.get(i) or {}
        pos = r.get("centroid")
        caption = r.get("caption", "")
        frame_h, frame_v = _frame_position(o.bbox, image_width, image_height)
        records.append(
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
    return records


def write_atomic(output_path: str | Path, snap: dict) -> None:
    """Write ``snap`` as JSON via ``.tmp`` → ``os.replace`` so readers never see a half-write."""
    output_path = Path(output_path)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as f:
        json.dump(snap, f)
    os.replace(tmp, output_path)
