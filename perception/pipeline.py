"""One-frame perception pipeline.

Takes a single PIL frame and the collaborator dependencies, runs
detection → 3D lift → caption → embed in sequence, and returns a list
of :class:`Detection` records ready to feed :meth:`SceneStore.upsert`.

This function is the integration point between the AI server and the
SDK. It is pure with respect to the store: it doesn't write anything;
the loop calls ``store.upsert(...)`` itself for each returned record.
That makes the pipeline trivially unit-testable and lets the loop
decide what to do with errors at each stage.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PIL import Image

from .types import (
    Captioner,
    Detection,
    Detector,
    Embedder,
    PositionLifter,
)

_log = logging.getLogger("perception.pipeline")


def _xyxy_to_cxcywh(bbox: tuple[int, int, int, int]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        float((x1 + x2) / 2),
        float((y1 + y2) / 2),
        float(x2 - x1),
        float(y2 - y1),
    ]


def _crop(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = image.size
    x1c = max(0, min(int(x1), w - 1))
    y1c = max(0, min(int(y1), h - 1))
    x2c = max(x1c + 1, min(int(x2), w))
    y2c = max(y1c + 1, min(int(y2), h))
    return image.crop((x1c, y1c, x2c, y2c))


def process_frame(
    frame: Image.Image,
    *,
    detector: Detector,
    lifter: PositionLifter,
    captioner: Captioner,
    embedder: Embedder,
    frame_ts: Optional[float] = None,
    position_timeout: float = 2.0,
    min_confidence: float = 0.0,
    caption_per_object: bool = False,
) -> tuple[list[Detection], dict[str, float]]:
    """Run one frame through the perception stack.

    Returns ``(detections, latency_ms)`` where ``latency_ms`` is a
    per-stage breakdown so the loop can attach it to the tick report.

    Stages:
      1. ``detector.detect(frame)``
      2. ``lifter.bboxes_to_positions(...)`` — batched, one call per frame
      3. For each surviving detection: crop, embed image, caption
         (if ``caption_per_object``) or use one whole-frame caption

    Detections with no 3D position (``None`` returned or all-zero coords
    are *not* filtered — only ``None`` is dropped) and detections below
    ``min_confidence`` are skipped.
    """
    latency: dict[str, float] = {}
    frame_ts = frame_ts if frame_ts is not None else time.time()

    # --- Stage 1: detect ---
    t0 = time.perf_counter()
    raw_detections = list(detector.detect(frame))
    latency["detect"] = (time.perf_counter() - t0) * 1000

    if not raw_detections:
        latency["lift"] = 0.0
        latency["caption"] = 0.0
        latency["embed"] = 0.0
        return [], latency

    # --- Stage 2: lift to 3D in one batched call ---
    coords = [_xyxy_to_cxcywh(tuple(d.bbox)) for d in raw_detections]
    t0 = time.perf_counter()
    positions = lifter.bboxes_to_positions(coords, timeout=position_timeout)
    latency["lift"] = (time.perf_counter() - t0) * 1000

    if positions is None:
        _log.info(
            "pipeline.lift_failed n_dets=%d (lifter returned None — likely timeout)",
            len(raw_detections),
        )
        latency["caption"] = 0.0
        latency["embed"] = 0.0
        return [], latency

    # --- Stage 3: caption + embed for each survivor ---
    t_cap = 0.0
    t_emb = 0.0
    out: list[Detection] = []
    # Optional shared whole-frame caption — saves N inferences.
    shared_caption: Optional[str] = None
    if not caption_per_object:
        t0 = time.perf_counter()
        try:
            shared_caption = captioner.caption(frame)
        except Exception as e:  # pragma: no cover — caller-visible via tick report
            _log.warning("shared caption failed: %s", e)
            shared_caption = ""
        t_cap += time.perf_counter() - t0

    for det, pos in zip(raw_detections, positions):
        conf = float(det.confidence or 0.0)
        if conf < min_confidence:
            continue
        if pos is None or len(pos) < 3:
            continue
        bbox = tuple(int(v) for v in det.bbox)  # type: ignore[assignment]
        if len(bbox) != 4:  # safety net
            continue
        crop = _crop(frame, bbox)

        t0 = time.perf_counter()
        emb = embedder.embed_image(crop)
        t_emb += time.perf_counter() - t0

        if caption_per_object:
            t0 = time.perf_counter()
            try:
                caption = captioner.caption(
                    crop,
                    prompt=(
                        f"Describe the {det.class_name}." if det.class_name else None
                    ),
                )
            except Exception as e:  # pragma: no cover
                _log.warning("per-object caption failed: %s", e)
                caption = det.class_name or ""
            t_cap += time.perf_counter() - t0
        else:
            caption = shared_caption or ""

        out.append(
            Detection(
                class_name=det.class_name or "unknown",
                class_id=det.class_id,
                confidence=conf,
                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                embedding=tuple(emb),
                caption=caption,
                ts=frame_ts,
                frame_ref=None,
            )
        )

    latency["caption"] = t_cap * 1000
    latency["embed"] = t_emb * 1000
    return out, latency
