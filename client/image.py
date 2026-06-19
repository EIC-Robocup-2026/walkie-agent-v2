"""Unified HTTP client for the image-processing API — ``/image/*``.

One :class:`ImageClient` replaces the six former per-task sub-clients
(object detection, pose, caption, image-embed, face, appearance). The image
is uploaded **once** per call; :meth:`ImageClient.process` runs any combination
of tasks server-side and returns a typed :class:`ImageResult`. Thin
single-task helpers (:meth:`detect`, :meth:`caption`, :meth:`estimate_poses`,
:meth:`embed_image`, :meth:`embed_text`, :meth:`faces`, :meth:`appearance`)
wrap :meth:`process` for the common one-off call sites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _b64_to_mask, _numpy_to_bytes, _pil_to_bytes


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """A single detected object from an image."""

    mask: "np.ndarray | None"  # 2D uint8 (H, W) {0,1} segmentation mask, or None
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_ratio: float  # fraction of image area
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    # Set only when the request included a fused ``per_detection`` block.
    caption: str | None = None
    embedding: list[float] | None = None
    embedding_dim: int | None = None


@dataclass
class PoseKeypoint:
    """A single detected keypoint on a person."""

    x: float
    y: float
    confidence: float
    name: str
    index: int  # COCO keypoint index (0-16)


@dataclass
class PersonPose:
    """A detected person together with their pose keypoints."""

    bbox: tuple[int, int, int, int]  # (cx, cy, w, h)
    confidence: float
    keypoints: list[PoseKeypoint] = field(default_factory=list)


@dataclass
class FaceEmbedding:
    """A single detected face together with its recognition embedding."""

    bbox_xyxy: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    embedding: list[float] = field(default_factory=list)
    det_score: float = 0.0

    def area(self) -> int:
        """Pixel area of the bbox — used to pick the largest (nearest) face."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class ImageResult:
    """Combined result of one :meth:`ImageClient.process` call.

    Only the fields for requested tasks are populated; the rest stay ``None``.
    """

    detection: list[DetectedObject] | None = None
    caption: str | None = None
    pose: list[PersonPose] | None = None
    embed: list[float] | None = None       # whole-frame image embedding
    embed_dim: int | None = None
    face: list[FaceEmbedding] | None = None
    appearance: list[float] | None = None  # whole-(crop) appearance embedding


# ---------------------------------------------------------------------------
# Deserializers
# ---------------------------------------------------------------------------

def _deserialize_detection(d: dict) -> DetectedObject:
    emb = d.get("embedding")
    return DetectedObject(
        bbox=tuple(d["bbox"]),
        area_ratio=d["area_ratio"],
        class_id=d.get("class_id"),
        class_name=d.get("class_name"),
        confidence=d.get("confidence"),
        mask=_b64_to_mask(d.get("mask_b64")),
        caption=d.get("caption"),
        embedding=(list(emb) if emb is not None else None),
        embedding_dim=d.get("embedding_dim"),
    )


def _deserialize_pose(p: dict) -> PersonPose:
    return PersonPose(
        bbox=tuple(p["bbox"]),
        confidence=p["confidence"],
        keypoints=[
            PoseKeypoint(
                index=kp["index"], name=kp["name"],
                x=kp["x"], y=kp["y"], confidence=kp["confidence"],
            )
            for kp in p.get("keypoints", [])
        ],
    )


def _deserialize_face(f: dict) -> FaceEmbedding:
    return FaceEmbedding(
        bbox_xyxy=tuple(int(v) for v in f["bbox_xyxy"]),
        embedding=[float(x) for x in f.get("embedding", [])],
        det_score=float(f.get("det_score", 0.0)),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ImageClient(WalkieBaseClient):
    """Client for the unified ``/image`` blueprint.

    Example::

        res = walkieAI.image.process(frame, detection=True, caption=True, pose=True)
        for o in res.detection:
            print(o.class_name, o.confidence)
        print(res.caption)

        # Single-task helpers
        objs    = walkieAI.image.detect(frame, prompts=["chair"], return_mask=True)
        caption = walkieAI.image.caption(frame, prompt="Describe the table")
        poses   = walkieAI.image.estimate_poses(frame)
        vec     = walkieAI.image.embed_image(crop)
        qvec    = walkieAI.image.embed_text("a mug")
        faces   = walkieAI.image.faces(frame)
        attire  = walkieAI.image.appearance(person_crop)
    """

    # -- whole-pipeline entrypoint ------------------------------------------

    def process(
        self,
        image: Image.Image | np.ndarray,
        *,
        detection: bool | dict | None = None,
        caption: bool | str | dict | None = None,
        pose: bool = False,
        embed: bool = False,
        face: bool = False,
        appearance: bool = False,
        per_detection: dict | None = None,
        max_size: int | None = None,
        jpeg_quality: int = 85,
    ) -> ImageResult:
        """Run any combination of vision tasks on *image* in one request.

        Args:
            image: PIL Image (RGB) or BGR numpy array, uploaded once.
            detection: ``True`` / ``{"prompts": [...], "return_mask": bool}`` to
                run open-vocabulary detection (``None``/``False`` to skip).
            caption: ``True``, a prompt ``str``, or ``{"prompt": str}`` to caption
                the whole frame.
            pose / embed / face / appearance: flags to run those whole-frame tasks.
            per_detection: optional fused crop pipeline run after detection, e.g.
                ``{"caption": {"prompt_template": "Describe the {class_name}.",
                "classes": [...]}, "embed": True, "crop_margin_px": 20}``. Requires
                ``detection``; results attach onto each ``DetectedObject``.
            max_size: downscale the image's longest edge to this many pixels for
                transport; returned detection/pose/face geometry is scaled back to
                the input resolution (transparent to callers). Leave ``None`` for
                any path whose masks feed depth/3D projection.
            jpeg_quality: JPEG quality (1-95).

        Returns:
            :class:`ImageResult` with only the requested task fields populated.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        image_bytes, scale, w, h = _encode_once(image, max_size, jpeg_quality)

        spec: dict[str, Any] = {}
        det_opts = _norm_detection(detection)
        if det_opts is not None:
            spec["detection"] = det_opts
        cap_opts = _norm_caption(caption)
        if cap_opts is not None:
            spec["caption"] = cap_opts
        if pose:
            spec["pose"] = True
        if embed:
            spec["embed"] = True
        if face:
            spec["face"] = True
        if appearance:
            spec["appearance"] = True
        if per_detection:
            spec["per_detection"] = per_detection

        data = self._post_files(
            "/image/process",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"spec": json.dumps(spec)},
        )

        result = ImageResult()
        inv = (1.0 / scale) if scale != 1.0 else 1.0

        if "detection" in data:
            dets = [_deserialize_detection(d) for d in data["detection"]]
            if scale != 1.0:
                for o in dets:
                    _rescale_detection(o, inv, w, h)
            result.detection = dets
        if "caption" in data:
            result.caption = data["caption"]
        if "pose" in data:
            poses = [_deserialize_pose(p) for p in data["pose"]]
            if scale != 1.0:
                for p in poses:
                    _rescale_pose(p, inv)
            result.pose = poses
        if "embed" in data:
            result.embed = list(data["embed"]["embedding"])
            result.embed_dim = data["embed"].get("dim")
        if "face" in data:
            faces = [_deserialize_face(f) for f in data["face"]]
            if scale != 1.0:
                for f in faces:
                    f.bbox_xyxy = tuple(int(round(v * inv)) for v in f.bbox_xyxy)
            result.face = faces
        if "appearance" in data:
            result.appearance = list(data["appearance"]["embedding"])

        return result

    # -- single-task convenience helpers ------------------------------------

    def detect(
        self,
        image: Image.Image | np.ndarray,
        *,
        prompts: list[str] | None = None,
        return_mask: bool = False,
        max_size: int | None = None,
        jpeg_quality: int = 85,
    ) -> list[DetectedObject]:
        """Detect objects in *image* (see :meth:`process` for ``max_size`` semantics)."""
        res = self.process(
            image,
            detection={"prompts": prompts, "return_mask": return_mask},
            max_size=max_size,
            jpeg_quality=jpeg_quality,
        )
        return res.detection or []

    def caption(
        self,
        image: Image.Image | np.ndarray,
        *,
        prompt: str | None = None,
        jpeg_quality: int = 85,
    ) -> str:
        """Caption the whole *image*."""
        res = self.process(
            image,
            caption={"prompt": prompt} if prompt is not None else True,
            jpeg_quality=jpeg_quality,
        )
        return res.caption or ""

    def estimate_poses(
        self, image: Image.Image | np.ndarray, *, jpeg_quality: int = 85
    ) -> list[PersonPose]:
        """Estimate body poses for all persons in *image*."""
        res = self.process(image, pose=True, jpeg_quality=jpeg_quality)
        return res.pose or []

    def embed_image(
        self, image: Image.Image | np.ndarray, *, jpeg_quality: int = 85
    ) -> list[float]:
        """CLIP image embedding for the whole *image* (send a crop for an object)."""
        res = self.process(image, embed=True, jpeg_quality=jpeg_quality)
        return res.embed or []

    def embed_text(self, text: str) -> list[float]:
        """CLIP text embedding for *text* (joint space with :meth:`embed_image`)."""
        if not text:
            raise ValueError("text must be non-empty")
        data = self._post_json("/image/embed-text", payload={"text": text})
        return list(data["embedding"])

    def faces(
        self, image: Image.Image | np.ndarray, *, jpeg_quality: int = 85
    ) -> list[FaceEmbedding]:
        """Detect every face in *image* and return one embedding per face."""
        res = self.process(image, face=True, jpeg_quality=jpeg_quality)
        return res.face or []

    def appearance(
        self, image: Image.Image | np.ndarray, *, jpeg_quality: int = 85
    ) -> list[float]:
        """Embed a **person crop** into one appearance (attire/body) vector."""
        res = self.process(image, appearance=True, jpeg_quality=jpeg_quality)
        return res.appearance or []


# ---------------------------------------------------------------------------
# Spec / geometry helpers
# ---------------------------------------------------------------------------

def _norm_detection(val: bool | dict | None) -> dict | None:
    if val is None or val is False:
        return None
    if val is True:
        return {}
    if isinstance(val, dict):
        return {k: v for k, v in val.items() if v is not None}
    return {}


def _norm_caption(val: bool | str | dict | None) -> dict | None:
    if val is None or val is False:
        return None
    if val is True:
        return {}
    if isinstance(val, str):
        return {"prompt": val}
    if isinstance(val, dict):
        return val
    return {}


def _encode_once(
    image: Image.Image | np.ndarray, max_size: int | None, jpeg_quality: int
) -> tuple[bytes, float, int, int]:
    """JPEG-encode *image* once, optionally downscaling its longest edge to
    *max_size*. Returns ``(bytes, scale, orig_w, orig_h)`` where ``scale`` < 1
    only when a downscale happened (never upscales)."""
    scale = 1.0
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        if max_size and max_size > 0 and max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = cv2.resize(
                image, (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        image_bytes = _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
    else:
        w, h = image.size
        if max_size and max_size > 0 and max(w, h) > max_size:
            scale = max_size / max(w, h)
            image = image.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        image_bytes = _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
    return image_bytes, scale, w, h


def _rescale_detection(obj: DetectedObject, inv: float, w: int, h: int) -> None:
    x1, y1, x2, y2 = obj.bbox
    obj.bbox = (
        int(round(x1 * inv)), int(round(y1 * inv)),
        int(round(x2 * inv)), int(round(y2 * inv)),
    )
    if obj.mask is not None:
        obj.mask = cv2.resize(obj.mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _rescale_pose(pose: PersonPose, inv: float) -> None:
    cx, cy, bw, bh = pose.bbox
    pose.bbox = (
        int(round(cx * inv)), int(round(cy * inv)),
        int(round(bw * inv)), int(round(bh * inv)),
    )
    for kp in pose.keypoints:
        kp.x *= inv
        kp.y *= inv
