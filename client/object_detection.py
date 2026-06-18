"""HTTP client for the Object Detection API — /object-detection/*"""

from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _b64_to_mask, _numpy_to_bytes, _pil_to_bytes

@dataclass
class DetectedObject:
    """A single detected object from an image."""

    mask: "np.ndarray | None"  # 2D uint8 (H, W) {0,1} segmentation mask, or None
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_ratio: float  # fraction of image area
    # Optional: set by providers that output class and confidence (e.g. YOLO)
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None

class ObjectDetectionClient(WalkieBaseClient):
    """Client for the ``/object-detection`` blueprint.

    Mirrors the interface of :class:`services.object_detection.ObjectDetection`.
    Returns real :class:`~services.object_detection.base.DetectedObject`
    dataclass instances, not raw dicts.

    Example::

        client = ObjectDetectionClient()
        detections = client.detect(pil_image)
        for obj in detections:
            print(obj.class_name, obj.confidence, obj.bbox)
    """

    def detect(
        self,
        image: Image.Image | np.ndarray,
        max_size: int | None = None,
        jpeg_quality: int = 85,
        prompts: list[str] | None = None,
        return_mask: bool = False,
        provider: str | None = None,
    ) -> list[DetectedObject]:
        """Detect objects in *image*.

        Args:
            image: A PIL Image (RGB) or BGR numpy array to run detection on.
            max_size: When set to a positive int, the image's longest edge is
                      downscaled to this many pixels before sending (cutting
                      JPEG-encode + transfer time). The returned bbox and mask
                      are scaled back to the ORIGINAL input resolution, so this
                      is transparent to callers — coords are always in input-image
                      pixel space. ``None`` (default) / ``<= 0`` sends full
                      resolution and never resizes. Leave it ``None`` for any
                      path whose masks feed depth/3D projection (small objects
                      and mask precision suffer from downscaling).
            jpeg_quality: JPEG quality (1-95). 85 is a good balance of speed
                          vs. detection accuracy.
            prompts: Optional open-vocabulary text prompts (noun phrases). Used
                     by concept providers (SAM3 / YOLOE); ignored by YOLO.
            return_mask: Request a segmentation mask per detection. When True,
                     each :class:`DetectedObject.mask` is a 2D uint8 {0,1} numpy
                     array (where the provider/model supports masks); otherwise
                     ``mask`` is ``None``.
            provider: Optional provider/model name. Only sent when set; requires
                     the server to support per-call provider selection (a no-op
                     otherwise).

        Returns:
            List of :class:`~services.object_detection.base.DetectedObject`.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        # Downscale for transport, remembering the factor so we can restore the
        # detector's bbox/mask to the original resolution afterward. ``scale`` < 1
        # only when an explicit max_size is smaller than the longest edge; we
        # never upscale.
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

        # Repeated "prompts" fields (the server also accepts a comma-separated
        # single value); "return_mask" as a string flag.
        form: list[tuple] = [("return_mask", "true" if return_mask else "false")]
        if prompts:
            form.extend(("prompts", p) for p in prompts)
        if provider:
            form.append(("provider", provider))

        data = self._post_files(
            "/object-detection/detect",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data=form,
        )
        results = [_deserialize_detection(d) for d in data]
        if scale != 1.0:
            # Map the detector's downscaled bbox/mask back to the input image's
            # coords so every caller sees input-resolution geometry. area_ratio
            # is a ratio of image area and so is scale-invariant — left as-is.
            inv = 1.0 / scale
            for obj in results:
                x1, y1, x2, y2 = obj.bbox
                obj.bbox = (
                    int(round(x1 * inv)), int(round(y1 * inv)),
                    int(round(x2 * inv)), int(round(y2 * inv)),
                )
                if obj.mask is not None:
                    obj.mask = cv2.resize(
                        obj.mask, (w, h), interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
        return results

    def available_providers(self) -> list[str]:
        """List all providers registered on the server."""
        return self._get("/object-detection/providers")


def _deserialize_detection(d: dict) -> DetectedObject:
    bbox = tuple(d["bbox"])  # (x1, y1, x2, y2)
    return DetectedObject(
        bbox=bbox,
        area_ratio=d["area_ratio"],
        class_id=d.get("class_id"),
        class_name=d.get("class_name"),
        confidence=d.get("confidence"),
        # Decode the base64 PNG mask into a 2D uint8 {0,1} array (None when the
        # server did not return one, e.g. return_mask=false).
        mask=_b64_to_mask(d.get("mask_b64")),
    )
