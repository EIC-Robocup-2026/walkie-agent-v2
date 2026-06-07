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
        max_size: int = 640,
        jpeg_quality: int = 85,
        prompts: list[str] | None = None,
        return_mask: bool = False,
    ) -> list[DetectedObject]:
        """Detect objects in *image*.

        Args:
            image: A PIL Image (RGB) or BGR numpy array to run detection on.
            max_size: Longest edge is scaled down to this before sending.
                      YOLO resizes internally anyway, so full resolution adds
                      no accuracy but costs encoding time and bandwidth.
            jpeg_quality: JPEG quality (1-95). 85 is a good balance of speed
                          vs. detection accuracy.
            prompts: Optional open-vocabulary text prompts (noun phrases). Used
                     by concept providers (SAM3 / YOLOE); ignored by YOLO.
            return_mask: Request a segmentation mask per detection. When True,
                     each :class:`DetectedObject.mask` is a 2D uint8 {0,1} numpy
                     array (where the provider/model supports masks); otherwise
                     ``mask`` is ``None``.

        Returns:
            List of :class:`~services.object_detection.base.DetectedObject`.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        if isinstance(image, np.ndarray):
            # h, w = image.shape[:2]
            # if max(w, h) > max_size:
            #     scale = max_size / max(w, h)
            #     image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            image_bytes = _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
        else:
            # w, h = image.size
            # if max(w, h) > max_size:
            #     scale = max_size / max(w, h)
            #     image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            image_bytes = _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality)

        # Repeated "prompts" fields (the server also accepts a comma-separated
        # single value); "return_mask" as a string flag.
        form: list[tuple] = [("return_mask", "true" if return_mask else "false")]
        if prompts:
            form.extend(("prompts", p) for p in prompts)

        data = self._post_files(
            "/object-detection/detect",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data=form,
        )
        return [_deserialize_detection(d) for d in data]

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
