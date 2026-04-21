"""HTTP client for the Object Detection API — /object-detection/*"""

from __future__ import annotations

from PIL import Image

from services.object_detection.base import DetectedObject

from .base import WalkieBaseClient, _b64_to_pil, _pil_to_bytes


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

    def detect(self, image: Image.Image) -> list[DetectedObject]:
        """Detect objects in *image*.

        Args:
            image: A PIL Image (RGB) to run detection on.

        Returns:
            List of :class:`~services.object_detection.base.DetectedObject`.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        data = self._post_files(
            "/object-detection/detect",
            files={"image": ("image.png", _pil_to_bytes(image), "image/png")},
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
        cropped_image=_b64_to_pil(d.get("cropped_image_b64")),
        mask=None,  # mask_b64 is a PNG-encoded mask — skip reconstruction for now
    )
