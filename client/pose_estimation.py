"""HTTP client for the Pose Estimation API — /pose-estimation/*"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _b64_to_pil, _numpy_to_bytes, _pil_to_bytes

@dataclass
class PersonPose:
    """A detected person together with their pose keypoints."""

    bbox: tuple[int, int, int, int]
    """Bounding box in ``(cx, cy, w, h)`` format (matching the existing YOLO
    object-detection convention used elsewhere in this project)."""
    confidence: float
    """Person detection confidence."""
    keypoints: list[PoseKeypoint] = field(default_factory=list)
    """List of 17 COCO keypoints."""

@dataclass
class PoseKeypoint:
    """A single detected keypoint on a person."""

    x: float
    """Pixel x-coordinate."""
    y: float
    """Pixel y-coordinate."""
    confidence: float
    """Detection confidence in [0, 1]."""
    name: str
    """Human-readable name (e.g. ``'nose'``, ``'left_shoulder'``)."""
    index: int
    """COCO keypoint index (0-16)."""

class PoseEstimationClient(WalkieBaseClient):
    """Client for the ``/pose-estimation`` blueprint.

    Mirrors the interface of :class:`services.pose_estimation.PoseEstimation`.
    Returns real :class:`~services.pose_estimation.base.PersonPose` dataclass
    instances with fully populated :class:`~services.pose_estimation.base.PoseKeypoint`
    lists — not raw dicts.

    Example::

        client = PoseEstimationClient()
        poses = client.estimate(pil_image)
        for person in poses:
            nose = next(kp for kp in person.keypoints if kp.name == "nose")
            print(nose.x, nose.y, nose.confidence)
    """

    def estimate(
        self, image: Image.Image | np.ndarray, jpeg_quality: int = 85
    ) -> list[PersonPose]:
        """Estimate body poses for all persons detected in *image*.

        Args:
            image: A PIL Image (RGB) or BGR numpy array to run pose estimation on.
            jpeg_quality: JPEG quality when *image* is a numpy array (1-95).

        Returns:
            List of :class:`~services.pose_estimation.base.PersonPose`.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        if isinstance(image, np.ndarray):
            image_bytes = _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
        else:
            files = {"image": ("image.jpg", _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality), "image/jpeg")}

        data = self._post_files("/pose-estimation/estimate", files=files)
        return [_deserialize_pose(p) for p in data]

    def available_providers(self) -> list[str]:
        """List all providers registered on the server."""
        return self._get("/pose-estimation/providers")


def _deserialize_keypoint(kp: dict) -> PoseKeypoint:
    return PoseKeypoint(
        index=kp["index"],
        name=kp["name"],
        x=kp["x"],
        y=kp["y"],
        confidence=kp["confidence"],
    )


def _deserialize_pose(p: dict) -> PersonPose:
    return PersonPose(
        bbox=tuple(p["bbox"]),
        confidence=p["confidence"],
        keypoints=[_deserialize_keypoint(kp) for kp in p.get("keypoints", [])],
    )
