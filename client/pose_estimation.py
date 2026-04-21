"""HTTP client for the Pose Estimation API — /pose-estimation/*"""

from __future__ import annotations

from PIL import Image

from services.pose_estimation.base import PersonPose, PoseKeypoint

from .base import WalkieBaseClient, _b64_to_pil, _pil_to_bytes


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

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        """Estimate body poses for all persons detected in *image*.

        Args:
            image: A PIL Image (RGB) to run pose estimation on.

        Returns:
            List of :class:`~services.pose_estimation.base.PersonPose`.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        data = self._post_files(
            "/pose-estimation/estimate",
            files={"image": ("image.png", _pil_to_bytes(image), "image/png")},
        )
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
        cropped_image=_b64_to_pil(p.get("cropped_image_b64")),
    )
