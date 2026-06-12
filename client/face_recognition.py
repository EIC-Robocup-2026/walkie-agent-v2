"""HTTP client for the Face Recognition API — /face-recognition/*

Stateless face detection + embedding. The walkie-ai-server ships an InsightFace
provider (``buffalo_l``: RetinaFace + ArcFace, 512-d **already L2-normalized**
``normed_embedding``). This client turns a frame into a list of detected faces,
each with an ``xyxy`` bbox, the embedding, and a detection score.

Enrollment, names, and matching all live on *this* side (``perception.PeopleStore``
+ the human sub-agent) — the server never stores or compares faces.

The request/response shape this client expects matches the server's
``api/routes/face_recognition.py`` 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _numpy_to_bytes, _pil_to_bytes


@dataclass
class FaceEmbedding:
    """A single detected face together with its recognition embedding."""

    bbox_xyxy: tuple[int, int, int, int]
    """Bounding box in ``(x1, y1, x2, y2)`` pixel coords (top-left, bottom-right)."""
    embedding: list[float] = field(default_factory=list)
    """L2-normalized recognition vector (``‖v‖₂ = 1``), constant dimension for
    every face and every call (512 for ``buffalo_l``). Match with cosine
    distance; never re-normalize."""
    det_score: float = 0.0
    """Face-detection confidence in ``[0, 1]``."""

    def area(self) -> int:
        """Pixel area of the bbox — used to pick the largest (nearest) face."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)


class FaceRecognitionClient(WalkieBaseClient):
    """Client for the ``/face-recognition`` blueprint.

    Example::

        client = FaceRecognitionClient()
        faces = client.embed(pil_image)          # list[FaceEmbedding]
        biggest = max(faces, key=lambda f: f.area())   # the person up front
        model = client.info()                    # {"model_name": ..., "dim": 512}
    """

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 60) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._cached_info: Optional[dict] = None

    def embed(
        self, image: Image.Image | np.ndarray, jpeg_quality: int = 85
    ) -> list[FaceEmbedding]:
        """Detect every face in *image* and return one embedding per face.

        Args:
            image: A PIL Image (RGB) or BGR numpy array.
            jpeg_quality: JPEG quality when *image* is a numpy array (1-95).

        Returns:
            One :class:`FaceEmbedding` per detected face; ``[]`` when none.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        data = self._post_files("/face-recognition/embed", files=self._image_files(image, jpeg_quality))
        return [_deserialize_face(f) for f in data]

    def info(self) -> dict:
        """Return ``{"model_name": str, "dim": int}`` (cached after first call).

        Used to stamp stored vectors with their producing model, so a future
        model swap is detectable.
        """
        if self._cached_info is None:
            self._cached_info = dict(self._get("/face-recognition/info"))
        return self._cached_info

    def get_embedding_dim(self) -> int:
        """Embedding dimension reported by the server (512 for ``buffalo_l``)."""
        return int(self.info()["dim"])

    def get_model_name(self) -> str:
        return str(self.info()["model_name"])

    def available_providers(self) -> list[str]:
        return self._get("/face-recognition/providers")

    @staticmethod
    def _image_files(image: Image.Image | np.ndarray, jpeg_quality: int) -> dict:
        if isinstance(image, np.ndarray):
            payload = _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
        else:
            payload = _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
        return {"image": ("image.jpg", payload, "image/jpeg")}


def _deserialize_face(f: dict) -> FaceEmbedding:
    return FaceEmbedding(
        bbox_xyxy=tuple(int(v) for v in f["bbox_xyxy"]),
        embedding=[float(x) for x in f.get("embedding", [])],
        det_score=float(f.get("det_score", 0.0)),
    )
