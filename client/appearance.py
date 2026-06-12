"""HTTP client for the Appearance (attire) re-ID API — /appearance/*

Stateless appearance embedding: a person crop in → one 512-d L2-normalized
OSNet re-ID vector out. Together with the face embedding this enables person
re-identification when the face is NOT visible (guest turned away, far, or
occluded) — the second modality of the fused people memory.

Pipeline design by Chalk (EIC team) — adopted from the `eic-human` subproject
(``eic_human/pipeline/appearance.py``: OSNet x1.0 via torchreid). The model
runs on walkie-ai-server; see ``docs/walkie_ai_server_appearance_service.md``
for the server-side handoff spec.

Enrollment and matching live on *this* side (``perception.PeopleStore``) —
the server never stores or compares embeddings.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _numpy_to_bytes, _pil_to_bytes


class AppearanceClient(WalkieBaseClient):
    """Client for the ``/appearance`` blueprint.

    Example::

        client = AppearanceClient()
        emb = client.embed(person_crop)      # list[float], 512-d unit length
        model = client.info()                # {"model_name": ..., "dim": 512}
    """

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 60) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._cached_info: Optional[dict] = None

    def embed(
        self, image: Image.Image | np.ndarray, jpeg_quality: int = 85
    ) -> list[float]:
        """Embed a person crop into one appearance (attire/body) vector.

        Send the **person's bounding-box crop**, not the whole frame — the
        embedding describes whatever is in the image, so a full frame mixes
        the background into the identity.

        Args:
            image: A PIL Image (RGB) or BGR numpy array of one person.
            jpeg_quality: JPEG quality when *image* is a numpy array (1-95).

        Returns:
            The L2-normalized embedding (constant dimension, 512 for OSNet).

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        data = self._post_files("/appearance/embed", files=self._image_files(image, jpeg_quality))
        return [float(x) for x in data["embedding"]]

    def info(self) -> dict:
        """Return ``{"model_name": str, "dim": int}`` (cached after first call)."""
        if self._cached_info is None:
            self._cached_info = dict(self._get("/appearance/info"))
        return self._cached_info

    def get_embedding_dim(self) -> int:
        """Embedding dimension reported by the server (512 for OSNet x1.0)."""
        return int(self.info()["dim"])

    def get_model_name(self) -> str:
        return str(self.info()["model_name"])

    @staticmethod
    def _image_files(image: Image.Image | np.ndarray, jpeg_quality: int) -> dict:
        if isinstance(image, np.ndarray):
            payload = _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
        else:
            payload = _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality)
        return {"image": ("image.jpg", payload, "image/jpeg")}
