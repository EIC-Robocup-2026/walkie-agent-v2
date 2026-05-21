"""HTTP client for the Image Embedding API — /image-embed/*

The walkie-ai-server ships a CLIP provider (``openai/clip-vit-base-patch16``,
512-dim) but the blueprint is **currently commented out** in
``walkie-ai-server/api/__init__.py``. Until the server team re-enables it,
this client will get 404s. Once enabled, no further changes here are needed.

The request/response shape this client expects is pinned by
``tests/perception/test_smoke_image_embed.py`` and matches the server's
``api/routes/image_embed.py`` 1:1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .base import WalkieBaseClient, _numpy_to_bytes, _pil_to_bytes


class ImageEmbedClient(WalkieBaseClient):
    """Client for the ``/image-embed`` blueprint.

    Provides joint image/text embeddings in a shared CLIP space.

    Example::

        client = ImageEmbedClient()
        img_vec = client.embed_image(pil_image)   # list[float], len == dim
        txt_vec = client.embed_text("a mug")
        score = client.similarity(pil_image, "a mug")  # float in [-1, 1]
    """

    def embed_image(
        self,
        image: Image.Image | np.ndarray,
        jpeg_quality: int = 85,
    ) -> list[float]:
        """Compute the CLIP image embedding for *image*.

        Returns a normalized vector as ``list[float]``. The vector length
        is the model's projection dim (512 for ViT-B/16); read it via
        :py:meth:`get_embedding_dim` if you need to allocate up-front.
        """
        files = self._image_files(image, jpeg_quality)
        data = self._post_files("/image-embed/embed-image", files=files)
        return list(data["embedding"])

    def embed_text(self, text: str) -> list[float]:
        """Compute the CLIP text embedding for *text*."""
        if not text:
            raise ValueError("text must be non-empty")
        data = self._post_json("/image-embed/embed-text", payload={"text": text})
        return list(data["embedding"])

    def similarity(
        self,
        image: Image.Image | np.ndarray,
        text: str,
        jpeg_quality: int = 85,
    ) -> float:
        """Cosine similarity between an image and a text query.

        Equivalent to ``cos(embed_image(image), embed_text(text))`` but
        served in a single HTTP round-trip.
        """
        if not text:
            raise ValueError("text must be non-empty")
        files = self._image_files(image, jpeg_quality)
        data = self._post_files(
            "/image-embed/similarity", files=files, data={"text": text}
        )
        return float(data["similarity"])

    def get_embedding_dim(self) -> int:
        """Fetch the model's embedding dimension by way of a tiny probe.

        The server doesn't expose dim as a standalone route, so we embed a
        1×1 black pixel and read ``dim`` from the response. Result is
        cached for subsequent calls.
        """
        if self._cached_dim is not None:
            return self._cached_dim
        probe = Image.new("RGB", (1, 1), color=(0, 0, 0))
        files = self._image_files(probe, 85)
        data = self._post_files("/image-embed/embed-image", files=files)
        self._cached_dim = int(data["dim"])
        return self._cached_dim

    def available_providers(self) -> list[str]:
        return self._get("/image-embed/providers")

    # ------------------------------------------------------------------

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 60) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self._cached_dim: Optional[int] = None

    @staticmethod
    def _image_files(image: Image.Image | np.ndarray, jpeg_quality: int) -> dict:
        if isinstance(image, np.ndarray):
            return {
                "image": (
                    "image.jpg",
                    _numpy_to_bytes(image, fmt="JPEG", quality=jpeg_quality),
                    "image/jpeg",
                )
            }
        return {
            "image": (
                "image.jpg",
                _pil_to_bytes(image, fmt="JPEG", quality=jpeg_quality),
                "image/jpeg",
            )
        }
