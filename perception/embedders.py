"""Concrete ``Embedder`` implementations.

This file adapts the HTTP-backed CLIP service (``client.ImageEmbedClient``)
to the :class:`perception.types.Embedder` Protocol so the perception loop
can use it as a drop-in.

Why a separate adapter (rather than making ``ImageEmbedClient`` itself
satisfy the protocol): the HTTP client is a transport-level concern that
shouldn't know about ``model_name`` semantics; the Embedder Protocol is a
domain-level concern that shouldn't know about HTTP. Separating them keeps
the test layers crisp — the HTTP smoke test mocks the network, the
Embedder test mocks the client.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from client.image_embed import ImageEmbedClient


class RemoteCLIPEmbedder:
    """``Embedder`` Protocol implementation backed by ``/image-embed/*``.

    The walkie-ai-server's CLIP provider is ``openai/clip-vit-base-patch16``,
    so we hardcode that as ``model_name``. If the server provider ever
    changes, update this string — it lands in every record's
    ``embedding_model`` metadata field, used for forward-compat invalidation.
    """

    DEFAULT_MODEL_NAME = "clip-vit-base-patch16"

    def __init__(
        self,
        client: ImageEmbedClient,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._dim: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self._client.get_embedding_dim()
        return self._dim

    def embed_image(self, image: Image.Image) -> list[float]:
        return self._client.embed_image(image)

    def embed_text(self, text: str) -> list[float]:
        return self._client.embed_text(text)
