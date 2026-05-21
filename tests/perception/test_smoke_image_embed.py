"""Smoke: document that the embedding endpoint is NOT exposed by default.

walkie-ai-server has a CLIP provider implementation in
``services/image_embed/`` and a route file ``api/routes/image_embed.py``,
but the blueprint registration is commented out in ``api/__init__.py``:

    # app.register_blueprint(image_embed.bp)

The perception subsystem needs image+text embeddings for scene queries.
Until the server re-enables ``/image-embed/*``, we either:

  (a) get the server team to register the blueprint (preferred), or
  (b) fall back to embedding the caption *text* via a separate model
      (e.g. sentence-transformers) on the agent side.

This smoke test pins the contract we will rely on once the endpoint is
live: the wrapper POSTs to ``/image-embed/embed-image`` and returns a
``{"embedding": list[float], "dim": int}`` payload. We assert the shape
against the existing server code so when the blueprint is enabled we
do not silently get the wrong response format.
"""

from __future__ import annotations

from unittest.mock import patch

from client.base import WalkieBaseClient


class _EmbedClient(WalkieBaseClient):
    """Provisional client mirroring the (currently disabled) /image-embed routes.

    This will be lifted into ``client/image_embed.py`` once the server
    blueprint is registered. The shape is taken directly from
    ``walkie-ai-server/api/routes/image_embed.py`` so the contract is
    pinned now and won't drift.
    """

    def embed_image_bytes(self, image_bytes: bytes) -> dict:
        return self._post_files(
            "/image-embed/embed-image",
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
        )

    def embed_text(self, text: str) -> dict:
        return self._post_json("/image-embed/embed-text", payload={"text": text})


def test_embed_image_shape_contract(tiny_jpeg_bytes, fake_success_response):
    client = _EmbedClient(base_url="http://stub")
    payload = {"embedding": [0.1] * 512, "dim": 512}
    with patch.object(client._session, "post", return_value=fake_success_response(payload)):
        out = client.embed_image_bytes(tiny_jpeg_bytes)

    assert set(out.keys()) == {"embedding", "dim"}
    assert out["dim"] == 512
    assert len(out["embedding"]) == 512


def test_embed_text_shape_contract(fake_success_response):
    client = _EmbedClient(base_url="http://stub")
    payload = {"embedding": [0.0] * 512, "dim": 512}
    with patch.object(client._session, "post", return_value=fake_success_response(payload)):
        out = client.embed_text("a coffee mug on a desk")

    assert out["dim"] == 512
    assert len(out["embedding"]) == 512
