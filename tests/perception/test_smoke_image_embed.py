"""Smoke + adapter tests for the CLIP image-embed client.

The wrapped HTTP endpoint (``/image-embed/*``) is currently disabled on
walkie-ai-server — see ``api/__init__.py:16``. To use this in production
the server team must register the blueprint and redeploy.

These tests mock the network so the client and the
:class:`perception.RemoteCLIPEmbedder` adapter can be validated without a
live server. The mocked payload shape is taken straight from
``walkie-ai-server/api/routes/image_embed.py`` so the contract stays
pinned.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from PIL import Image

from client.image_embed import ImageEmbedClient
from perception import RemoteCLIPEmbedder


# ---------------------------------------------------------------------------
# HTTP client smoke tests
# ---------------------------------------------------------------------------


def test_embed_image_returns_list_of_floats(tiny_pil_image, fake_success_response):
    client = ImageEmbedClient(base_url="http://stub")
    payload = {"embedding": [0.1] * 512, "dim": 512}
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ) as post:
        out = client.embed_image(tiny_pil_image)

    assert isinstance(out, list)
    assert len(out) == 512
    assert post.call_args.args[0] == "http://stub/image-embed/embed-image"
    assert "image" in post.call_args.kwargs["files"]


def test_embed_text_returns_list_of_floats(fake_success_response):
    client = ImageEmbedClient(base_url="http://stub")
    payload = {"embedding": [0.0] * 512, "dim": 512}
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ) as post:
        out = client.embed_text("a coffee mug on a desk")

    assert len(out) == 512
    assert post.call_args.args[0] == "http://stub/image-embed/embed-text"


def test_embed_text_rejects_empty_string():
    client = ImageEmbedClient(base_url="http://stub")
    with pytest.raises(ValueError, match="non-empty"):
        client.embed_text("")


def test_similarity_returns_float(tiny_pil_image, fake_success_response):
    client = ImageEmbedClient(base_url="http://stub")
    payload = {"similarity": 0.72}
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ) as post:
        score = client.similarity(tiny_pil_image, "mug")

    assert score == pytest.approx(0.72)
    assert post.call_args.args[0] == "http://stub/image-embed/similarity"
    assert post.call_args.kwargs["data"] == {"text": "mug"}


def test_get_embedding_dim_caches_after_first_call(
    tiny_pil_image, fake_success_response
):
    """The probe must hit the server only once; subsequent calls hit the cache."""
    client = ImageEmbedClient(base_url="http://stub")
    payload = {"embedding": [0.0] * 512, "dim": 512}
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ) as post:
        dim1 = client.get_embedding_dim()
        dim2 = client.get_embedding_dim()

    assert dim1 == 512 == dim2
    assert post.call_count == 1


# ---------------------------------------------------------------------------
# Embedder Protocol adapter
# ---------------------------------------------------------------------------


def test_remote_clip_embedder_satisfies_protocol(fake_success_response):
    """RemoteCLIPEmbedder must expose model_name, dim, embed_image, embed_text."""
    client = ImageEmbedClient(base_url="http://stub")
    embedder = RemoteCLIPEmbedder(client)

    payload = {"embedding": [0.5] * 512, "dim": 512}
    img = Image.new("RGB", (4, 4), color=(0, 128, 255))
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ):
        # model_name comes from the adapter, not the network
        assert embedder.model_name == "clip-vit-base-patch16"
        # dim lazy-loaded; subsequent reads cached
        assert embedder.dim == 512
        # actual embedding round-trip
        vec_img = embedder.embed_image(img)
        vec_txt = embedder.embed_text("mug")

    assert len(vec_img) == 512
    assert len(vec_txt) == 512


def test_remote_clip_embedder_dim_caches(fake_success_response, tiny_pil_image):
    client = ImageEmbedClient(base_url="http://stub")
    embedder = RemoteCLIPEmbedder(client)
    payload = {"embedding": [0.0] * 512, "dim": 512}
    with patch.object(
        client._session, "post", return_value=fake_success_response(payload)
    ) as post:
        embedder.dim
        embedder.dim
        embedder.dim
    # One probe hit total — the rest came from the cache.
    assert post.call_count == 1


def test_remote_clip_embedder_custom_model_name():
    """If the server provider changes, the adapter is the place to update."""
    client = ImageEmbedClient(base_url="http://stub")
    embedder = RemoteCLIPEmbedder(client, model_name="clip-vit-large-patch14")
    assert embedder.model_name == "clip-vit-large-patch14"
