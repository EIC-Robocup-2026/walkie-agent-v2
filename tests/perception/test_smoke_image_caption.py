"""Smoke: ImageCaptionClient unwraps the {caption: str} payload.

Captions become the ``documents`` field in our ChromaDB collection, so we
need the wrapper to return a plain string we can hand straight to Chroma.
"""

from __future__ import annotations

from unittest.mock import patch

from client.image_caption import ImageCaptionClient


def test_caption_returns_string(tiny_pil_image, fake_success_response):
    client = ImageCaptionClient(base_url="http://stub")
    with patch.object(
        client._session,
        "post",
        return_value=fake_success_response({"caption": "a grey square on a grey square"}),
    ) as post:
        out = client.caption(tiny_pil_image, prompt="describe")

    assert out == "a grey square on a grey square"
    assert post.call_args.args[0] == "http://stub/image-caption/caption"
    assert post.call_args.kwargs["data"] == {"prompt": "describe"}


def test_caption_batch_returns_strings(tiny_pil_image, fake_success_response):
    client = ImageCaptionClient(base_url="http://stub")
    with patch.object(
        client._session,
        "post",
        return_value=fake_success_response({"captions": ["a", "b"]}),
    ):
        out = client.caption_batch([tiny_pil_image, tiny_pil_image])

    assert out == ["a", "b"]
