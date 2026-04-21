"""Tests for the Image Caption API — /image-caption/*"""

import io

import requests
import pytest
from PIL import Image


def _make_image_bytes(color: tuple = (100, 149, 237), size: tuple = (100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# /image-caption/providers
# ---------------------------------------------------------------------------

def test_image_caption_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/image-caption/providers")
    assert resp.status_code == 200


def test_image_caption_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/image-caption/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# /image-caption/caption  — validation errors
# ---------------------------------------------------------------------------

def test_image_caption_caption_missing_image(base_url):
    resp = requests.post(f"{base_url}/image-caption/caption")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_image_caption_caption_invalid_image_bytes(base_url):
    resp = requests.post(
        f"{base_url}/image-caption/caption",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /image-caption/caption  — happy path
# ---------------------------------------------------------------------------

def test_image_caption_caption_returns_string(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/image-caption/caption",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "caption" in body["data"]
    assert isinstance(body["data"]["caption"], str)
    assert len(body["data"]["caption"]) > 0


def test_image_caption_caption_with_prompt(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/image-caption/caption",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"prompt": "Describe the colors in this image."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["data"]["caption"], str)


# ---------------------------------------------------------------------------
# /image-caption/caption-batch  — validation errors
# ---------------------------------------------------------------------------

def test_image_caption_batch_missing_images(base_url):
    resp = requests.post(f"{base_url}/image-caption/caption-batch")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "images" in body["error"].lower()


# ---------------------------------------------------------------------------
# /image-caption/caption-batch  — happy path
# ---------------------------------------------------------------------------

def test_image_caption_batch_returns_list(base_url, sample_image_bytes):
    img_a = _make_image_bytes(color=(255, 0, 0))
    img_b = _make_image_bytes(color=(0, 0, 255))
    resp = requests.post(
        f"{base_url}/image-caption/caption-batch",
        files=[
            ("images", ("a.png", img_a, "image/png")),
            ("images", ("b.png", img_b, "image/png")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "captions" in body["data"]
    assert isinstance(body["data"]["captions"], list)
    assert len(body["data"]["captions"]) == 2


def test_image_caption_batch_each_caption_is_string(base_url, sample_image_bytes):
    img_a = _make_image_bytes(color=(255, 0, 0))
    img_b = _make_image_bytes(color=(0, 255, 0))
    body = requests.post(
        f"{base_url}/image-caption/caption-batch",
        files=[
            ("images", ("a.png", img_a, "image/png")),
            ("images", ("b.png", img_b, "image/png")),
        ],
    ).json()
    assert body["success"] is True
    for caption in body["data"]["captions"]:
        assert isinstance(caption, str)
        assert len(caption) > 0


def test_image_caption_batch_with_prompts(base_url):
    img_a = _make_image_bytes(color=(255, 0, 0))
    img_b = _make_image_bytes(color=(0, 0, 255))
    resp = requests.post(
        f"{base_url}/image-caption/caption-batch",
        files=[
            ("images", ("a.png", img_a, "image/png")),
            ("images", ("b.png", img_b, "image/png")),
        ],
        data=[
            ("prompts", "What color is this?"),
            ("prompts", "What color is this?"),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["captions"]) == 2
