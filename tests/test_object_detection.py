"""Tests for the Object Detection API — /object-detection/*"""

import requests
import pytest


# ---------------------------------------------------------------------------
# /object-detection/providers
# ---------------------------------------------------------------------------

def test_object_detection_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/object-detection/providers")
    assert resp.status_code == 200


def test_object_detection_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/object-detection/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# /object-detection/detect  — validation errors
# ---------------------------------------------------------------------------

def test_object_detection_detect_missing_image(base_url):
    resp = requests.post(f"{base_url}/object-detection/detect")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_object_detection_detect_invalid_image_bytes(base_url):
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /object-detection/detect  — happy path
# ---------------------------------------------------------------------------

def test_object_detection_detect_returns_list(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)


def test_object_detection_detect_detection_shape(base_url, sample_image_bytes):
    body = requests.post(
        f"{base_url}/object-detection/detect",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    ).json()
    assert body["success"] is True
    for det in body["data"]:
        assert "bbox" in det
        assert "class_name" in det
        assert "confidence" in det
        assert "area_ratio" in det
        assert "class_id" in det
        assert len(det["bbox"]) == 4
        assert 0.0 <= det["confidence"] <= 1.0
        assert 0.0 <= det["area_ratio"] <= 1.0
