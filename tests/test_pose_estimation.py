"""Tests for the Pose Estimation API — /pose-estimation/*"""

import requests
import pytest


KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# /pose-estimation/providers
# ---------------------------------------------------------------------------

def test_pose_estimation_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/pose-estimation/providers")
    assert resp.status_code == 200


def test_pose_estimation_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/pose-estimation/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# /pose-estimation/estimate  — validation errors
# ---------------------------------------------------------------------------

def test_pose_estimation_estimate_missing_image(base_url):
    resp = requests.post(f"{base_url}/pose-estimation/estimate")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "image" in body["error"].lower()


def test_pose_estimation_estimate_invalid_image_bytes(base_url):
    resp = requests.post(
        f"{base_url}/pose-estimation/estimate",
        files={"image": ("bad.png", b"not-an-image", "image/png")},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /pose-estimation/estimate  — happy path
# ---------------------------------------------------------------------------

def test_pose_estimation_estimate_returns_list(base_url, sample_image_bytes):
    resp = requests.post(
        f"{base_url}/pose-estimation/estimate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert isinstance(body["data"], list)


def test_pose_estimation_estimate_pose_shape(base_url, sample_image_bytes):
    body = requests.post(
        f"{base_url}/pose-estimation/estimate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    ).json()
    assert body["success"] is True
    for pose in body["data"]:
        assert "bbox" in pose
        assert "confidence" in pose
        assert "keypoints" in pose
        assert "cropped_image_b64" in pose
        assert len(pose["bbox"]) == 4
        assert 0.0 <= pose["confidence"] <= 1.0
        assert isinstance(pose["keypoints"], list)


def test_pose_estimation_estimate_keypoint_shape(base_url, sample_image_bytes):
    body = requests.post(
        f"{base_url}/pose-estimation/estimate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
    ).json()
    assert body["success"] is True
    for pose in body["data"]:
        for kp in pose["keypoints"]:
            assert "index" in kp
            assert "name" in kp
            assert "x" in kp
            assert "y" in kp
            assert "confidence" in kp
            assert isinstance(kp["index"], int)
            assert 0.0 <= kp["confidence"] <= 1.0
