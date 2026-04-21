"""Tests for the STT (Speech-to-Text) API — /stt/*"""

import requests
import pytest


# ---------------------------------------------------------------------------
# /stt/providers
# ---------------------------------------------------------------------------

def test_stt_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/stt/providers")
    assert resp.status_code == 200


def test_stt_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/stt/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# /stt/transcribe  — validation errors (no model inference required)
# ---------------------------------------------------------------------------

def test_stt_transcribe_missing_audio_field(base_url):
    resp = requests.post(f"{base_url}/stt/transcribe")
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "audio" in body["error"].lower()


# ---------------------------------------------------------------------------
# /stt/transcribe  — happy path (requires a running model)
# ---------------------------------------------------------------------------

def test_stt_transcribe_silent_wav_returns_transcription(base_url, sample_audio_bytes):
    resp = requests.post(
        f"{base_url}/stt/transcribe",
        files={"audio": ("silence.wav", sample_audio_bytes, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "transcription" in body["data"]
    assert isinstance(body["data"]["transcription"], str)


def test_stt_transcribe_response_is_string(base_url, sample_audio_bytes):
    body = requests.post(
        f"{base_url}/stt/transcribe",
        files={"audio": ("silence.wav", sample_audio_bytes, "audio/wav")},
    ).json()
    assert isinstance(body["data"]["transcription"], str)
