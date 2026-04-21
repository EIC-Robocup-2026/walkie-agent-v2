"""Tests for the TTS (Text-to-Speech) API — /tts/*"""

import requests
import pytest


SAMPLE_TEXT = "Hello, this is a test."


# ---------------------------------------------------------------------------
# /tts/providers
# ---------------------------------------------------------------------------

def test_tts_providers_returns_200(base_url):
    resp = requests.get(f"{base_url}/tts/providers")
    assert resp.status_code == 200


def test_tts_providers_success_shape(base_url):
    body = requests.get(f"{base_url}/tts/providers").json()
    assert body["success"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


# ---------------------------------------------------------------------------
# /tts/synthesize  — validation errors
# ---------------------------------------------------------------------------

def test_tts_synthesize_missing_text_field(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize", json={})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "text" in body["error"].lower()


def test_tts_synthesize_empty_text_field(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize", json={"text": ""})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False


# ---------------------------------------------------------------------------
# /tts/synthesize  — happy path
# ---------------------------------------------------------------------------

def test_tts_synthesize_returns_audio_bytes(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize", json={"text": SAMPLE_TEXT})
    assert resp.status_code == 200
    assert len(resp.content) > 0


def test_tts_synthesize_content_type_is_audio(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize", json={"text": SAMPLE_TEXT})
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("audio/")


def test_tts_synthesize_accepts_form_data(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize", data={"text": SAMPLE_TEXT})
    assert resp.status_code == 200
    assert len(resp.content) > 0


# ---------------------------------------------------------------------------
# /tts/synthesize-stream  — validation errors
# ---------------------------------------------------------------------------

def test_tts_synthesize_stream_missing_text(base_url):
    resp = requests.post(f"{base_url}/tts/synthesize-stream", json={})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "text" in body["error"].lower()


# ---------------------------------------------------------------------------
# /tts/synthesize-stream  — happy path
# ---------------------------------------------------------------------------

def test_tts_synthesize_stream_returns_audio_bytes(base_url):
    resp = requests.post(
        f"{base_url}/tts/synthesize-stream",
        json={"text": SAMPLE_TEXT},
        stream=True,
    )
    assert resp.status_code == 200
    chunks = list(resp.iter_content(chunk_size=4096))
    total_bytes = sum(len(c) for c in chunks)
    assert total_bytes > 0


def test_tts_synthesize_stream_content_type_is_audio(base_url):
    resp = requests.post(
        f"{base_url}/tts/synthesize-stream",
        json={"text": SAMPLE_TEXT},
        stream=True,
    )
    assert resp.status_code == 200
    assert resp.headers["Content-Type"].startswith("audio/")
