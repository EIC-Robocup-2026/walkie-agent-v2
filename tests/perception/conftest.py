"""Shared fixtures for the perception smoke tests.

These tests verify the *shape* of every external API the perception
subsystem will rely on. They mock the network/transport boundary so the
tests can run without a real robot or running AI server.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image


def _make_json_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    """Build a stand-in for a ``requests.Response`` that returns ``payload``."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def fake_success_response():
    """Factory: wrap ``data`` in the walkie-ai-server success envelope."""

    def _build(data: Any) -> MagicMock:
        return _make_json_response({"success": True, "data": data})

    return _build


@pytest.fixture
def fake_error_response():
    """Factory: wrap an error message in the failure envelope."""

    def _build(message: str = "boom", status: int = 200) -> MagicMock:
        return _make_json_response({"success": False, "error": message}, status=status)

    return _build


@pytest.fixture
def tiny_pil_image() -> Image.Image:
    """A 4x4 RGB image — just enough to satisfy the encoder path."""
    return Image.new("RGB", (4, 4), color=(128, 128, 128))


@pytest.fixture
def tiny_jpeg_bytes(tiny_pil_image: Image.Image) -> bytes:
    """JPEG-encoded form of :func:`tiny_pil_image`."""
    buf = io.BytesIO()
    tiny_pil_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
