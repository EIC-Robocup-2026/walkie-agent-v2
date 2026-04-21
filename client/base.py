"""Shared base for all Walkie HTTP API clients."""

from __future__ import annotations

import base64
import io
from typing import Any

import requests
from PIL import Image


class WalkieAPIError(Exception):
    """Raised when the API returns ``{"success": false, ...}``."""


def _pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode a PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _b64_to_pil(b64: str | None) -> Image.Image | None:
    """Decode a base64 string back into a PIL Image, or return None."""
    if b64 is None:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class WalkieBaseClient:
    """Base HTTP client shared by all sub-clients.

    Holds a single :class:`requests.Session` (connection pooling),
    the server base URL, and a default timeout.  All sub-clients
    inherit from this class and call :py:meth:`_get` / :py:meth:`_post_json`
    / :py:meth:`_post_files` rather than constructing requests directly.
    """

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        """GET *path* and return ``body["data"]``, raising on errors."""
        resp = self._session.get(f"{self._base_url}{path}", timeout=self._timeout)
        resp.raise_for_status()
        return self._unwrap(resp.json())

    def _post_json(self, path: str, payload: dict) -> Any:
        """POST JSON *payload* to *path* and return ``body["data"]``."""
        resp = self._session.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return self._unwrap(resp.json())

    def _post_files(
        self,
        path: str,
        files: list[tuple] | dict,
        data: list[tuple] | dict | None = None,
    ) -> Any:
        """POST a multipart/form-data request and return ``body["data"]``."""
        resp = self._session.post(
            f"{self._base_url}{path}",
            files=files,
            data=data or {},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return self._unwrap(resp.json())

    def _post_files_stream(
        self,
        path: str,
        files: list[tuple] | dict,
        data: list[tuple] | dict | None = None,
        chunk_size: int = 4096,
    ):
        """POST multipart/form-data and yield raw audio chunks (no JSON unwrap)."""
        resp = self._session.post(
            f"{self._base_url}{path}",
            files=files,
            data=data or {},
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()
        yield from resp.iter_content(chunk_size=chunk_size)

    def _post_json_stream(self, path: str, payload: dict, chunk_size: int = 4096):
        """POST JSON and yield raw audio chunks (no JSON unwrap)."""
        resp = self._session.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=self._timeout,
            stream=True,
        )
        resp.raise_for_status()
        yield from resp.iter_content(chunk_size=chunk_size)

    @staticmethod
    def _unwrap(body: dict) -> Any:
        """Return ``body["data"]`` or raise :class:`WalkieAPIError`."""
        if not body.get("success"):
            raise WalkieAPIError(body.get("error", "Unknown API error"))
        return body["data"]
