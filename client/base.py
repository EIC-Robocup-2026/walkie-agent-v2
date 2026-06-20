"""Shared base for all Walkie HTTP API clients."""

from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image


class WalkieAPIError(Exception):
    """Raised when the API returns ``{"success": false, ...}``."""


def _numpy_to_bytes(arr: np.ndarray, fmt: str = "JPEG", quality: int = 85) -> bytes:
    """Encode a BGR numpy array to bytes via cv2 (faster than PIL).

    Avoids the channel-swap + PIL object allocation + PIL encoder overhead.
    cv2 JPEG encoding uses libjpeg-turbo internally.
    """
    if fmt.upper() in ("JPEG", "JPG"):
        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for format {fmt!r}")
    return buf.tobytes()


def _pil_to_bytes(image: Image.Image, fmt: str = "PNG", quality: int = 85) -> bytes:
    """Encode a PIL Image to bytes (fallback for callers that already hold a PIL object)."""
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if fmt.upper() in ("JPEG", "JPG"):
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return buf.getvalue()


def _numpy_to_npy_bytes(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to ``.npy`` bytes (lossless, shape + dtype preserved).

    Used to ship dense numeric arrays (e.g. an ``(N, 3)`` point cloud) over the
    multipart file channel, the same way images go as encoded-image bytes — far
    cheaper and exact compared to JSON lists.
    """
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def _b64_to_pil(b64: str | None) -> Image.Image | None:
    """Decode a base64 string back into a PIL Image, or return None."""
    if b64 is None:
        return None
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _b64_to_mask(b64: str | None) -> "np.ndarray | None":
    """Decode a base64 PNG mask into a 2D uint8 {0,1} array, or None.

    Inverts the server's encoding (a binary mask saved as an 8-bit grayscale
    PNG scaled by 255), thresholding back to {0, 1}.
    """
    if b64 is None:
        return None
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    return (np.asarray(img) > 127).astype(np.uint8)


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
