"""Test mocks for the perception subsystem.

Every fake satisfies the corresponding Protocol in ``types.py`` via duck
typing — no Protocol/ABC subclassing required. Tests construct these
with scripted inputs/outputs so the whole pipeline runs deterministically
without a robot, a model, or a network.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional, Sequence

from PIL import Image


# ---------------------------------------------------------------------------
# Detection record used by the fakes (matches client.object_detection shape)
# ---------------------------------------------------------------------------


@dataclass
class FakeDetectedObject:
    class_name: Optional[str]
    class_id: Optional[int]
    confidence: Optional[float]
    bbox: tuple[int, int, int, int]
    area_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class FakeCamera:
    """Cycles through a fixed sequence of PIL frames on each ``capture_pil()``."""

    def __init__(self, frames: Sequence[Image.Image]) -> None:
        if not frames:
            raise ValueError("FakeCamera needs at least one frame")
        self._frames = list(frames)
        self._idx = 0

    def capture_pil(self) -> Image.Image:
        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return frame


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class FakeDetector:
    """Scripted detections keyed by frame index.

    Optionally raises a configured exception on a given tick to exercise
    the loop's error-recovery path. Frame indexing is zero-based and
    advances with each ``detect()`` call.
    """

    def __init__(
        self,
        scripted: dict[int, list[FakeDetectedObject]] | list[list[FakeDetectedObject]],
        *,
        raise_on_idx: Optional[int] = None,
        exc: Exception = RuntimeError("fake detector failure"),
    ) -> None:
        if isinstance(scripted, list):
            self._scripted = {i: v for i, v in enumerate(scripted)}
        else:
            self._scripted = dict(scripted)
        self._idx = 0
        self._raise_on_idx = raise_on_idx
        self._exc = exc

    def detect(self, image: Image.Image) -> list[FakeDetectedObject]:
        idx = self._idx
        self._idx += 1
        if self._raise_on_idx is not None and idx == self._raise_on_idx:
            raise self._exc
        return list(self._scripted.get(idx, []))


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------


class FakeCaptioner:
    """Returns a configurable caption — either fixed text or a per-prompt map.

    ``delay`` simulates inference latency without actually blocking the
    event loop (the loop's ``asyncio.to_thread`` wrapper turns it into a
    yield point).
    """

    def __init__(
        self,
        captions: dict[str, str] | str = "a generic scene",
        *,
        delay: float = 0.0,
    ) -> None:
        self._captions = captions
        self._delay = delay

    def caption(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        if self._delay > 0:
            time.sleep(self._delay)
        if isinstance(self._captions, str):
            return self._captions
        key = prompt or ""
        return self._captions.get(key, self._captions.get("", "a generic scene"))


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embeddings without downloading a model.

    Image embeddings are hashed from a fingerprint of the input (a small
    downscaled sum of pixel intensities so two visually similar PIL
    images get nearby vectors). Text embeddings are hashed from the
    string. Both are L2-normalized to length 1.

    To control specific embeddings in a test, pass ``override_text`` or
    ``override_image`` mappings — a matching key returns the supplied
    vector verbatim (pre-normalized by the caller if desired).
    """

    DEFAULT_DIM = 16

    def __init__(
        self,
        *,
        dim: int = DEFAULT_DIM,
        override_text: Optional[dict[str, Sequence[float]]] = None,
        override_image: Optional[dict[str, Sequence[float]]] = None,
        model_name: str = "fake-embedder-v1",
    ) -> None:
        self._dim = dim
        self._override_text = {k: list(v) for k, v in (override_text or {}).items()}
        self._override_image = {k: list(v) for k, v in (override_image or {}).items()}
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_image(self, image: Image.Image) -> list[float]:
        # Cheap perceptual fingerprint: average color of a 4x4 downsample.
        thumb = image.convert("RGB").resize((4, 4))
        key = "img:" + ",".join(
            f"{r},{g},{b}" for (r, g, b) in thumb.getdata()
        )
        if key in self._override_image:
            return list(self._override_image[key])
        return self._hash_to_unit(key)

    def embed_text(self, text: str) -> list[float]:
        if text in self._override_text:
            return list(self._override_text[text])
        return self._hash_to_unit("txt:" + text)

    def _hash_to_unit(self, seed: str) -> list[float]:
        digest = hashlib.sha256(seed.encode()).digest()
        # Expand bytes deterministically into ``dim`` floats in [-1, 1]
        bs = (digest * ((self._dim // len(digest)) + 1))[: self._dim]
        raw = [(b / 127.5) - 1.0 for b in bs]
        norm = sum(x * x for x in raw) ** 0.5
        if norm <= 1e-12:
            return [1.0] + [0.0] * (self._dim - 1)
        return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Position lifter
# ---------------------------------------------------------------------------


class FakePositionLifter:
    """Maps each bbox to a 3D position via a lookup or a function.

    ``scripted`` keys are ``(cx, cy, w, h)`` tuples (input shape of
    ``bboxes_to_positions``); the value is the ``[x, y, z]`` to return.
    If a bbox isn't in the lookup, falls back to ``default`` (default:
    ``[0, 0, 0]``) so tests can mass-script.

    Pass ``timeout_after=N`` to return ``None`` on the Nth call,
    simulating an upstream ROS-3D node timeout.
    """

    def __init__(
        self,
        scripted: Optional[dict[tuple[float, float, float, float], list[float]]] = None,
        *,
        default: list[float] | None = None,
        timeout_after: Optional[int] = None,
    ) -> None:
        self._scripted = scripted or {}
        self._default = list(default) if default is not None else [0.0, 0.0, 0.0]
        self._timeout_after = timeout_after
        self._calls = 0

    def bboxes_to_positions(
        self,
        coords: list[list[float]],
        timeout: float = 5.0,
    ) -> Optional[list[list[float]]]:
        self._calls += 1
        if self._timeout_after is not None and self._calls >= self._timeout_after:
            return None
        out = []
        for c in coords:
            key = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            out.append(list(self._scripted.get(key, self._default)))
        return out


# ---------------------------------------------------------------------------
# Convenience: tiny PIL image with a deterministic fingerprint
# ---------------------------------------------------------------------------


def make_tiny_image(seed: int, size: tuple[int, int] = (8, 8)) -> Image.Image:
    """8x8 RGB image whose pixel values depend on ``seed`` — used to make
    each scripted detection embed to a different vector."""
    digest = hashlib.sha256(str(seed).encode()).digest()
    img = Image.new("RGB", size)
    px = img.load()
    n = size[0] * size[1]
    for i in range(n):
        x = i % size[0]
        y = i // size[0]
        r, g, b = digest[(3 * i) % 32], digest[(3 * i + 1) % 32], digest[(3 * i + 2) % 32]
        px[x, y] = (r, g, b)
    return img
