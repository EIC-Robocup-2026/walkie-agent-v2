"""Frozen dataclasses + Protocols used across the perception package.

The Protocols describe the *narrowest possible* contract each collaborator
needs to satisfy. Concrete implementations live elsewhere:

  Detector       → client.ObjectDetectionClient
  Captioner      → client.ImageCaptionClient
  Embedder       → (server-side) /image-embed/* client, or FakeEmbedder
  PositionLifter → walkie_sdk.modules.tools.Tools.bboxes_to_positions
  CameraSource   → interfaces.devices.camera.Camera

By depending on Protocols, the perception code stays test-friendly:
mocks.py provides a Fake* implementation for each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from PIL import Image


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """A single object detection from one frame, before scene-store dedup.

    The pipeline produces one of these per detected object per tick. The
    store then decides whether to INSERT a new SceneEntry or UPDATE an
    existing one.
    """

    class_name: str
    class_id: Optional[int]
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    position: tuple[float, float, float]
    """``(x, y, z)`` in the upstream YOLO-3D frame (typically ``map``)."""
    embedding: tuple[float, ...]
    """L2-normalized visual or text embedding."""
    caption: str
    ts: float
    """Epoch seconds when the frame was captured."""
    frame_ref: Optional[str] = None
    """Path to the archived source frame, if persisted."""


@dataclass(frozen=True)
class SceneEntry:
    """A single record in the scene memory — read-only view from the store.

    Returned by every query method on :class:`SceneStore`. Wraps the raw
    ChromaDB metadata into a typed structure so callers never touch
    untyped dicts.
    """

    id: str
    class_name: str
    class_id: Optional[int]
    position: tuple[float, float, float]
    position_frame: str
    position_conf: float
    caption: str
    bbox_last_xyxy: tuple[int, int, int, int]
    frame_ref: Optional[str]
    first_seen_ts: float
    last_seen_ts: float
    sightings: int
    embedding: tuple[float, ...]
    embedding_model: str
    embedding_dim: int
    distance: Optional[float] = None
    """When returned from a vector knn query, the cosine distance to the
    query vector. ``None`` for metadata-only reads."""


@dataclass(frozen=True)
class SceneDiff:
    """Result of :meth:`SceneStore.diff`.

    Each list is sorted by ``last_seen_ts`` descending.
    """

    appeared: tuple[SceneEntry, ...]
    """Entries whose ``first_seen_ts > since_ts``."""

    refreshed: tuple[SceneEntry, ...]
    """Entries with ``first_seen_ts <= since_ts`` but ``last_seen_ts > since_ts``."""

    disappeared: tuple[SceneEntry, ...]
    """Entries with ``last_seen_ts <= since_ts`` (not seen since the cutoff)."""


@dataclass(frozen=True)
class DedupDecision:
    """Outcome of dedup classification for one incoming :class:`Detection`."""

    action: str
    """One of ``"insert"``, ``"update"``."""
    target_id: Optional[str] = None
    """For ``update``: the SceneEntry id to merge into. ``None`` for insert."""
    reason: str = ""
    """Human-readable explanation; logged for debugging."""


@dataclass(frozen=True)
class TickReport:
    """Telemetry emitted at the end of each background-loop tick.

    Logged as JSON and optionally passed to a user-provided ``on_tick``
    callback so callers can wire it into Prometheus / file logs / a HUD.
    """

    ts: float
    n_detections: int
    n_inserts: int
    n_updates: int
    n_skipped: int
    """Detections we received but discarded (e.g. 3D lift failed)."""
    n_pruned: int = 0
    """Records evicted by the periodic prune on this tick (0 on most ticks —
    pruning runs on its own cadence, not every tick)."""
    latency_ms: dict[str, float] = field(default_factory=dict)
    """Per-stage wall time in milliseconds: ``capture``, ``detect``,
    ``lift``, ``caption``, ``embed``, ``store``."""
    error: Optional[str] = None
    """If the tick raised before completing, the exception message."""


# ---------------------------------------------------------------------------
# Collaborator protocols
# ---------------------------------------------------------------------------


class CameraSource(Protocol):
    """Anything that can hand us a PIL frame on demand."""

    def capture_pil(self) -> Image.Image: ...


class Detector(Protocol):
    """Object detection. ``confidence`` ∈ [0, 1]."""

    def detect(self, image: Image.Image) -> Sequence["RawDetection"]: ...


class RawDetection(Protocol):
    """The minimum surface this package needs from a detection result.

    Both :class:`client.object_detection.DetectedObject` and the fake
    detector satisfy this — they expose ``class_name``, ``class_id``,
    ``confidence``, and ``bbox`` (xyxy).
    """

    class_name: Optional[str]
    class_id: Optional[int]
    confidence: Optional[float]
    bbox: tuple[int, int, int, int]


class Captioner(Protocol):
    """Image captioning. May accept an optional steering ``prompt``."""

    def caption(self, image: Image.Image, prompt: Optional[str] = None) -> str: ...


class Embedder(Protocol):
    """Joint image+text embedding model.

    ``embed_image`` may receive a *crop* of the source frame (the
    detection's bbox region) rather than the full frame, so two adjacent
    objects produce distinct vectors. Both methods must return
    L2-normalized vectors of identical dimension. ``dim`` is exposed
    separately so the store can record it in metadata.
    """

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_image(self, image: Image.Image) -> list[float]: ...

    def embed_text(self, text: str) -> list[float]: ...


class PositionLifter(Protocol):
    """2D bbox(es) → 3D world-frame position(s).

    Matches the signature of ``walkie_sdk.modules.tools.Tools.bboxes_to_positions``:
    input ``[cx, cy, w, h]`` per bbox; output ``[x, y, z]`` per bbox in the
    upstream YOLO-3D node's frame, or ``None`` on timeout.
    """

    def bboxes_to_positions(
        self,
        coords: list[list[float]],
        timeout: float = 5.0,
    ) -> Optional[list[list[float]]]: ...
