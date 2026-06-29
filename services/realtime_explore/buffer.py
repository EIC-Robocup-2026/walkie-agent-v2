"""On-disk snapshot ring buffer — cheap continuous capture for the batch builder.

The ``ready``-stage perception loop captures an RGB-D frame + its detections every
tick. Running the full association/fusion pipeline inline on every tick is wasteful;
instead each frame is appended here as a compact, lossless-enough :class:`Snapshot`,
and a separate batch build worker replays a window of snapshots later to (re)build the
scene graph. This keeps the hot capture path to one ``np.savez_compressed`` per frame.

On-disk layout (all under ``buffer_dir``)::

    index.jsonl                 one JSON line per snapshot (metadata only, no pixels)
    snap_<id>.npz               per-snapshot sidecar: depth (uint16 mm), packed masks,
                                optional rgb

Per-detection class/caption/embedding live INLINE in ``index.jsonl`` so the build
worker can read them (e.g. for association ordering) without decoding any pixels.

Depth is stored as ``uint16`` millimetres (``0`` = invalid sentinel, range 0..65.535 m);
masks are bit-packed with :func:`numpy.packbits`. Reconstruction is exact for masks and
within 1 mm for depth. Everything is numpy-only — no cv2/PIL/open3d — so the buffer
round-trips under a bare ``numpy/scipy`` install.

Eviction is a ring: once more than ``cap`` snapshots are present, the oldest sidecars
are unlinked and ``index.jsonl`` is rewritten without them. A build worker can take a
:meth:`SnapshotBuffer.building` pin to freeze the id list and defer eviction of those
ids until it releases the pin — appends past ``cap`` are allowed to accumulate, then the
backlog is evicted in one catch-up once the pin lifts.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

_DEPTH_MAX_M = 65.535  # uint16 mm range: round(65.535 * 1000) == 65535
_INDEX_NAME = "index.jsonl"


# ---------------------------------------------------------------------------
# Canonical capture records
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One open-vocabulary detection within a :class:`Snapshot`."""

    class_name: str
    class_id: Optional[int]
    conf: float
    bbox: tuple[int, int, int, int]  # xyxy
    caption: str
    clip_emb: list[float]
    mask: np.ndarray  # (H, W) uint8/bool in {0, 1}


@dataclass
class Snapshot:
    """One captured RGB-D frame + its detections, as appended to the buffer."""

    ts: float
    depth: np.ndarray  # (H, W) float32 metres; NaN / <= 0 == invalid
    intr: tuple[float, float, float, float, float, float]  # (fx, fy, cx, cy, w, h)
    cam_R: np.ndarray  # (3, 3) world<-camera rotation
    cam_t: np.ndarray  # (3,) world camera position
    robot_pose: Optional[dict]  # {"x", "y", "heading"} or None
    detections: list[Detection] = field(default_factory=list)
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8


# ---------------------------------------------------------------------------
# (de)serialization helpers — pure numpy
# ---------------------------------------------------------------------------
def _encode_depth(depth: np.ndarray) -> np.ndarray:
    """float32 metres -> uint16 millimetres; NaN / <= 0 -> 0 sentinel, clipped to range."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    mm = np.zeros(d.shape, dtype=np.uint16)
    clipped = np.clip(d, 0.0, _DEPTH_MAX_M)
    mm[valid] = np.round(clipped[valid] * 1000.0).astype(np.uint16)
    return mm


def _decode_depth(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 millimetres -> float32 metres; 0 sentinel -> NaN."""
    mm = np.asarray(depth_mm, dtype=np.uint16)
    out = (mm.astype(np.float32)) / 1000.0
    out[mm == 0] = np.nan
    return out


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    """Bit-pack an (H, W) {0,1} mask to a flat uint8 array (shape carried separately)."""
    m = np.asarray(mask)
    flat = (m.reshape(-1) != 0).astype(np.uint8)
    return np.packbits(flat)


def _unpack_mask(packed: np.ndarray, h: int, w: int) -> np.ndarray:
    """Inverse of :func:`_pack_mask` -> (H, W) uint8 mask in {0, 1}."""
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    return bits.reshape(h, w).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------
class SnapshotBuffer:
    """An on-disk ring buffer of :class:`Snapshot`\\ s with deferred-eviction pins.

    Thread-safe: a single lock guards the index list and all on-disk mutations, so the
    capture thread (:meth:`append`) and a build worker (:meth:`building` /
    :meth:`load_window`) can share one buffer.
    """

    def __init__(self, buffer_dir: str | Path, *, cap: int = 400, keep_rgb: bool = False):
        self.dir = Path(buffer_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap = int(cap)
        self.keep_rgb = bool(keep_rgb)
        self._index_path = self.dir / _INDEX_NAME
        self._lock = threading.RLock()
        # In-memory mirror of index.jsonl: list of dict entries, oldest..newest.
        self._entries: list[dict] = self._read_index()
        # ids that an active build pin protects from eviction (None == no pin).
        self._pinned: Optional[set[str]] = None

    # -- index I/O ----------------------------------------------------------
    def _read_index(self) -> list[dict]:
        if not self._index_path.exists():
            return []
        entries: list[dict] = []
        with self._index_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue  # tolerate a half-written trailing line
        return entries

    def _rewrite_index(self) -> None:
        """Atomically rewrite index.jsonl from the in-memory mirror (.tmp -> os.replace)."""
        tmp = self._index_path.with_suffix(self._index_path.suffix + ".tmp")
        with tmp.open("w") as f:
            for e in self._entries:
                f.write(json.dumps(e))
                f.write("\n")
        os.replace(tmp, self._index_path)

    def _sidecar_path(self, snap_id: str) -> Path:
        return self.dir / f"snap_{snap_id}.npz"

    # -- append -------------------------------------------------------------
    def append(self, snap: Snapshot) -> str:
        """Append ``snap``; write its sidecar + index line; ring-evict the overflow.

        Returns the new snapshot id. Eviction of ids present at the most recent
        :meth:`building` pin is deferred until the pin is released.
        """
        snap_id = uuid.uuid4().hex
        sidecar = self._sidecar_path(snap_id)

        # Build the index entry (metadata only — pixels go to the sidecar).
        det_meta = []
        save: dict[str, np.ndarray] = {}
        for i, d in enumerate(snap.detections):
            mask = np.asarray(d.mask)
            mh, mw = (int(mask.shape[0]), int(mask.shape[1])) if mask.ndim == 2 else (0, 0)
            save[f"mask_{i}"] = _pack_mask(mask)
            det_meta.append(
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "conf": float(d.conf),
                    "bbox": [int(x) for x in d.bbox],
                    "caption": d.caption,
                    "clip_emb": [float(x) for x in d.clip_emb],
                    "mask_h": mh,
                    "mask_w": mw,
                }
            )

        save["depth_mm"] = _encode_depth(snap.depth)
        if self.keep_rgb and snap.rgb is not None:
            save["rgb"] = np.asarray(snap.rgb, dtype=np.uint8)

        entry = {
            "id": snap_id,
            "ts": float(snap.ts),
            "intr": [float(x) for x in snap.intr],
            "cam_R": [float(x) for x in np.asarray(snap.cam_R, dtype=float).reshape(-1)],
            "cam_t": [float(x) for x in np.asarray(snap.cam_t, dtype=float).reshape(-1)],
            "robot_pose": snap.robot_pose,
            "sidecar": sidecar.name,
            "detections": det_meta,
        }

        with self._lock:
            # Write the sidecar first; only then commit it to the index, so a crash
            # mid-write never leaves a referenced-but-missing sidecar.
            np.savez_compressed(sidecar, **save)
            self._entries.append(entry)
            self._evict_locked()
            self._rewrite_index()
        return snap_id

    # -- eviction -----------------------------------------------------------
    def _evict_locked(self) -> None:
        """Drop oldest sidecars beyond ``cap``, skipping ids pinned by a live build.

        While a build pin is held the whole pass is deferred: the buffer is allowed to
        grow past ``cap`` (the build is replaying a frozen window and must not lose any
        of it, nor the fresh frames captured alongside it). The backlog is evicted in
        one catch-up when :meth:`building` releases the pin.
        """
        if self._pinned is not None:
            return
        while len(self._entries) > self.cap:
            e = self._entries.pop(0)  # oldest
            self._unlink_sidecar(e)

    def _unlink_sidecar(self, entry: dict) -> None:
        path = self.dir / entry.get("sidecar", f"snap_{entry['id']}.npz")
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # -- pin / build guard --------------------------------------------------
    @contextmanager
    def building(self) -> Iterator[list[str]]:
        """Freeze the current id list and block its eviction for the ``with`` body.

        Yields the snapshot ids present at entry (oldest..newest). While the pin is
        held, :meth:`append` may push the buffer past ``cap`` but will not unlink any
        pinned sidecar; on exit the deferred overflow is evicted in one catch-up.
        """
        with self._lock:
            snap_ids = [e["id"] for e in self._entries]
            self._pinned = set(snap_ids)
        try:
            yield snap_ids
        finally:
            with self._lock:
                self._pinned = None
                self._evict_locked()
                self._rewrite_index()

    # -- sizing / listing ---------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def ids(self) -> list[str]:
        """Snapshot ids in buffer order (oldest..newest)."""
        with self._lock:
            return [e["id"] for e in self._entries]

    # -- loading ------------------------------------------------------------
    def _entry_for(self, snap_id: str) -> Optional[dict]:
        with self._lock:
            for e in self._entries:
                if e["id"] == snap_id:
                    return e
        return None

    def _load_entry(self, entry: dict) -> Optional[Snapshot]:
        """Decode one index entry + its sidecar into a :class:`Snapshot` (None on error)."""
        sidecar = self.dir / entry.get("sidecar", f"snap_{entry['id']}.npz")
        try:
            data = np.load(sidecar, allow_pickle=False)
        except (FileNotFoundError, OSError, ValueError, EOFError):
            return None  # missing / half-written sidecar — skip
        try:
            depth = _decode_depth(data["depth_mm"])
            dets: list[Detection] = []
            for i, dm in enumerate(entry.get("detections", [])):
                key = f"mask_{i}"
                if key in data:
                    mask = _unpack_mask(data[key], int(dm["mask_h"]), int(dm["mask_w"]))
                else:
                    mask = np.zeros((int(dm["mask_h"]), int(dm["mask_w"])), dtype=np.uint8)
                dets.append(
                    Detection(
                        class_name=dm["class_name"],
                        class_id=dm["class_id"],
                        conf=float(dm["conf"]),
                        bbox=tuple(int(x) for x in dm["bbox"]),
                        caption=dm.get("caption", ""),
                        clip_emb=list(dm.get("clip_emb", [])),
                        mask=mask,
                    )
                )
            rgb = np.asarray(data["rgb"], dtype=np.uint8) if "rgb" in data else None
        finally:
            close = getattr(data, "close", None)
            if close is not None:
                close()
        return Snapshot(
            ts=float(entry["ts"]),
            depth=depth,
            intr=tuple(float(x) for x in entry["intr"]),
            cam_R=np.asarray(entry["cam_R"], dtype=float).reshape(3, 3),
            cam_t=np.asarray(entry["cam_t"], dtype=float).reshape(3),
            robot_pose=entry.get("robot_pose"),
            detections=dets,
            rgb=rgb,
        )

    def load(self, snap_id: str) -> Snapshot:
        """Lazily load + decode a single snapshot by id (raises if missing/unreadable)."""
        entry = self._entry_for(snap_id)
        if entry is None:
            raise KeyError(snap_id)
        snap = self._load_entry(entry)
        if snap is None:
            raise FileNotFoundError(f"sidecar for snapshot {snap_id} is missing or unreadable")
        return snap

    def load_window(self, n: Optional[int] = None) -> list[Snapshot]:
        """Load the newest ``n`` snapshots (all if ``None``), returned oldest-first.

        Sidecars that fail to load (missing / half-written) are silently skipped.
        """
        with self._lock:
            entries = list(self._entries)
        if n is not None:
            entries = entries[-int(n):] if n > 0 else []
        out: list[Snapshot] = []
        for e in entries:
            snap = self._load_entry(e)
            if snap is not None:
                out.append(snap)
        return out

    def load_all(self) -> list[Snapshot]:
        """Load every buffered snapshot, oldest-first (skipping unreadable sidecars)."""
        return self.load_window(None)
