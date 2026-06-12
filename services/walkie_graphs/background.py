"""BackgroundStore — the classless world cloud (walls, floor, furniture shells).

Every capture lifts the *remainder* of its depth frame (valid pixels not under
any detection mask) into a coarse cloud. Those points have no class, caption,
or embedding — they exist for two jobs:

- **ICP anchor**: capture-level registration aligns each new capture against
  the map, and the background is most of what overlaps between two views of
  the same place (objects are sparse; walls are everywhere).
- **Visualization**: Rerun renders it grey under the object clouds.

Growth is bounded by **fixed-origin voxel dedup**: each point maps to a 5 cm
(default) grid cell key, and only points landing in a never-seen cell are
appended — re-observing the same wall every 2 s adds nothing. The grid origin
is fixed at the world origin (unlike :func:`geometry.voxel_downsample`, whose
grid shifts with each batch's minimum — keys would not be comparable across
calls). The first point seen in a cell represents it forever (no re-averaging:
at 5 cm that error is below the sensor's). Past ``max_points`` the oldest
cells are evicted FIFO, which also slowly forgets stale geometry.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from .pcd_ops import subsample

# Grid keys pack 3 signed cell coords into one int64 (21 bits each → ±1M cells,
# i.e. ±52 km at a 5 cm voxel — far beyond any map).
_KEY_OFF = 1 << 20
_KEY_MAX = (1 << 21) - 1


class BackgroundStore:
    """Bounded, deduplicated, classless world cloud with npz persistence."""

    def __init__(
        self,
        path: str,
        *,
        voxel_m: float = 0.05,
        max_points: int = 300_000,
    ) -> None:
        self.path = path
        self.voxel_m = float(voxel_m)
        self.max_points = int(max_points)
        self._lock = threading.Lock()
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._keys = np.zeros((0,), dtype=np.int64)
        self._key_set: set[int] = set()
        self.load()

    @classmethod
    def from_env(cls) -> "BackgroundStore":
        return cls(
            os.getenv("WALKIE_GRAPHS_BG_PATH", "graph_background.npz"),
            voxel_m=float(os.getenv("WALKIE_GRAPHS_BG_VOXEL_M", "0.05")),
            max_points=int(os.getenv("WALKIE_GRAPHS_BG_MAX_POINTS", "300000")),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._points)

    def _grid_keys(self, pts: np.ndarray) -> np.ndarray:
        cells = np.floor(pts / self.voxel_m).astype(np.int64) + _KEY_OFF
        np.clip(cells, 0, _KEY_MAX, out=cells)
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def add(self, points: np.ndarray) -> int:
        """Fold a capture's remainder in; returns how many new points were kept.

        Only the first point per never-seen grid cell is appended, so repeated
        views of mapped geometry are free. FIFO-evicts past ``max_points``.
        """
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return 0
        pts = pts.reshape(-1, 3)
        with self._lock:
            keys = self._grid_keys(pts)
            _, first_idx = np.unique(keys, return_index=True)
            fresh = sorted(
                int(i) for i in first_idx if int(keys[i]) not in self._key_set
            )
            if not fresh:
                return 0
            new_pts, new_keys = pts[fresh], keys[fresh]
            self._points = (
                np.vstack([self._points, new_pts]) if len(self._points) else new_pts
            )
            self._keys = np.concatenate([self._keys, new_keys])
            self._key_set.update(new_keys.tolist())
            over = len(self._points) - self.max_points
            if over > 0:
                self._key_set.difference_update(self._keys[:over].tolist())
                self._points = self._points[over:]
                self._keys = self._keys[over:]
            return len(new_pts)

    def crop(
        self,
        aabb_min,
        aabb_max,
        *,
        pad: float = 0.5,
        budget: int = 0,
    ) -> np.ndarray:
        """Points inside the padded box — the ICP-target assembly query."""
        with self._lock:
            pts = self._points
            if len(pts) == 0:
                return pts
            lo = np.asarray(aabb_min, dtype=np.float32) - pad
            hi = np.asarray(aabb_max, dtype=np.float32) + pad
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
            out = pts[inside]
        return subsample(out, budget) if budget else out

    def points(self) -> np.ndarray:
        """A snapshot copy of the whole cloud (for viz)."""
        with self._lock:
            return self._points.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write the cloud atomically (tmp + ``os.replace``)."""
        with self._lock:
            points, keys = self._points.copy(), self._keys.copy()
        tmp = f"{self.path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(f, points=points, keys=keys)
        os.replace(tmp, self.path)

    def load(self) -> None:
        """Reload from disk; a missing or corrupt file starts empty."""
        try:
            with np.load(self.path) as data:
                points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
                keys = np.asarray(data["keys"], dtype=np.int64).reshape(-1)
        except Exception:  # noqa: BLE001 — absent/corrupt store is a fresh start
            return
        if len(points) != len(keys):
            return
        with self._lock:
            self._points, self._keys = points, keys
            self._key_set = set(keys.tolist())

    def clear(self) -> None:
        """Drop everything, in memory and on disk."""
        with self._lock:
            self._points = np.zeros((0, 3), dtype=np.float32)
            self._keys = np.zeros((0,), dtype=np.int64)
            self._key_set = set()
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
