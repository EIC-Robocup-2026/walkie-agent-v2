"""BackgroundStore: fixed-origin voxel dedup, FIFO cap, crop, persistence."""

from __future__ import annotations

import numpy as np

from services.walkie_graphs.background import BackgroundStore


def _grid(x0, x1, *, step=0.1, z=0.0):
    """A flat lattice of points covering [x0, x1) in x and y."""
    a = np.arange(x0, x1, step, dtype=np.float32)
    xs, ys = np.meshgrid(a, a)
    return np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z, np.float32)], axis=1)


def test_add_dedups_repeated_views(tmp_path):
    bg = BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05)
    pts = _grid(0, 1)
    n1 = bg.add(pts)
    assert n1 == len(bg) > 0
    # Re-observing the same wall adds nothing (fixed-origin keys are stable).
    assert bg.add(pts) == 0
    assert len(bg) == n1
    # Jittered re-observation within the same cells adds nothing either.
    assert bg.add(pts + 0.001) == 0


def test_add_within_batch_dedup():
    bg = BackgroundStore("unused.npz", voxel_m=0.05)
    same_cell = np.zeros((10, 3), dtype=np.float32) + 0.01
    assert bg.add(same_cell) == 1


def test_fifo_eviction_past_cap(tmp_path):
    bg = BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05, max_points=100)
    first = _grid(0, 2)  # 400 cells -> immediately over the 100-point cap
    bg.add(first)
    assert len(bg) == 100
    # The oldest cells were evicted, so their region can be re-learned.
    assert bg.add(first[:10]) > 0
    assert len(bg) == 100


def test_crop_and_budget(tmp_path):
    bg = BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05)
    bg.add(_grid(0, 1))
    bg.add(_grid(5, 6))
    near = bg.crop((0, 0, -1), (1, 1, 1), pad=0.0)
    assert len(near) > 0
    assert near[:, 0].max() <= 1.0
    # pad reaches across a gap
    padded = bg.crop((1.0, 1.0, -1), (1.1, 1.1, 1), pad=0.5)
    assert len(padded) > 0
    capped = bg.crop((0, 0, -1), (10, 10, 1), budget=7)
    assert len(capped) == 7


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "bg.npz")
    bg = BackgroundStore(path, voxel_m=0.05)
    bg.add(_grid(0, 1))
    n = len(bg)
    bg.save()

    again = BackgroundStore(path, voxel_m=0.05)
    assert len(again) == n
    assert np.array_equal(again.points(), bg.points())
    # Key set was rebuilt: dedup still holds across the reload.
    assert again.add(_grid(0, 1)) == 0


def test_corrupt_or_missing_file_starts_empty(tmp_path):
    path = tmp_path / "bg.npz"
    path.write_bytes(b"not an npz")
    assert len(BackgroundStore(str(path))) == 0
    assert len(BackgroundStore(str(tmp_path / "absent.npz"))) == 0


def test_crop_with_keys_matches_crop(tmp_path):
    bg = BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05)
    bg.add(_grid(0, 1))
    pts, keys = bg.crop_with_keys((0, 0, -1), (0.5, 0.5, 1))
    assert len(pts) == len(keys) > 0
    assert np.array_equal(pts, bg.crop((0, 0, -1), (0.5, 0.5, 1), pad=0.0))


def test_remove_keys_preserves_fifo_and_reopens_cells(tmp_path):
    bg = BackgroundStore(str(tmp_path / "bg.npz"), voxel_m=0.05, max_points=150)
    first, second = _grid(0, 0.5), _grid(1, 1.5)  # 25 cells each
    bg.add(first)
    bg.add(second)
    # Carve a sub-box of the FIRST batch.
    pts, keys = bg.crop_with_keys((0, 0, -1), (0.2, 0.2, 1))
    assert bg.remove_keys(keys) == len(keys) > 0
    assert len(bg) == 50 - len(keys)
    # Carved cells re-open: the same region can be re-learned.
    assert bg.add(first) == len(keys)
    # FIFO order intact: overflow still evicts the oldest surviving cells first.
    survivors_first = bg.points()[0]
    bg.max_points = len(bg) - 5
    bg.add(_grid(3, 3.5)[:5])  # push 5 over → evict 10 oldest... cap delta
    assert not (bg.points() == survivors_first).all(axis=1).any()
    assert bg.remove_keys(np.array([], dtype=np.int64)) == 0


def test_clear_wipes_memory_and_disk(tmp_path):
    path = tmp_path / "bg.npz"
    bg = BackgroundStore(str(path))
    bg.add(_grid(0, 1))
    bg.save()
    assert path.exists()
    bg.clear()
    assert len(bg) == 0
    assert not path.exists()
    # Cleared cells can be re-added.
    assert bg.add(_grid(0, 1)) > 0
