"""SnapshotBuffer: round-trip fidelity, ring eviction, ordering, and the build pin.

Pure numpy — synthetic 8x8 frames with packed masks. No cv2/open3d/PIL/chromadb.
Covers the golden cases: depth within 1mm, masks exact, pose/intr exact, detection
metadata exact; ring eviction (len == cap, oldest dropped); ids() ordering;
load_window(n) newest-n oldest-first; and the building() pin deferring eviction.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.walkie_graphs.buffer import Detection, Snapshot, SnapshotBuffer


# ---------------------------------------------------------------------------
# Synthetic frame factory
# ---------------------------------------------------------------------------
def _mask(h=8, w=8, *, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) > 0.5).astype(np.uint8)


def _depth(h=8, w=8, *, seed=0) -> np.ndarray:
    """Depth in metres with a few invalid pixels (0 and NaN) sprinkled in."""
    rng = np.random.default_rng(seed)
    d = (rng.random((h, w)).astype(np.float32) * 3.0 + 0.5)  # 0.5..3.5 m
    d[0, 0] = 0.0  # explicit invalid
    d[1, 1] = np.nan  # explicit invalid
    d[2, 2] = -1.0  # negative -> invalid
    return d


def _snap(ts=1.0, *, seed=0, n_det=2, h=8, w=8, rgb=False, pose=True) -> Snapshot:
    dets = []
    for i in range(n_det):
        dets.append(
            Detection(
                class_name=f"obj{i}",
                class_id=i,
                conf=0.5 + 0.1 * i,
                bbox=(i, i + 1, i + 5, i + 6),
                caption=f"a small obj{i}",
                clip_emb=[float(i), 0.5, -0.25, 1.0],
                mask=_mask(h, w, seed=seed * 10 + i),
            )
        )
    return Snapshot(
        ts=ts,
        depth=_depth(h, w, seed=seed),
        intr=(525.0, 524.0, 319.5, 239.5, float(w), float(h)),
        cam_R=np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        cam_t=np.array([1.5, -2.0, 0.75], dtype=float),
        robot_pose={"x": 1.5, "y": -2.0, "heading": 0.3} if pose else None,
        detections=dets,
        rgb=(np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3) if rgb else None),
    )


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------
def test_round_trip_through_fresh_buffer(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=10)
    snap = _snap(ts=42.5, seed=3)
    sid = buf.append(snap)
    assert isinstance(sid, str) and sid

    # Reopen from disk to prove nothing is cached in RAM.
    buf2 = SnapshotBuffer(tmp_path, cap=10)
    assert buf2.ids() == [sid]
    got = buf2.load(sid)

    assert got.ts == pytest.approx(42.5)
    assert got.intr == snap.intr
    np.testing.assert_array_equal(got.cam_R, snap.cam_R)
    np.testing.assert_array_equal(got.cam_t, snap.cam_t)
    assert got.robot_pose == snap.robot_pose

    # Depth within 1 mm on valid pixels; invalids round-trip to NaN.
    orig, dec = snap.depth, got.depth
    valid = np.isfinite(orig) & (orig > 0)
    assert np.all(np.abs(dec[valid] - orig[valid]) <= 1e-3 + 1e-6)
    assert np.all(np.isnan(dec[~valid]))

    # Detections: metadata exact, masks exact.
    assert len(got.detections) == len(snap.detections)
    for gd, od in zip(got.detections, snap.detections):
        assert gd.class_name == od.class_name
        assert gd.class_id == od.class_id
        assert gd.conf == pytest.approx(od.conf)
        assert gd.bbox == od.bbox
        assert gd.caption == od.caption
        assert gd.clip_emb == pytest.approx(od.clip_emb)
        np.testing.assert_array_equal(gd.mask.astype(np.uint8), od.mask.astype(np.uint8))


def test_keep_rgb_round_trips_only_when_enabled(tmp_path):
    rgb_snap = _snap(seed=1, rgb=True)

    off = SnapshotBuffer(tmp_path / "off", cap=10, keep_rgb=False)
    sid = off.append(rgb_snap)
    assert off.load(sid).rgb is None

    on = SnapshotBuffer(tmp_path / "on", cap=10, keep_rgb=True)
    sid = on.append(rgb_snap)
    got = on.load(sid)
    assert got.rgb is not None
    np.testing.assert_array_equal(got.rgb, rgb_snap.rgb)


def test_bool_mask_round_trips_as_uint8(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=5)
    m = (_mask(seed=7) > 0)  # bool dtype
    snap = _snap(seed=2)
    snap.detections[0].mask = m
    sid = buf.append(snap)
    got = buf.load(sid).detections[0].mask
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, m.astype(np.uint8))


# ---------------------------------------------------------------------------
# Ring eviction + ordering
# ---------------------------------------------------------------------------
def test_ring_eviction_drops_oldest_and_caps_len(tmp_path):
    cap = 5
    buf = SnapshotBuffer(tmp_path, cap=cap)
    ids = [buf.append(_snap(ts=float(i), seed=i)) for i in range(cap + 3)]

    assert len(buf) == cap
    # The last `cap` ids survive, in order; the first 3 are gone.
    surviving = ids[-cap:]
    assert buf.ids() == surviving
    for gone in ids[:3]:
        with pytest.raises((KeyError, FileNotFoundError)):
            buf.load(gone)
    # Sidecars for evicted snapshots are unlinked from disk.
    assert not (tmp_path / f"snap_{ids[0]}.npz").exists()
    assert (tmp_path / f"snap_{surviving[0]}.npz").exists()

    # A fresh buffer over the same dir sees exactly the survivors.
    assert SnapshotBuffer(tmp_path, cap=cap).ids() == surviving


def test_ids_ordered_oldest_to_newest(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=100)
    ids = [buf.append(_snap(ts=float(i), seed=i)) for i in range(6)]
    assert buf.ids() == ids


def test_load_window_returns_newest_n_oldest_first(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=100)
    ids = [buf.append(_snap(ts=float(i), seed=i)) for i in range(6)]

    win = buf.load_window(3)
    assert [round(s.ts) for s in win] == [3, 4, 5]  # newest 3, oldest-first

    assert buf.load_window(0) == []
    assert [round(s.ts) for s in buf.load_window(None)] == [0, 1, 2, 3, 4, 5]
    assert [round(s.ts) for s in buf.load_all()] == [0, 1, 2, 3, 4, 5]
    # Asking for more than present yields everything.
    assert len(buf.load_window(99)) == 6


# ---------------------------------------------------------------------------
# building() pin: defer eviction of pinned ids, then catch up
# ---------------------------------------------------------------------------
def test_building_pin_defers_eviction_then_catches_up(tmp_path):
    cap = 4
    buf = SnapshotBuffer(tmp_path, cap=cap)
    base = [buf.append(_snap(ts=float(i), seed=i)) for i in range(cap)]
    assert len(buf) == cap

    with buf.building() as pinned:
        assert pinned == base  # frozen id list, oldest..newest
        # Append past cap while pinned — overflow must NOT evict pinned ids.
        extra = [buf.append(_snap(ts=float(10 + i), seed=10 + i)) for i in range(3)]
        assert len(buf) == cap + 3
        for sid in base:
            assert (tmp_path / f"snap_{sid}.npz").exists()
            buf.load(sid)  # still readable while pinned

    # Pin released -> catch-up eviction down to cap, keeping the newest.
    assert len(buf) == cap
    survivors = (base + extra)[-cap:]
    assert buf.ids() == survivors
    for gone in (base + extra)[:-cap]:
        assert not (tmp_path / f"snap_{gone}.npz").exists()


def test_building_pin_protects_old_ids_even_when_all_new_are_pinned(tmp_path):
    # Edge: everything currently present is pinned and we overflow heavily.
    cap = 2
    buf = SnapshotBuffer(tmp_path, cap=cap)
    base = [buf.append(_snap(ts=float(i), seed=i)) for i in range(cap)]

    with buf.building() as pinned:
        assert set(pinned) == set(base)
        for i in range(5):
            buf.append(_snap(ts=float(100 + i), seed=100 + i))
        # No pinned base id may be evicted while the pin is held.
        assert set(base).issubset(set(buf.ids()))

    assert len(buf) == cap


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------
def test_missing_sidecar_is_skipped_on_window_load(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=10)
    ids = [buf.append(_snap(ts=float(i), seed=i)) for i in range(3)]

    # Corrupt one sidecar by removing it out from under the index.
    (tmp_path / f"snap_{ids[1]}.npz").unlink()

    win = buf.load_window(None)
    assert [round(s.ts) for s in win] == [0, 2]  # the missing one is skipped
    with pytest.raises(FileNotFoundError):
        buf.load(ids[1])


def test_half_written_index_line_is_tolerated(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=10)
    good = buf.append(_snap(ts=1.0, seed=1))

    # Append a junk trailing line as if a crash truncated a write.
    with (tmp_path / "index.jsonl").open("a") as f:
        f.write('{"id": "broken", "ts": 2.0,\n')  # invalid JSON

    reopened = SnapshotBuffer(tmp_path, cap=10)
    assert reopened.ids() == [good]
    reopened.load(good)


def test_snapshot_with_no_detections(tmp_path):
    buf = SnapshotBuffer(tmp_path, cap=10)
    sid = buf.append(_snap(seed=4, n_det=0))
    got = buf.load(sid)
    assert got.detections == []
    assert got.depth.shape == (8, 8)
