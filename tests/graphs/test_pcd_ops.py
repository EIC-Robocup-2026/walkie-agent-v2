"""Device-aware point-cloud backend: device resolution, ICP, transforms, voxel."""

from __future__ import annotations

import numpy as np
import pytest

from services.realtime_explore import pcd_ops
from interfaces.perception.geometry import voxel_downsample as voxel_np

try:
    import open3d as _o3d

    _CUDA = bool(getattr(_o3d.core, "cuda", None) and _o3d.core.cuda.is_available())
except Exception:  # noqa: BLE001 — no open3d on this machine
    _o3d = None
    _CUDA = False


@pytest.fixture(autouse=True)
def _fresh_device_cache():
    pcd_ops._DEVICE_CACHE.clear()
    yield
    pcd_ops._DEVICE_CACHE.clear()


def _corner_cloud(n=400, seed=3):
    """An L-shaped corner of aperiodic points (pins all translations for ICP)."""
    rng = np.random.default_rng(seed)
    floor = np.stack([rng.uniform(0, 0.25, n), rng.uniform(0, 0.25, n), np.zeros(n)], axis=1)
    wall = np.stack([rng.uniform(0, 0.25, n), np.zeros(n), rng.uniform(0, 0.25, n)], axis=1)
    return np.vstack([floor, wall]).astype(np.float32)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------
def test_resolve_device_cpu_override(monkeypatch):
    monkeypatch.setenv("WALKIE_EXPLORE_O3D_DEVICE", "cpu")
    assert pcd_ops.resolve_device() == "cpu"


def test_resolve_device_without_open3d_is_cpu(monkeypatch):
    import interfaces.perception.dbscan as dbscan_mod

    monkeypatch.setattr(dbscan_mod, "_O3D", False)
    monkeypatch.setenv("WALKIE_EXPLORE_O3D_DEVICE", "auto")
    assert pcd_ops.resolve_device() == "cpu"
    # An explicit cuda request degrades to cpu instead of raising.
    pcd_ops._DEVICE_CACHE.clear()
    monkeypatch.setenv("WALKIE_EXPLORE_O3D_DEVICE", "cuda")
    assert pcd_ops.resolve_device() == "cpu"


def test_resolve_device_auto_matches_build(monkeypatch):
    pytest.importorskip("open3d")
    monkeypatch.setenv("WALKIE_EXPLORE_O3D_DEVICE", "auto")
    assert pcd_ops.resolve_device() == ("cuda" if _CUDA else "cpu")


# ---------------------------------------------------------------------------
# icp
# ---------------------------------------------------------------------------
def test_icp_recovers_pose_offset():
    pytest.importorskip("open3d")
    target = _corner_cloud()
    offset = np.array([0.05, 0.03, 0.02], dtype=np.float32)
    source = target + offset
    T, fitness = pcd_ops.icp(source, target, max_corr_dist=0.1)
    assert fitness > 0.9
    aligned = pcd_ops.apply_transform(source, T)
    assert np.abs(aligned - target).max() < 0.005  # 5cm error reduced to <5mm


def test_icp_disabled_or_degenerate_is_identity():
    target = _corner_cloud()
    for src, tgt, dist in [
        (target + 0.05, target, 0.0),  # disabled
        (np.zeros((0, 3)), target, 0.1),  # empty source
        (target, np.zeros((2, 3)), 0.1),  # degenerate target
    ]:
        T, fit = pcd_ops.icp(src, tgt, max_corr_dist=dist)
        assert np.array_equal(T, np.eye(4)) and fit == 0.0


def test_icp_identity_without_open3d(monkeypatch):
    import interfaces.perception.dbscan as dbscan_mod

    monkeypatch.setattr(dbscan_mod, "_O3D", False)
    target = _corner_cloud()
    T, fit = pcd_ops.icp(target + 0.05, target, max_corr_dist=0.1)
    assert np.array_equal(T, np.eye(4)) and fit == 0.0


@pytest.mark.skipif(not _CUDA, reason="no CUDA device available to Open3D")
def test_icp_cuda_matches_cpu():
    target = _corner_cloud()
    source = target + np.array([0.05, 0.03, 0.02], dtype=np.float32)
    T_cpu, fit_cpu = pcd_ops.icp(source, target, max_corr_dist=0.1, device="cpu")
    T_gpu, fit_gpu = pcd_ops.icp(source, target, max_corr_dist=0.1, device="cuda")
    assert fit_cpu > 0.9 and fit_gpu > 0.9
    # Same correction within a millimetre — the engines may differ in iteration detail.
    assert np.abs(T_cpu[:3, 3] - T_gpu[:3, 3]).max() < 1e-3


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------
def test_apply_transform_translation_and_empty():
    T = np.eye(4)
    T[:3, 3] = (1.0, -2.0, 0.5)
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    out = pcd_ops.apply_transform(pts, T)
    assert np.allclose(out, pts + np.array([1.0, -2.0, 0.5], dtype=np.float32))
    assert pcd_ops.apply_transform(np.zeros((0, 3)), T).shape == (0, 3)


def test_rotation_angle_deg():
    a = np.radians(10.0)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    assert pcd_ops.rotation_angle_deg(T) == pytest.approx(10.0, abs=1e-6)
    assert pcd_ops.rotation_angle_deg(np.eye(4)) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# voxel_downsample / subsample
# ---------------------------------------------------------------------------
def test_voxel_cpu_matches_geometry_helper():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, size=(5000, 3)).astype(np.float32)
    out = pcd_ops.voxel_downsample(pts, 0.05, device="cpu")
    assert np.array_equal(out, voxel_np(pts, 0.05))


@pytest.mark.skipif(not _CUDA, reason="no CUDA device available to Open3D")
def test_voxel_cuda_parity():
    rng = np.random.default_rng(8)
    pts = rng.uniform(0, 2, size=(80_000, 3)).astype(np.float32)
    cpu = pcd_ops.voxel_downsample(pts, 0.05, device="cpu")
    gpu = pcd_ops.voxel_downsample(pts, 0.05, device="cuda")
    # Grid origins differ between backends; cell population should agree closely.
    assert abs(len(gpu) - len(cpu)) / len(cpu) < 0.05
    assert np.allclose(gpu.mean(axis=0), cpu.mean(axis=0), atol=0.01)


def test_subsample_stride_and_noop():
    pts = np.arange(300, dtype=np.float32).reshape(100, 3)
    out = pcd_ops.subsample(pts, 10)
    assert len(out) == 10
    assert np.array_equal(out[0], pts[0]) and np.array_equal(out[-1], pts[-1])
    assert pcd_ops.subsample(pts, 0) is pts
    assert pcd_ops.subsample(pts, 200) is pts


def test_warmup_returns_device():
    assert pcd_ops.warmup() in ("cpu", "cuda")
