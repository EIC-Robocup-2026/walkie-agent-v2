"""Device-aware point-cloud ops: Open3D tensor API on CUDA, CPU fallbacks.

The capture pipeline registers each capture's full cloud against the map and
voxelizes large background remainders — clouds big enough that the GPU pays
for the host↔device transfer (per ``tools/check_gpu.py``, small clouds don't).
This module is the single place that decision lives:

- ``resolve_device()`` picks ``"cuda"`` or ``"cpu"`` from
  ``WALKIE_GRAPHS_O3D_DEVICE`` (``auto`` | ``cpu`` | ``cuda``); ``auto`` takes
  the GPU iff the installed Open3D build actually sees a CUDA device.
- Every op degrades: tensor-CUDA → legacy/numpy CPU → identity. A machine
  without open3d (or without a GPU) runs the same code with the same
  semantics, just slower — tests never need a GPU.

Open3D is imported lazily through :func:`dbscan._open3d` (the import costs
~2 s; tests monkeypatch ``dbscan._O3D`` to force the no-open3d paths).
"""

from __future__ import annotations

import os

import numpy as np

from .dbscan import _open3d
from .geometry import voxel_downsample as _voxel_downsample_np

# Resolved device per requested mode ("auto"/"cpu"/"cuda" -> "cpu"/"cuda").
# Keyed by the env value so tests that flip WALKIE_GRAPHS_O3D_DEVICE re-resolve.
_DEVICE_CACHE: dict[str, str] = {}

# Below this size the host->GPU transfer outweighs the voxelization win.
_GPU_VOXEL_MIN_POINTS = 50_000


def resolve_device() -> str:
    """The compute device for tensor ops: ``"cuda"`` or ``"cpu"``.

    ``WALKIE_GRAPHS_O3D_DEVICE=auto`` (default) picks CUDA iff the installed
    Open3D build reports an available device; ``cuda`` degrades silently to
    ``cpu`` when unavailable (a robot whose GPU is busy/absent must still map).
    Never raises.
    """
    want = os.getenv("WALKIE_GRAPHS_O3D_DEVICE", "auto").strip().lower()
    cached = _DEVICE_CACHE.get(want)
    if cached is not None:
        return cached
    device = "cpu"
    if want in ("auto", "cuda"):
        o3d = _open3d()
        if o3d:
            try:
                if getattr(o3d.core, "cuda", None) and o3d.core.cuda.is_available():
                    device = "cuda"
            except Exception:  # noqa: BLE001 — CUDA probing must never crash the loop
                device = "cpu"
    _DEVICE_CACHE[want] = device
    return device


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_corr_dist: float,
    *,
    max_iter: int = 30,
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    """Rigid point-to-point ICP of ``source`` onto ``target``.

    Returns ``(T, fitness)`` — a 4×4 transform (apply with
    :func:`apply_transform`) and the fraction of source points with a
    correspondence within ``max_corr_dist``. Point-to-point needs no normals.
    Identity + 0.0 when disabled (``max_corr_dist <= 0``), inputs are
    degenerate, open3d is missing, or registration fails — callers gate on
    fitness, so a failed solve simply means "no correction".
    """
    eye = np.eye(4)
    src = np.asarray(source, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if max_corr_dist <= 0 or len(src) < 3 or len(tgt) < 3:
        return eye, 0.0
    o3d = _open3d()
    if not o3d:
        return eye, 0.0
    if (device or resolve_device()) == "cuda":
        try:
            return _icp_tensor(o3d, src, tgt, max_corr_dist, max_iter, "CUDA:0")
        except Exception:  # noqa: BLE001 — GPU hiccup falls back to the CPU solve
            pass
    try:
        return _icp_legacy(o3d, src, tgt, max_corr_dist, max_iter)
    except Exception:  # noqa: BLE001
        return eye, 0.0


def _icp_tensor(o3d, src, tgt, max_corr_dist, max_iter, device_str):
    core = o3d.core
    treg = o3d.t.pipelines.registration
    dev = core.Device(device_str)
    sp = o3d.t.geometry.PointCloud(core.Tensor(src, device=dev))
    tp = o3d.t.geometry.PointCloud(core.Tensor(tgt, device=dev))
    res = treg.icp(
        sp,
        tp,
        float(max_corr_dist),
        core.Tensor(np.eye(4)),
        treg.TransformationEstimationPointToPoint(),
        treg.ICPConvergenceCriteria(max_iteration=int(max_iter)),
    )
    return res.transformation.cpu().numpy().astype(np.float64), float(res.fitness)


def _icp_legacy(o3d, src, tgt, max_corr_dist, max_iter):
    reg = o3d.pipelines.registration
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src.astype(np.float64)))
    tp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt.astype(np.float64)))
    res = reg.registration_icp(
        sp,
        tp,
        float(max_corr_dist),
        np.eye(4),
        reg.TransformationEstimationPointToPoint(),
        reg.ICPConvergenceCriteria(max_iteration=int(max_iter)),
    )
    return np.asarray(res.transformation, dtype=np.float64), float(res.fitness)


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to an ``(N, 3)`` cloud, returning float32."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return pts.reshape(0, 3)
    T = np.asarray(T, dtype=np.float64)
    return (pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)).astype(
        np.float32
    )


def rotation_angle_deg(T: np.ndarray) -> float:
    """The rotation magnitude of a 4×4 transform, in degrees."""
    R = np.asarray(T, dtype=np.float64)[:3, :3]
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def voxel_downsample(
    points: np.ndarray, voxel: float, *, device: str | None = None
) -> np.ndarray:
    """Voxel-grid downsample (one mean point per cell), GPU when it pays.

    Same semantics as :func:`geometry.voxel_downsample`; clouds of at least
    ``_GPU_VOXEL_MIN_POINTS`` go through the tensor API on CUDA when resolved,
    everything else (and every failure) uses the numpy path.
    """
    pts = np.asarray(points)
    if voxel is None or voxel <= 0 or len(pts) == 0:
        return pts
    if (device or resolve_device()) == "cuda" and len(pts) >= _GPU_VOXEL_MIN_POINTS:
        o3d = _open3d()
        if o3d:
            try:
                core = o3d.core
                cloud = o3d.t.geometry.PointCloud(
                    core.Tensor(pts.astype(np.float32), device=core.Device("CUDA:0"))
                )
                down = cloud.voxel_down_sample(float(voxel))
                return down.point.positions.cpu().numpy().astype(np.float32)
            except Exception:  # noqa: BLE001 — GPU hiccup falls back to numpy
                pass
    return _voxel_downsample_np(pts, voxel)


def subsample(points: np.ndarray, budget: int) -> np.ndarray:
    """Cap a cloud to ``budget`` points by deterministic uniform stride.

    Stride (not random choice) preserves spatial coverage of an ordered
    deprojection and keeps results reproducible. No-op when within budget or
    ``budget <= 0``.
    """
    pts = np.asarray(points)
    if budget <= 0 or len(pts) <= budget:
        return pts
    idx = np.linspace(0, len(pts) - 1, budget).astype(np.int64)
    return pts[idx]


def warmup() -> str:
    """Pay the open3d import + first-solve JIT/CUDA-context cost up front.

    Returns the resolved device so the caller can log it. Best-effort: any
    failure leaves later calls to their own fallbacks.
    """
    device = resolve_device()
    try:
        rng = np.random.default_rng(0)
        pts = (rng.normal(size=(256, 3)) * 0.1).astype(np.float32)
        icp(pts, pts + 0.01, 0.1, max_iter=5)
    except Exception:  # noqa: BLE001 — warmup is best-effort
        pass
    return device
