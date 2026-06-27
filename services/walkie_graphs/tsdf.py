"""Volumetric TSDF fusion of a snapshot window into one clean structural cloud.

The headline of the redesign: take the messy RGB-D window with *optimized* poses and
fuse it into a single clean volumetric map via Open3D's tensor ``VoxelBlockGrid``. A
voxel must be seen by ``weight_threshold`` frames to survive extraction, which is what
makes the result clean from messy input (transients / flying pixels never reach the
threshold).

Everything is guarded: no Open3D, no CUDA, or any failure → returns ``None`` and the
build proceeds with object nodes only (the only task consumer needs centroids, not a
surface). Depth is float32 **metres**, so ``depth_scale=1.0``. Extrinsic is the
**inverse** of the camera→map pose.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pcd_ops import resolve_device

try:
    import open3d as _o3d  # noqa: F401
except Exception:  # pragma: no cover
    _o3d = None


def available() -> bool:
    """True when Open3D's tensor geometry is importable (TSDF can run)."""
    return _o3d is not None and hasattr(_o3d, "t")


def fuse(
    snapshots,
    poses,
    *,
    voxel: float = 0.02,
    depth_max: float = 4.0,
    block_resolution: int = 16,
    block_count: int = 40000,
    weight_threshold: float = 3.0,
    device: str | None = None,
    log=print,
) -> Optional[np.ndarray]:
    """Integrate ``snapshots`` (with ``poses``, camera→map) into one ``(M, 3)`` cloud.

    Returns ``None`` if Open3D is unavailable, RGB is missing on any frame, or any
    integrate/extract step fails.
    """
    if not available():
        log("[tsdf] open3d tensor API unavailable — skipping volumetric fusion")
        return None
    if any(getattr(s, "rgb", None) is None for s in snapshots):
        log("[tsdf] needs per-frame RGB (WALKIE_GRAPHS_KEEP_RGB=1) — skipping")
        return None
    o3d = _o3d
    try:
        import open3d.core as o3c

        dev = o3d.core.Device("CUDA:0") if (device or resolve_device()) == "cuda" else o3d.core.Device("CPU:0")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight"),
            attr_dtypes=(o3c.float32, o3c.float32),
            attr_channels=((1), (1)),
            voxel_size=float(voxel),
            block_resolution=int(block_resolution),
            block_count=int(block_count),
            device=dev,
        )
        fx, fy, cx, cy, w, h = snapshots[0].intr
        intr = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3c.float64)
        n_ok = 0
        for s, p in zip(snapshots, poses):
            depth = np.asarray(s.depth, np.float32)
            depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
            d_img = o3d.t.geometry.Image(o3c.Tensor(depth)).to(dev)
            T = np.eye(4)
            T[:3, :3] = p.R
            T[:3, 3] = p.t
            extr = o3c.Tensor(np.linalg.inv(T), o3c.float64)  # world→camera
            try:
                frustum = vbg.compute_unique_block_coordinates(
                    d_img, intr, extr, depth_scale=1.0, depth_max=depth_max
                )
                vbg.integrate(frustum, d_img, intr, extr, depth_scale=1.0, depth_max=depth_max)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — one bad frame must not abort the fuse
                log(f"[tsdf] frame skipped: {e}")
        if n_ok == 0:
            return None
        pcd = vbg.extract_point_cloud(weight_threshold=weight_threshold)
        pts = pcd.point.positions.cpu().numpy().astype(np.float32)
        log(f"[tsdf] fused {n_ok}/{len(snapshots)} frames → {len(pts)} surface points")
        return pts
    except Exception as e:  # noqa: BLE001
        log(f"[tsdf] volumetric fusion failed ({e}) — skipping")
        return None
