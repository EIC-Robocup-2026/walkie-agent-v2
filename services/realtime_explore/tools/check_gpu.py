"""Diagnose whether Open3D can (and does) use the GPU on this machine.

Run with::

    uv run python -m services.realtime_explore.tools.check_gpu

Reports, in order:

1. The Open3D build's CUDA support and visible CUDA devices.
2. A real tensor op executed on the GPU (proof it works, not just "available").
3. ICP timings across the three possible engines:
   - **legacy CPU** (``o3d.pipelines.registration``) — what walkie_graphs uses today;
     this pipeline is CPU-only *by design*, a GPU cannot accelerate it.
   - **tensor CPU** (``o3d.t.pipelines.registration``)
   - **tensor CUDA** — the only GPU ICP path; requires a CUDA build + device, and
     porting walkie_graphs to the tensor API to use it.

The verdict at the end says whether a GPU port would actually pay off for our cloud
sizes (a few thousand points — small enough that transfer overhead often wins).
"""

from __future__ import annotations

import shutil
import subprocess
import time

import numpy as np


def _gpu_name() -> str:
    if not shutil.which("nvidia-smi"):
        return "(nvidia-smi not found — no NVIDIA driver?)"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "(nvidia-smi gave no output)"
    except Exception as e:  # noqa: BLE001
        return f"(nvidia-smi failed: {e})"


def _make_clouds(n: int = 6000):
    rng = np.random.default_rng(3)
    floor = np.stack([rng.uniform(0, 0.7, n), rng.uniform(0, 0.7, n), rng.normal(0, 0.005, n)], axis=1)
    wall = np.stack([rng.uniform(0, 0.7, n), rng.normal(0, 0.005, n), rng.uniform(0, 0.8, n)], axis=1)
    target = np.vstack([floor, wall])
    source = (target + np.array([0.05, 0.03, 0.02]))[::2]
    return source, target


def _bench(label, fn, repeat=10):
    fn()  # warmup (JIT/transfer/allocator)
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    print(f"  {label}: {(time.perf_counter() - t) / repeat * 1000:7.1f} ms")


def main() -> None:
    print("=" * 72)
    print("Open3D GPU diagnostic")
    print("=" * 72)

    print(f"\nSystem GPU      : {_gpu_name()}")

    import open3d as o3d

    print(f"Open3D version  : {o3d.__version__}")

    # ------------------------------------------------------------------
    # 1. Does this BUILD have CUDA, and is a device visible?
    # ------------------------------------------------------------------
    cuda_built = hasattr(o3d.core, "cuda")
    cuda_available = bool(cuda_built and o3d.core.cuda.is_available())
    device_count = o3d.core.cuda.device_count() if cuda_built else 0
    print(f"CUDA in build   : {cuda_built}")
    print(f"CUDA available  : {cuda_available}  (devices: {device_count})")

    # ------------------------------------------------------------------
    # 2. Prove it: run a real tensor op on the GPU.
    # ------------------------------------------------------------------
    gpu_works = False
    if cuda_available:
        try:
            dev = o3d.core.Device("CUDA:0")
            a = o3d.core.Tensor(np.random.rand(512, 512).astype(np.float32), device=dev)
            b = (a @ a).cpu().numpy()  # matmul on GPU, copy back
            assert b.shape == (512, 512)
            gpu_works = True
            print("GPU tensor op   : OK (512x512 matmul executed on CUDA:0)")
        except Exception as e:  # noqa: BLE001
            print(f"GPU tensor op   : FAILED — {type(e).__name__}: {e}")
            print("                  (CUDA visible but unusable — often a too-new GPU")
            print("                   architecture for the wheel's CUDA toolkit)")
    else:
        print("GPU tensor op   : skipped (no CUDA device available to Open3D)")

    # ------------------------------------------------------------------
    # 3. ICP engines, timed on walkie-scale clouds.
    # ------------------------------------------------------------------
    source, target = _make_clouds()
    print(f"\nICP benchmark ({len(source)} vs {len(target)} points, 30 iterations):")

    # legacy pipeline — what walkie_graphs uses; CPU-only by design
    reg = o3d.pipelines.registration
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    tp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    _bench(
        "legacy  CPU  (o3d.pipelines — current walkie_graphs path)",
        lambda: reg.registration_icp(
            sp, tp, 0.1, np.eye(4), reg.TransformationEstimationPointToPoint(),
            reg.ICPConvergenceCriteria(max_iteration=30),
        ),
    )

    # tensor pipeline, CPU
    treg = o3d.t.pipelines.registration
    src_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(source.astype(np.float32)))
    tgt_t = o3d.t.geometry.PointCloud(o3d.core.Tensor(target.astype(np.float32)))
    try:
        _bench(
            "tensor  CPU  (o3d.t.pipelines)",
            lambda: treg.icp(
                src_t, tgt_t, 0.1, o3d.core.Tensor(np.eye(4)),
                treg.TransformationEstimationPointToPoint(),
                treg.ICPConvergenceCriteria(max_iteration=30),
            ),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  tensor  CPU : FAILED — {type(e).__name__}: {e}")

    # tensor pipeline, CUDA — the only GPU ICP
    if gpu_works:
        try:
            dev = o3d.core.Device("CUDA:0")
            src_g = src_t.to(dev)
            tgt_g = tgt_t.to(dev)
            _bench(
                "tensor  CUDA (o3d.t.pipelines on GPU)",
                lambda: treg.icp(
                    src_g, tgt_g, 0.1, o3d.core.Tensor(np.eye(4)),
                    treg.TransformationEstimationPointToPoint(),
                    treg.ICPConvergenceCriteria(max_iteration=30),
                ),
            )
        except Exception as e:  # noqa: BLE001
            print(f"  tensor  CUDA: FAILED — {type(e).__name__}: {e}")
    else:
        print("  tensor  CUDA: skipped (GPU unusable per step 2)")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\nVerdict:")
    print("  - walkie_graphs' capture registration (pcd_ops.icp) uses the tensor API")
    print("    on CUDA when WALKIE_GRAPHS_O3D_DEVICE resolves to cuda, with a legacy")
    print("    CPU fallback — the timings above show what each engine costs here.")
    if gpu_works:
        print("  - This machine CAN run Open3D tensor ops on the GPU — compare the CUDA")
        print("    vs CPU timings above: at our small cloud sizes (≤ a few thousand")
        print("    points after subsampling), host↔GPU transfer often eats the gain.")
    elif cuda_available:
        print("  - CUDA is visible but unusable (see step 2) — a GPU port would not run.")
    else:
        print("  - No CUDA device is usable from this Open3D build — GPU is not an")
        print("    option here; CPU-side gating (cooldown/skip) is the right lever.")


if __name__ == "__main__":
    main()
