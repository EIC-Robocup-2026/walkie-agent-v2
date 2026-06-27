"""Batch pose refinement — turn a window of noisy nav poses into clean ones.

The v2 build runs *offline-ish* over a window of buffered RGB-D snapshots, so it can
reconcile pose error **once over the whole window** instead of per frame (the v1
sin). Two modes:

- ``"baseline"`` (default) — trust the robot's nav/TF pose as captured. Returns each
  snapshot's ``(R, t)`` untouched. No Open3D, no risk. This is the permanent default
  until a replayed-buffer measurement proves ``"auto"`` actually wins (a bad pose-graph
  can make poses *worse* than settled nav — see ``docs/WALKIE_GRAPHS.md``).
- ``"auto"`` — Open3D multiway registration: pairwise RGB-D odometry between nearby
  frames + sparse loop closures → one ``PoseGraph`` → ``global_optimization``
  (Levenberg–Marquardt), anchored at frame 0's nav pose so the result stays in the
  navigable map frame. Every Open3D call is guarded and every edge is sanity-bounded
  against the nav delta; **any** failure falls back to that frame's nav pose, so
  ``auto`` can never disconnect the graph or drift past the nav prior.

``auto`` needs per-frame RGB (``WALKIE_GRAPHS_KEEP_RGB=1``); without it, it degrades to
``baseline``.
"""

from __future__ import annotations

import numpy as np

from interfaces.perception.geometry import CameraPose
from .pcd_ops import resolve_device

try:  # Open3D is optional at import time; auto-mode guards every use.
    import open3d as _o3d  # noqa: F401
except Exception:  # pragma: no cover - depends on the box
    _o3d = None


def _open3d():
    return _o3d


def refine_poses(
    snapshots,
    *,
    mode: str = "baseline",
    device: str | None = None,
    max_depth: float = 4.0,
    depth_diff: float = 0.07,
    loop_radius: float = 0.6,
    max_loops: int = 40,
    nav_trans_tol: float = 0.5,
    nav_rot_tol_deg: float = 30.0,
    log=print,
) -> list[CameraPose]:
    """Return one :class:`CameraPose` (R, t; camera→map) per snapshot.

    ``baseline`` echoes the captured nav poses; ``auto`` runs Open3D multiway
    registration seeded by them. Never raises — degrades to the nav poses.
    """
    nav = [CameraPose(R=np.asarray(s.cam_R, float), t=np.asarray(s.cam_t, float)) for s in snapshots]
    if mode != "auto" or len(snapshots) < 3:
        return nav
    o3d = _open3d()
    if o3d is None:
        log("[poses] open3d unavailable — using nav poses (baseline)")
        return nav
    if any(getattr(s, "rgb", None) is None for s in snapshots):
        log("[poses] auto mode needs per-frame RGB (WALKIE_GRAPHS_KEEP_RGB=1) — using nav poses")
        return nav
    try:
        return _refine_auto(
            o3d, snapshots, nav,
            device=device or resolve_device(),
            max_depth=max_depth, depth_diff=depth_diff,
            loop_radius=loop_radius, max_loops=max_loops,
            nav_trans_tol=nav_trans_tol, nav_rot_tol_deg=nav_rot_tol_deg,
            log=log,
        )
    except Exception as e:  # noqa: BLE001 — any failure → nav poses, never wreck the map
        log(f"[poses] auto refinement failed ({e}) — using nav poses")
        return nav


# ---------------------------------------------------------------------------
# auto-mode internals (Open3D)
# ---------------------------------------------------------------------------
def _pose_mat(p: CameraPose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = p.R
    T[:3, 3] = p.t
    return T


def _rot_angle_deg(R: np.ndarray) -> float:
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _intrinsic_o3d(o3d, intr):
    fx, fy, cx, cy, w, h = intr
    return o3d.camera.PinholeCameraIntrinsic(int(w), int(h), fx, fy, cx, cy)


def _rgbd(o3d, s, max_depth):
    color = o3d.geometry.Image(np.ascontiguousarray(s.rgb).astype(np.uint8))
    depth = np.asarray(s.depth, np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=True,
    )


def _odometry(o3d, rgbd_a, rgbd_b, intr_o3d, init):
    """Pairwise RGB-D odometry b←a. Returns (ok, T_ab, info) or (False, None, None)."""
    opt = o3d.pipelines.odometry.OdometryOption()
    ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_a, rgbd_b, intr_o3d, init,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), opt,
    )
    return ok, T, info


def _refine_auto(o3d, snaps, nav, *, device, max_depth, depth_diff,
                 loop_radius, max_loops, nav_trans_tol, nav_rot_tol_deg, log):
    n = len(snaps)
    reg = o3d.pipelines.registration
    rgbds = [_rgbd(o3d, s, max_depth) for s in snaps]
    intr0 = _intrinsic_o3d(o3d, snaps[0].intr)
    nav_T = [_pose_mat(p) for p in nav]                  # camera→map
    nav_inv = [np.linalg.inv(T) for T in nav_T]

    pg = reg.PoseGraph()
    # Node poses are stored as map→camera (Open3D convention: node.pose = T_world_cam
    # used as extrinsic via inverse). We seed absolute nodes from nav.
    pg.nodes.append(reg.PoseGraphNode(nav_T[0]))
    for i in range(1, n):
        pg.nodes.append(reg.PoseGraphNode(nav_T[i]))

    def add_edge(i, j, uncertain):
        # nav-relative initial guess: map a-cam points into b-cam.
        init = nav_inv[j] @ nav_T[i]
        ok, T, info = _odometry(o3d, rgbds[i], rgbds[j], intr0, init)
        if not ok:
            return False
        # Sanity-bound the solved edge against the nav delta: reject a confident-but-
        # wrong odometry result so a bad (loop) edge can't drag good nodes.
        d = np.linalg.norm(T[:3, 3] - init[:3, 3])
        dr = _rot_angle_deg(T[:3, :3] @ np.linalg.inv(init[:3, :3]))
        if d > nav_trans_tol or dr > nav_rot_tol_deg:
            return False
        pg.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=uncertain))
        return True

    # Sequential (odometry) edges — uncertain=False.
    for i in range(n - 1):
        add_edge(i, i + 1, uncertain=False)

    # Sparse loop-closure edges between spatially-near non-adjacent frames.
    cam_xyz = np.array([p.t for p in nav])
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cam_xyz)
        pairs = sorted(tree.query_pairs(loop_radius))
    except Exception:  # noqa: BLE001
        pairs = []
    loops = 0
    for i, j in pairs:
        if j - i <= 1 or loops >= max_loops:
            continue
        if add_edge(i, j, uncertain=True):
            loops += 1
    log(f"[poses] auto: {n} frames, {n - 1} seq + {loops} loop edges")

    opt = reg.GlobalOptimizationOption(
        max_correspondence_distance=depth_diff, edge_prune_threshold=0.25, reference_node=0,
    )
    reg.global_optimization(
        pg, reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(), opt,
    )

    out: list[CameraPose] = []
    for i in range(n):
        T = np.asarray(pg.nodes[i].pose)            # camera→map (optimized)
        R, t = T[:3, :3], T[:3, 3]
        # Final guard: if the optimized pose wandered past the nav tolerance, keep nav.
        if (np.linalg.norm(t - nav_T[i][:3, 3]) > nav_trans_tol
                or _rot_angle_deg(R @ nav_inv[i][:3, :3]) > nav_rot_tol_deg):
            out.append(nav[i])
        else:
            out.append(CameraPose(R=R.copy(), t=t.copy()))
    return out
