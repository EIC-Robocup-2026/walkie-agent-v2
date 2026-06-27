"""Offline batch build: a window of snapshots → fused object observations (+ map).

This is the v2 replacement for v1's per-frame ``ingest_frame`` fold. It runs over a
whole window at once, so pose error and instance identity are each resolved **once**:

1. ``refine_poses`` → one clean camera→map pose per snapshot (``baseline`` = nav).
2. lift every detection's mask with its frame's optimized pose
   (:func:`interfaces.perception.geometry.deproject_mask`, the same flying-pixel cleanup
   v1 uses) → a world-frame :class:`~services.walkie_graphs.associate.Observation`.
3. ``associate`` → constrained-agglomerative object clusters
   (:class:`~services.walkie_graphs.associate.ObjectObservation`).
4. (optional) ``tsdf.fuse`` → one clean volumetric structural cloud.

The caller (the build worker in :mod:`~services.walkie_graphs.service_v2`) merges the
observations into the persisted :class:`~services.walkie_graphs.scene.SceneStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from interfaces.perception.geometry import (
    CameraPose,
    Intrinsics,
    deproject_mask,
    depth_discontinuity_mask,
)
from . import poses as _poses
from . import tsdf as _tsdf
from .associate import Observation, ObjectObservation, associate


@dataclass
class BuildResult:
    observations: list[ObjectObservation]
    structural_cloud: Optional[np.ndarray] = None
    poses: list[CameraPose] = field(default_factory=list)
    n_snapshots: int = 0
    n_detections: int = 0


def _lift_snapshot(snap, pose, *, voxel_m, max_points, erode_px, max_depth, sor_k,
                   edge_thresh, min_points) -> list[Observation]:
    fx, fy, cx, cy, w, h = snap.intr
    intr = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(w), height=int(h))
    intr = intr.scaled_to(snap.depth.shape[1], snap.depth.shape[0])
    edge = depth_discontinuity_mask(snap.depth, edge_thresh) if edge_thresh > 0 else None
    out: list[Observation] = []
    for det in snap.detections:
        pts = deproject_mask(
            det.mask, snap.depth, intr, pose,
            voxel=voxel_m, max_points=max_points, erode_px=erode_px,
            edge_mask=edge, max_depth=max_depth, sor_k=sor_k,
        )
        if len(pts) < min_points:
            continue
        out.append(Observation(
            class_name=det.class_name, class_id=det.class_id, conf=float(det.conf),
            bbox=tuple(det.bbox), caption=det.caption or "",
            clip_emb=list(det.clip_emb or []), ts=float(snap.ts), points=pts,
        ))
    return out


def build_scene(
    snapshots,
    *,
    pose_mode: str = "baseline",
    do_tsdf: bool = False,
    # lift
    voxel_m: float = 0.02,
    max_points: int = 2000,
    erode_px: int = 2,
    max_depth: float = 4.0,
    sor_k: int = 0,
    edge_thresh: float = 0.05,
    min_points: int = 50,
    # association
    overlap_min: float = 0.2,
    clip_min: float = 0.85,
    max_dist_m: float = 0.5,
    require_same_class: bool = True,
    default_max_extent: float = 2.5,
    max_extent_by_class: dict[str, float] | None = None,
    # tsdf
    tsdf_voxel: float = 0.02,
    log=print,
) -> BuildResult:
    """Build object observations (and optionally a structural cloud) from a window."""
    snaps = list(snapshots)
    if not snaps:
        return BuildResult(observations=[])

    poses = _poses.refine_poses(snaps, mode=pose_mode, max_depth=max_depth, log=log)

    observations: list[Observation] = []
    for snap, pose in zip(snaps, poses):
        observations.extend(_lift_snapshot(
            snap, pose, voxel_m=voxel_m, max_points=max_points, erode_px=erode_px,
            max_depth=max_depth, sor_k=sor_k, edge_thresh=edge_thresh, min_points=min_points,
        ))

    clusters = associate(
        observations,
        overlap_min=overlap_min, clip_min=clip_min, max_dist_m=max_dist_m,
        require_same_class=require_same_class, voxel_m=voxel_m,
        default_max_extent=default_max_extent, max_extent_by_class=max_extent_by_class,
    )

    structural = None
    if do_tsdf:
        structural = _tsdf.fuse(snaps, poses, voxel=tsdf_voxel, depth_max=max_depth, log=log)

    log(f"[build] {len(snaps)} snapshots, {len(observations)} lifts → {len(clusters)} objects"
        + (f", {len(structural)} surface pts" if structural is not None else ""))
    return BuildResult(
        observations=clusters, structural_cloud=structural, poses=poses,
        n_snapshots=len(snaps), n_detections=len(observations),
    )
