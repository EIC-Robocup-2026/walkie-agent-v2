"""Best-of-N grasp selection — snap the camera a few times, keep the best grasp.

GraspNet only ever sees a camera-**optical** cloud (X-right, Y-down, Z-forward),
which is what keeps it in-distribution. This skill maps the winning grasp back to
the **map frame** using the snapshot's frozen capture-time pose, so callers get a
map-frame grasp point (and a backed-off pre-grasp point) ready to hand to the arm.

    cand = grasp_object(ctx, ["red can"], attempts=5, approach_preference="side")
    if cand:
        ctx.goto_pregrasp(cand.pregrasp_xyz)   # caller's arm/nav logic
        ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tasks.base import TaskContext


@dataclass
class GraspCandidate:
    """The best grasp found, expressed in the **map frame**.

    ``grasp_xyz`` is the gripper-closing centre; ``pregrasp_xyz`` is that point
    backed off ``standoff_m`` along the *negative* approach direction — where the
    gripper should arrive before driving straight in. ``rotation`` is the 3x3 grasp
    frame in the map frame (column 0 = approach/travel, column 1 = closing/spread);
    ``approach`` is its unit approach axis. ``score`` is GraspNet's quality and
    ``width`` the gripper opening in metres.
    """

    grasp_xyz: tuple[float, float, float]
    pregrasp_xyz: tuple[float, float, float]
    rotation: np.ndarray  # (3, 3) in the map frame
    approach: np.ndarray  # (3,) unit approach direction in the map frame
    width: float
    score: float


def _to_map_frame(snap, g, standoff_m: float) -> GraspCandidate:
    """Lift one GraspNet pose (optical frame) into the snapshot's map frame."""
    R_cam, t_cam = snap.cam.R, snap.cam.t
    grasp = R_cam @ np.asarray(g.translation, dtype=float) + t_cam
    R_map = R_cam @ g.rotation
    approach = R_map[:, 2]  # unit travel direction toward the object
    pregrasp = grasp - approach * standoff_m
    return GraspCandidate(
        grasp_xyz=(float(grasp[0]), float(grasp[1]), float(grasp[2])),
        pregrasp_xyz=(float(pregrasp[0]), float(pregrasp[1]), float(pregrasp[2])),
        rotation=R_map,
        approach=approach,
        width=float(g.width),
        score=float(g.score),
    )


def grasp_object(
    ctx: TaskContext,
    prompts: list[str],
    *,
    attempts: int = 5,
    standoff_m: float = 0.10,
    voxel: float = 0.005,
    erode_px: int = 5,
    min_points: int = 50,
    antipodal: bool = True,
    approach_preference: str = "none",
    approach_weight: float | None = None,
) -> GraspCandidate | None:
    """Best-of-N grasp for the first object matching *prompts*, in the map frame.

    Captures up to *attempts* snapshots; on each it runs masked open-vocab
    detection for *prompts*, lifts the top detection's mask to a camera-optical
    cloud, and asks GraspNet for the single best grasp. The highest-scoring grasp
    across all attempts wins, mapped to the map frame against the geometry of the
    very snapshot it came from (accurate even after detection/GraspNet latency).

    Args:
        ctx: Task context (camera, AI client).
        prompts: Open-vocab detector prompts for the target (e.g. ``["red can"]``).
        attempts: How many snapshots to take and score (best-of-N).
        standoff_m: Pre-grasp back-off distance along the approach axis (metres).
        voxel: Voxel-downsample size for the lifted object cloud (metres).
        erode_px: Mask erosion before lifting, to shed rim/background pixels.
        min_points: Skip an attempt whose lifted cloud is smaller than this.
        antipodal: Run GraspNet's antipodal surface-normal validation.
        approach_preference: Bias grasp selection by approach direction relative to
            gravity: ``"side"`` favours horizontal approaches (e.g. grabbing a can
            around its side under a high fixed camera), ``"top"`` favours approaches
            pointing straight down (e.g. a spoon lying flat), ``"none"`` leaves
            GraspNet's ranking untouched. The "up" reference is derived
            automatically from each snapshot's pose (the map frame's +Z gravity axis,
            expressed in the cloud's optical frame), so the caller need not supply it.
        approach_weight: How strongly the preference outranks GraspNet's own score
            (server default ~1.0; higher favours the preferred approach harder). Only
            used when ``approach_preference`` is set; ``None`` keeps the server default.

    Returns:
        The winning :class:`GraspCandidate` (with ``grasp_xyz`` and
        ``pregrasp_xyz`` in the map frame), or ``None`` if no attempt produced a
        graspable detection.
    """
    best: GraspCandidate | None = None

    for i in range(attempts):
        tag = f"attempt {i + 1}/{attempts}"
        snap = ctx.snapshot()
        if snap is None or not snap.has_geometry:
            print(f"[grasp] {tag}: no snapshot geometry (is the ZED running?)")
            continue

        detections = ctx.walkieAI.image.detect(snap.img, prompts=prompts, return_mask=True)
        detections = [d for d in detections if d.mask is not None]
        if not detections:
            print(f"[grasp] {tag}: no masked detections for {prompts}")
            continue
        det = detections[0]

        cloud = snap.mask_to_points(det.mask, voxel=voxel, frame="optical", erode_px=erode_px)
        if cloud.shape[0] < min_points:
            print(f"[grasp] {tag}: only {cloud.shape[0]} pts lifted — too far/occluded?")
            continue

        infer_kwargs: dict = {"antipodal": antipodal, "max_grasps": 1}
        if approach_preference != "none":
            # World-up = the map frame's +Z (gravity) axis, expressed in the
            # camera-optical frame the cloud lives in, so the server can bias
            # side/top approaches against gravity.
            infer_kwargs["approach_preference"] = approach_preference
            infer_kwargs["up"] = snap.cam.R.T @ np.array([0.0, 0.0, 1.0])
            if approach_weight is not None:
                infer_kwargs["approach_weight"] = approach_weight
        grasps = ctx.walkieAI.grasp.infer(cloud, **infer_kwargs)
        if not grasps:
            print(f"[grasp] {tag}: GraspNet returned nothing")
            continue

        g = grasps[0]
        if best is not None and g.score <= best.score:
            print(f"[grasp] {tag}: score {g.score:.3f} (keeping best {best.score:.3f})")
            continue

        best = _to_map_frame(snap, g, standoff_m)
        gx, gy, gz = best.grasp_xyz
        print(f"[grasp] {tag}: new best score {best.score:.3f} "
              f"grasp=({gx:+.3f},{gy:+.3f},{gz:+.3f}) width={best.width * 100:.1f}cm")

    if best is None:
        print(f"[grasp] no graspable detection for {prompts} in {attempts} attempt(s)")
    return best
