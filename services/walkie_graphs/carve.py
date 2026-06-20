"""Free-space carving — remove map geometry a later view sees straight through.

A point cloud only ever *adds* evidence: every sighting appends, nothing says
"that spot is now empty". Two failure modes accumulate as a result —

- **moved-object ghosts**: an object mapped at A and then physically moved leaves
  its old node sitting at A forever (the incremental matcher can't know A is now
  bare);
- **edge-shadow trails**: a silhouette's flying pixels hang behind the object;
  they're occluded from the viewpoint that created them but stick out into space
  that a *lateral* viewpoint sees through.

Free-space carving is the missing negative evidence. For a trusted capture (its
pose is either registration-off or an accepted ICP solve — never a rejected/cold
guess, which would carve a mis-posed hole in a good map) we project every nearby
stored point back into the depth image: if the sensor measured something
*farther* than the point — i.e. it saw straight through where the point claims to
be solid — that point is in free space and is removed.

Pure numpy, no SDK imports. The pose handling mirrors :mod:`geometry`:
``P_world = P_optical @ R.T + t`` forward, ``P_optical = (P_world - t) @ R``
inverse (``R`` orthogonal). After an accepted capture registration the effective
pose folds the correction in (:func:`corrected_pose`).
"""

from __future__ import annotations

import numpy as np

from interfaces.perception.geometry import CameraPose, Intrinsics


def corrected_pose(cam: CameraPose, correction: np.ndarray | None) -> CameraPose:
    """The capture's effective optical-frame pose after a registration correction.

    ``register_capture`` applies its 4×4 ``T`` to the *lifted world points*
    (``p' = p @ T[:3,:3].T + T[:3,3]``). Composing that with the raw optical→world
    pose (``p = p_opt @ R.T + t``) gives an effective pose ``R' = C·R``,
    ``t' = C·t + c`` so the projection math below can use one pose uniformly.
    ``correction=None`` or identity returns ``cam`` unchanged.
    """
    if correction is None:
        return cam
    C = np.asarray(correction, dtype=np.float64)
    Cr, Ct = C[:3, :3], C[:3, 3]
    if np.allclose(Cr, np.eye(3)) and np.allclose(Ct, 0.0):
        return cam
    return CameraPose(R=Cr @ cam.R, t=Cr @ cam.t + Ct)


def frustum_aabb(
    intr: Intrinsics,
    pose: CameraPose,
    shape_hw: tuple[int, int],
    *,
    z_min: float = 0.05,
    z_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB of the view pyramid between ``z_min`` and ``z_max``.

    The eight corners (four image corners × two depths) back-projected to the map
    bound everything the capture could possibly carve, so the background/object
    queries only ever consider points actually in view.
    """
    h, w = shape_hw
    us = np.array([0.0, w, 0.0, w], dtype=np.float64)
    vs = np.array([0.0, 0.0, h, h], dtype=np.float64)
    corners = []
    for z in (float(z_min), float(z_max)):
        Xc = (us - intr.cx) * z / intr.fx
        Yc = (vs - intr.cy) * z / intr.fy
        Zc = np.full_like(us, z)
        P_opt = np.stack([Xc, Yc, Zc], axis=1)
        corners.append(P_opt @ pose.R.T + pose.t)
    pts = np.vstack(corners)
    return pts.min(axis=0), pts.max(axis=0)


def free_space_mask(
    points: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    pose: CameraPose,
    *,
    margin_base: float = 0.05,
    margin_rel: float = 0.02,
    z_min: float = 0.05,
    max_z: float = 4.0,
) -> np.ndarray:
    """Boolean ``(N,)`` mask: True for points the capture sees *straight through*.

    A stored point is carved when, projected into this depth image, the sensor
    measured a surface **farther** than the point (``d > z + margin``): the camera
    saw past where the point claims to be solid, so that space is empty. The margin
    ``margin_base + margin_rel·z`` (grows with depth, matching the ZED's error) keeps
    a point that simply *is* the measured surface. No evidence (never carved) when:
    the point is behind the camera or nearer than ``z_min``; out of frame; the pixel
    has no/invalid depth; the measured surface is *nearer* (``d < z`` → the point is
    occluded, not seen through); or the point is beyond ``max_z`` (untrusted range).
    """
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros(len(pts), dtype=bool)
    if len(pts) == 0:
        return out
    h, w = depth.shape[:2]
    p_opt = (pts - pose.t) @ pose.R  # inverse of P_opt @ R.T + t
    z = p_opt[:, 2]
    testable = (z > max(z_min, 1e-6)) & (z <= max_z)
    if not testable.any():
        return out
    u = np.full(len(pts), -1.0)
    v = np.full(len(pts), -1.0)
    zt = z[testable]
    u[testable] = p_opt[testable, 0] * intr.fx / zt + intr.cx
    v[testable] = p_opt[testable, 1] * intr.fy / zt + intr.cy
    ui = np.floor(u).astype(np.int64)
    vi = np.floor(v).astype(np.int64)
    in_frame = testable & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not in_frame.any():
        return out
    idx = np.where(in_frame)[0]
    d = depth[vi[idx], ui[idx]].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    zi = z[idx]
    margin = margin_base + margin_rel * zi
    carve = valid & (d > zi + margin)  # saw past the point → free space
    out[idx[carve]] = True
    return out
