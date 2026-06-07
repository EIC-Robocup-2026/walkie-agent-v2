"""Camera geometry for walkie_graphs — pure numpy, no robot/SDK imports.

Everything needed to turn a masked detection in an RGB-D frame into 3D points in
the **world (map) frame**:

1. :class:`Intrinsics` — pinhole model. The SDK exposes no ``camera_info`` for the
   ZED 2i, so the default is derived from the horizontal FOV
   (``fx = (W/2) / tan(HFOV/2)``, ``fy = fx`` for square rectified pixels,
   principal point at the image center). Pass real calibrated values to override.
2. :func:`compute_camera_pose` — composes the camera's world pose from the robot's
   planar pose + lift height + head tilt + fixed mount offsets.
3. :func:`pixel_to_world` / :func:`deproject_mask` — back-project depth pixels.

Frame conventions
-----------------
Robot LOCAL frame: ``x = forward, y = left, z = up``. World frame uses the same
axes; ``heading`` is yaw (CCW from world +x, ROS REP-103). The camera OPTICAL
frame is OpenCV style: ``x = right, y = down, z = forward``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # cv2 is a hard dep of the app; guard only so unit tests can run headless
    import cv2
except Exception:  # pragma: no cover - cv2 always present in this project
    cv2 = None


# Fixed camera mount offsets, robot LOCAL frame (metres), x=forward y=left z=up.
# LIFT_TO_HEAD: lift-top -> head tilt pivot. PIVOT_TO_OPTIC: tilt pivot -> optical
# center (this one rotates with the head tilt servo).
DEFAULT_LIFT_TO_HEAD = (0.265, 0.0, 0.422)
DEFAULT_PIVOT_TO_OPTIC = (0.065, 0.0, 0.0)

LIFT_CM_TO_M = 0.01  # walkie.robot.lift.get(norm_pos=False) returns CENTIMETRES


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
def rot_z(yaw: float) -> np.ndarray:
    """Yaw rotation about +z (CCW)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> np.ndarray:
    """Pitch rotation about +y (head tilt; positive = camera looks down)."""
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_hfov(
        cls,
        width: int,
        height: int,
        hfov_deg: float = 110.0,
        *,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
    ) -> "Intrinsics":
        """Build intrinsics, deriving any value left as ``None`` from the FOV.

        ``fx`` defaults to ``(width / 2) / tan(hfov / 2)`` (ZED 2i HFOV ≈ 110°),
        ``fy`` defaults to ``fx`` (square pixels on the rectified stream), and the
        principal point defaults to the image center.
        """
        f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        fx = f if fx is None else float(fx)
        fy = fx if fy is None else float(fy)
        cx = (width / 2.0) if cx is None else float(cx)
        cy = (height / 2.0) if cy is None else float(cy)
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=int(width), height=int(height))


# ---------------------------------------------------------------------------
# Camera world pose
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraPose:
    """Camera pose in the world frame.

    ``R`` maps a point from the robot-local camera frame to the world frame
    (``R = Rz(heading) @ Ry(tilt)``); ``t`` is the optical-center position.
    """

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)


def compute_camera_pose(
    robot_x: float,
    robot_y: float,
    heading: float,
    lift_cm: float,
    tilt_rad: float,
    *,
    lift_to_head: tuple[float, float, float] = DEFAULT_LIFT_TO_HEAD,
    pivot_to_optic: tuple[float, float, float] = DEFAULT_PIVOT_TO_OPTIC,
) -> CameraPose:
    """Compose the camera's world pose.

    ``cam_pos = (robot_x, robot_y, 0) + Rz(heading) @ [ (0,0,lift_m)
    + LIFT_TO_HEAD + Ry(tilt) @ PIVOT_TO_OPTIC ]``, where ``lift_m = lift_cm/100``.
    Orientation (world ← camera-local) is ``Rz(heading) @ Ry(tilt)``.

    Args:
        robot_x, robot_y: Robot planar position in the world/map frame (metres).
        heading: Robot yaw (radians).
        lift_cm: Lift height in **centimetres** (the SDK's native unit).
        tilt_rad: Head tilt in radians (positive = looking down).
    """
    lift_m = lift_cm * LIFT_CM_TO_M
    Rz = rot_z(heading)
    Ry = rot_y(tilt_rad)

    cam_local = (
        np.array([0.0, 0.0, lift_m])
        + np.asarray(lift_to_head, dtype=float)
        + Ry @ np.asarray(pivot_to_optic, dtype=float)
    )
    cam_pos_world = np.array([robot_x, robot_y, 0.0]) + Rz @ cam_local
    return CameraPose(R=Rz @ Ry, t=cam_pos_world)


# ---------------------------------------------------------------------------
# Deprojection (depth -> world points)
# ---------------------------------------------------------------------------
def _optical_to_local(P_optical: np.ndarray) -> np.ndarray:
    """OpenCV optical axes (x right, y down, z forward) → robot-local (fwd, left, up)."""
    # forward = Zc, left = -Xc, up = -Yc
    return np.stack([P_optical[:, 2], -P_optical[:, 0], -P_optical[:, 1]], axis=1)


def pixel_to_world(
    u: float, v: float, depth: float, intr: Intrinsics, pose: CameraPose
) -> tuple[float, float, float]:
    """Back-project a single pixel + depth to a world-frame point."""
    Xc = (u - intr.cx) * depth / intr.fx
    Yc = (v - intr.cy) * depth / intr.fy
    Zc = depth
    p_local = np.array([Zc, -Xc, -Yc])  # optical -> robot-local
    p_world = pose.t + pose.R @ p_local
    return float(p_world[0]), float(p_world[1]), float(p_world[2])


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Grid-quantize to ``voxel``-sized cells, returning one mean point per cell."""
    if voxel is None or voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_cells = int(inverse.max()) + 1
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    counts = np.bincount(inverse, minlength=n_cells).reshape(-1, 1)
    return (sums / counts).astype(np.float32)


def deproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    pose: CameraPose,
    *,
    voxel: float | None = None,
    max_points: int | None = None,
) -> np.ndarray:
    """Back-project all masked pixels with valid depth to an ``(N, 3)`` world cloud.

    NaN/zero depth pixels are dropped. If ``mask`` and ``depth`` differ in shape
    (the ZED color and depth streams can), the mask is resized to the depth
    resolution with nearest-neighbour interpolation. Optionally voxel-downsampled
    and capped at ``max_points`` (deterministic uniform stride).
    """
    if mask.shape[:2] != depth.shape[:2]:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("cv2 required to resize mask to depth resolution")
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    d = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xs, ys, d = xs[valid], ys[valid], d[valid]

    Xc = (xs - intr.cx) * d / intr.fx
    Yc = (ys - intr.cy) * d / intr.fy
    Zc = d
    P_optical = np.stack([Xc, Yc, Zc], axis=1)
    P_local = _optical_to_local(P_optical)
    P_world = (pose.R @ P_local.T).T + pose.t

    if voxel:
        P_world = voxel_downsample(P_world, voxel)
    if max_points and len(P_world) > max_points:
        idx = np.linspace(0, len(P_world) - 1, max_points).astype(np.int64)
        P_world = P_world[idx]
    return P_world.astype(np.float32)
