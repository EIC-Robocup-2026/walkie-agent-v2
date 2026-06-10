"""Camera geometry for walkie_graphs — pure numpy, no robot/SDK imports.

Turns a masked detection in an RGB-D frame into 3D points in the **map frame**,
using two things the walkie-sdk now provides directly:

1. :class:`Intrinsics` — real pinhole intrinsics from ``bot.camera.get_intrinsics()``
   (``fx, fy, cx, cy``). :meth:`Intrinsics.scaled_to` rescales them if the depth
   image is a different resolution than the ``CameraInfo`` they came from.
2. :class:`CameraPose` — the camera **optical** frame's pose in the map frame, built
   in the service from ``bot.transform.lookup("map", "<cam>_optical_frame")``. The
   optical frame's axes (``x right, y down, z forward``) are exactly the axes the
   pinhole back-projection produces, so the rotation maps camera points straight into
   the map — no intermediate body-frame conversion, no manual lift/tilt composition.

:func:`deproject_mask` ties them together: ``P_map = P_optical @ R.T + t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # cv2 is a hard dep of the app; guard only so unit tests can run headless
    import cv2
except Exception:  # pragma: no cover - cv2 always present in this project
    cv2 = None


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics (pixels). Straight from ``camera.get_intrinsics()``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Rescale to a different image resolution (e.g. depth ≠ CameraInfo size).

        The ZED head registers depth to the rectified left image, so this is usually
        a no-op; it only matters if the depth stream is downscaled relative to the
        ``CameraInfo`` the intrinsics came from.
        """
        if not self.width or not self.height or (width == self.width and height == self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(
            self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, int(width), int(height)
        )


# ---------------------------------------------------------------------------
# Camera pose (optical frame -> map)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraPose:
    """Camera **optical** frame pose in the map frame.

    ``R`` (3x3) rotates a point from the camera optical frame into the map frame and
    ``t`` (3,) is the optical centre in the map frame, so a point ``p`` maps as
    ``R @ p + t`` (equivalently ``p @ R.T + t`` for a batch). Build it from the SDK
    transform: ``R = quaternion_to_matrix(*q)``, ``t = (x, y, z)``.
    """

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)


# ---------------------------------------------------------------------------
# Deprojection (depth -> map points)
# ---------------------------------------------------------------------------
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
    """Back-project all masked pixels with valid depth to an ``(N, 3)`` map-frame cloud.

    NaN/zero depth pixels are dropped. If ``mask`` and ``depth`` differ in shape, the
    mask is resized to the depth resolution (nearest-neighbour). Each pixel is
    back-projected into the camera optical frame and mapped into the world by the
    optical-frame pose (``P_map = P_optical @ R.T + t``). Optionally voxel-downsampled
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

    # Pinhole back-projection into the camera optical frame (x right, y down, z fwd).
    Xc = (xs - intr.cx) * d / intr.fx
    Yc = (ys - intr.cy) * d / intr.fy
    Zc = d
    P_optical = np.stack([Xc, Yc, Zc], axis=1)
    # Optical-frame pose maps these straight into the map frame.
    P_world = P_optical @ pose.R.T + pose.t

    if voxel:
        P_world = voxel_downsample(P_world, voxel)
    if max_points and len(P_world) > max_points:
        idx = np.linspace(0, len(P_world) - 1, max_points).astype(np.int64)
        P_world = P_world[idx]
    return P_world.astype(np.float32)
