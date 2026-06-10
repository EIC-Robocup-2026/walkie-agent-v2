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
    """Grid-quantize to ``voxel``-sized cells, returning one mean point per cell.

    Each cell's integer coordinates are hashed into a single 1-D key, so the grouping
    uses a fast 1-D ``np.unique`` and C-level ``np.bincount`` summation — far cheaper on
    dense clouds than ``np.unique(..., axis=0)`` (a 2-D lexsort) + ``np.add.at`` (an
    unbuffered scatter), which dominated the per-detection deprojection cost.
    """
    pts = np.asarray(points)
    if voxel is None or voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(axis=0)  # shift to non-negative so the hash stays small
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inverse = np.unique(flat, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse)
    p = pts.astype(np.float64)
    sums = np.stack(
        [
            np.bincount(inverse, weights=p[:, 0]),
            np.bincount(inverse, weights=p[:, 1]),
            np.bincount(inverse, weights=p[:, 2]),
        ],
        axis=1,
    )
    return (sums / counts[:, None]).astype(np.float32)


def depth_discontinuity_mask(depth: np.ndarray, thresh: float) -> np.ndarray | None:
    """Boolean HxW map: True where a pixel borders a depth jump larger than ``thresh`` (m).

    These are the "flying pixel" / "mixed pixel" edges: at an object's silhouette a
    depth pixel straddles the foreground and the background, so the sensor reports an
    averaged depth that back-projects to a point hanging in space behind the object (a
    shadow). Both pixels on either side of each jump are flagged so the whole bleed
    band is excluded. Invalid (NaN/≤0) depth is ignored (``NaN > thresh`` is ``False``),
    and those pixels are dropped anyway. Returns ``None`` when ``thresh <= 0``.
    """
    if thresh is None or thresh <= 0:
        return None
    d = depth.astype(np.float32)
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    edge = np.zeros(d.shape, dtype=bool)
    # Vertical then horizontal neighbour jumps; mark both sides of each.
    vj = np.abs(np.diff(d, axis=0)) > thresh
    edge[:-1, :] |= vj
    edge[1:, :] |= vj
    hj = np.abs(np.diff(d, axis=1)) > thresh
    edge[:, :-1] |= hj
    edge[:, 1:] |= hj
    return edge


def deproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    pose: CameraPose,
    *,
    voxel: float | None = None,
    max_points: int | None = None,
    erode_px: int = 0,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Back-project all masked pixels with valid depth to an ``(N, 3)`` map-frame cloud.

    NaN/zero depth pixels are dropped. If ``mask`` and ``depth`` differ in shape, the
    mask is resized to the depth resolution (nearest-neighbour). Each pixel is
    back-projected into the camera optical frame and mapped into the world by the
    optical-frame pose (``P_map = P_optical @ R.T + t``). Optionally voxel-downsampled
    and capped at ``max_points`` (deterministic uniform stride).

    Two edge-cleanup options remove depth "flying pixels" (the shadow trailing off an
    object's silhouette):

    * ``erode_px`` shrinks the mask inward by that many pixels, dropping the unreliable
      rim where foreground/background mix.
    * ``edge_mask`` (a depth-resolution boolean map from :func:`depth_discontinuity_mask`,
      computed once per frame) drops any masked pixel sitting on a depth jump.
    """
    if mask.shape[:2] != depth.shape[:2]:
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("cv2 required to resize mask to depth resolution")
        mask = cv2.resize(
            mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Crop to the mask's bounding box: erode / nonzero / back-projection over a mostly
    # empty multi-megapixel frame is the per-detection hotspot, and the object only
    # occupies a small window. ``np.any`` row/col reductions find the box cheaply; all
    # subsequent per-pixel work runs on the (small) crop, with a pad so erosion still
    # sees real background. Pixel coords are offset back to full-frame for the pinhole.
    rows = np.any(mask, axis=1)
    if not rows.any():
        return np.zeros((0, 3), dtype=np.float32)
    cols = np.any(mask, axis=0)
    yy = np.where(rows)[0]
    xx = np.where(cols)[0]
    pad = int(erode_px) + 1 if (erode_px and erode_px > 0) else 0
    h, w = depth.shape[:2]
    y0, y1 = max(0, int(yy[0]) - pad), min(h, int(yy[-1]) + 1 + pad)
    x0, x1 = max(0, int(xx[0]) - pad), min(w, int(xx[-1]) + 1 + pad)

    sub_mask = mask[y0:y1, x0:x1]
    sub_depth = depth[y0:y1, x0:x1]
    sub_edge = edge_mask[y0:y1, x0:x1] if edge_mask is not None else None

    if erode_px and erode_px > 0 and cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        sub_mask = cv2.erode(sub_mask.astype(np.uint8), kernel, iterations=int(erode_px))

    ys, xs = np.nonzero(sub_mask)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    d = sub_depth[ys, xs].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    if sub_edge is not None:
        valid &= ~sub_edge[ys, xs]
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xs, ys, d = xs[valid], ys[valid], d[valid]

    # Pinhole back-projection into the camera optical frame (x right, y down, z fwd),
    # offsetting the cropped pixel coords back to full-frame (cx/cy are full-frame).
    Xc = ((xs + x0) - intr.cx) * d / intr.fx
    Yc = ((ys + y0) - intr.cy) * d / intr.fy
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
