"""Perception primitives — pure 3D geometry + point-cloud clustering.

Leaf modules (numpy / cv2 / open3d / scipy only, no SDK or service imports) that
turn camera pixels into map-frame geometry. Kept here, next to the camera device
wrapper that consumes them, rather than inside ``services.walkie_graphs`` — they
are reusable across tasks (manipulation, HRI, navigation) via
:class:`interfaces.devices.camera.CameraSnapshot`, and living outside the service
package keeps the camera ↔ walkie_graphs import graph acyclic.

- :mod:`geometry` — depth deprojection, ``CameraPose``/``Intrinsics``, voxel downsample.
- :mod:`dbscan` — DBSCAN clustering + statistical outlier removal.
"""

from __future__ import annotations

from .geometry import (
    CameraPose,
    Intrinsics,
    deproject_mask,
    depth_discontinuity_mask,
    voxel_downsample,
)

__all__ = [
    "CameraPose",
    "Intrinsics",
    "deproject_mask",
    "depth_discontinuity_mask",
    "voxel_downsample",
]
