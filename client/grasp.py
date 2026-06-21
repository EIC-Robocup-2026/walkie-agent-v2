"""Grasp-pose client — POST a point cloud to ``/grasp``, get 6-DOF grasps back.

The server runs GraspNet-1Billion on a supplied ``(N, 3)`` object cloud and returns
ranked grasp poses **in the same frame as the cloud you send**. It does no robot
transforms, so the caller owns the framing:

  - Send the object's points in the **camera-optical** frame (X-right, Y-down,
    Z-forward) so GraspNet stays in-distribution. ``CameraSnapshot.mask_to_points(
    mask, frame="optical")`` produces exactly that.
  - Grasps come back in that optical frame; map them back to the world / arm frame
    yourself (``p_map = snap.cam.R @ p_opt + snap.cam.t``).

Example::

    cloud = snap.mask_to_points(det.mask, frame="optical")   # (N, 3) optical
    grasps = walkie.grasp.infer(cloud, antipodal=True, max_grasps=10)
    best = grasps[0]
    print(best.translation, best.score, best.width)

    # Bias selection by approach direction. `up` is world-up expressed in the
    # cloud's frame (gravity = -up) — derive it from the camera extrinsics.
    side = walkie.grasp.infer(cloud, approach_preference="side", up=up_optical)  # a can
    top = walkie.grasp.infer(cloud, approach_preference="top", up=up_optical)    # a spoon
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .base import WalkieBaseClient, _numpy_to_npy_bytes


@dataclass
class GraspPose:
    """One 6-DOF grasp, in the frame of the cloud that was sent.

    ``rotation`` is a 3x3 matrix whose columns are the GraspNet axes: column 0 is
    the **approach** (travel) direction, column 1 the **closing** (gripper-spread)
    direction. ``width`` is the gripper opening in metres, ``score`` GraspNet's
    quality. ``antipodal_score`` is ``None`` unless the request set ``antipodal=True``.
    """

    translation: tuple[float, float, float]
    rotation: np.ndarray  # (3, 3)
    width: float
    score: float
    antipodal_score: float | None = None

    @property
    def approach(self) -> np.ndarray:
        """Unit approach/travel direction (rotation column 0), in the cloud's frame."""
        return self.rotation[:, 0]

    @property
    def closing(self) -> np.ndarray:
        """Unit gripper-closing direction (rotation column 1), in the cloud's frame."""
        return self.rotation[:, 1]


def _deserialize_grasp(g: dict) -> GraspPose:
    anti = g.get("antipodal_score")
    return GraspPose(
        translation=tuple(float(x) for x in g["translation"]),  # type: ignore[arg-type]
        rotation=np.asarray(g["rotation"], dtype=np.float64).reshape(3, 3),
        width=float(g["width"]),
        score=float(g["score"]),
        antipodal_score=(float(anti) if anti is not None else None),
    )


class GraspClient(WalkieBaseClient):
    """Client for the ``/grasp`` endpoint (GraspNet-1Billion over HTTP)."""

    def infer(
        self,
        cloud: np.ndarray,
        *,
        score_threshold: float = 0.0,
        max_grasps: int = 20,
        antipodal: bool = False,
        voxel_size: float | None = None,
        num_point: int | None = None,
        outlier_removal: bool = True,
        cluster_filter: bool = False,
        approach_preference: str = "none",
        up: np.ndarray | Sequence[float] | None = None,
        approach_weight: float | None = None,
    ) -> list[GraspPose]:
        """Generate grasp poses for an ``(N, 3)`` object cloud.

        Args:
            cloud: ``(N, 3)`` XYZ points in a single frame (optical recommended).
            score_threshold: Drop grasps below this GraspNet quality (0 = no filter).
            max_grasps: Cap on the number of poses returned (best-first).
            antipodal: Run antipodal surface-normal validation — rejects grasps that
                don't lie on the object surface and refines each grasp's width/centre.
            voxel_size: Override the server's voxel-downsample size (metres).
            num_point: Override how many points are fed to GraspNet.
            outlier_removal: Statistical-outlier cleanup before inference.
            cluster_filter: Keep only the largest DBSCAN cluster (for clouds that
                still carry neighbour/background points).
            approach_preference: Softly bias selection by approach direction relative
                to ``up``: ``"side"`` favours horizontal approaches (e.g. grabbing a
                can around its side), ``"top"`` favours approaches pointing down along
                gravity (e.g. a spoon lying flat), ``"none"`` leaves GraspNet's
                ranking untouched. Requires ``up`` when not ``"none"``.
            up: World-up direction expressed **in the cloud's frame** (gravity =
                ``-up``); a 3-vector, need not be unit length. In the camera-optical
                frame this is the camera's up axis from its extrinsics. Required for a
                ``"side"`` / ``"top"`` preference.
            approach_weight: Strength of the preference bonus added to the GraspNet
                score (server default ~1.0; higher favours the preferred approach more
                strongly). Ignored without a preference.

        Returns:
            ``list[GraspPose]`` sorted best-first, in the input cloud's frame.
            Empty when GraspNet finds nothing above the threshold.
        """
        arr = np.asarray(cloud, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"cloud must be (N, 3); got shape {arr.shape}")

        preference = str(approach_preference or "none").lower()
        if preference not in ("none", "side", "top"):
            raise ValueError(
                "approach_preference must be 'none', 'side', or 'top'; "
                f"got {approach_preference!r}"
            )

        spec: dict = {
            "score_threshold": float(score_threshold),
            "max_grasps": int(max_grasps),
            "antipodal": bool(antipodal),
            "outlier_removal": bool(outlier_removal),
            "cluster_filter": bool(cluster_filter),
            "approach_preference": preference,
        }
        if voxel_size is not None:
            spec["voxel_size"] = float(voxel_size)
        if num_point is not None:
            spec["num_point"] = int(num_point)
        if up is not None:
            up_vec = np.asarray(up, dtype=float).reshape(-1)
            if up_vec.shape != (3,):
                raise ValueError(f"up must be a 3-vector; got shape {np.asarray(up).shape}")
            spec["up"] = up_vec.tolist()
        if approach_weight is not None:
            spec["approach_weight"] = float(approach_weight)
        if preference != "none" and "up" not in spec:
            raise ValueError(
                f"approach_preference={preference!r} requires an 'up' vector "
                "(world-up expressed in the cloud's frame)"
            )

        import json

        data = self._post_files(
            "/grasp",
            files={"cloud": ("cloud.npy", _numpy_to_npy_bytes(arr), "application/octet-stream")},
            data={"spec": json.dumps(spec)},
        )
        return [_deserialize_grasp(g) for g in data.get("grasps", [])]
