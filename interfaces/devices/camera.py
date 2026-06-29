"""Camera input for capturing images and video frames.

Hosts both the live :class:`Camera` device wrapper and :class:`CameraSnapshot` —
one instant's sensors + pose with mask/bbox → 3D lifting — so a single module
owns "give me a frame now" and "capture this moment, lift pixels from it later".
"""

from __future__ import annotations

import os
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from interfaces.perception.geometry import (
    CameraPose,
    Intrinsics,
    deproject_mask,
    depth_discontinuity_mask,
)

if TYPE_CHECKING:
    from walkie_sdk.robot import WalkieRobot


class Camera:
    """Camera interface for capturing images.

    Supports two mutually-exclusive sources:
      - **Robot mode**: frames come from a ``WalkieRobot`` instance.
      - **Local mode**: frames come from a local webcam via ``cv2.VideoCapture``.

    Provides methods for capturing single frames and returning images
    in various formats (numpy array, PIL Image, bytes).
    """

    def __init__(
        self,
        robot: "WalkieRobot | None" = None,
        device: int | None = None,
    ) -> None:
        """Initialize camera.

        Exactly one of *robot* or *device* must be provided.

        Args:
            robot: WalkieRobot instance for robot camera access.
            device: Local webcam device index (e.g. 0 for the default laptop camera).

        Raises:
            ValueError: If both or neither source is provided.
        """
        if robot is not None and device is not None:
            raise ValueError("Provide either 'robot' or 'device', not both.")
        if robot is None and device is None:
            raise ValueError("Provide either 'robot' or 'device'.")

        self._bot = robot
        self._device = device
        self._cap: cv2.VideoCapture | None = None  # only used in local mode

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera source.

        - Local mode: opens the ``cv2.VideoCapture`` for the configured device.
        - Robot mode: no-op (the robot manages its own camera lifecycle).

        Raises:
            RuntimeError: If the local camera cannot be opened.
        """
        if self._device is not None:
            self._cap = cv2.VideoCapture(self._device)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Failed to open local camera (device={self._device})."
                )

    def close(self) -> None:
        """Release the camera source.

        - Local mode: releases the ``cv2.VideoCapture``.
        - Robot mode: no-op.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Capture a single frame from the camera.

        Returns:
            Frame as numpy array in BGR format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        if self._bot is not None:
            import time as _time
            deadline = _time.monotonic() + 5.0
            frame = None
            while frame is None and _time.monotonic() < deadline:
                frame = self._bot.camera.get_frame()
                if frame is None:
                    _time.sleep(0.05)
            if frame is None:
                raise RuntimeError("Failed to get frame from robot camera.")
        else:
            if self._cap is None or not self._cap.isOpened():
                raise RuntimeError("Local camera is not open. Call open() first.")
            ret, frame = self._cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from local camera.")
        return frame

    def capture_rgb(self) -> np.ndarray:
        """Capture a single frame in RGB format.

        Returns:
            Frame as numpy array in RGB format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame = self.capture()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def capture_pil(self) -> Image.Image:
        """Capture a single frame as PIL Image.

        Returns:
            Frame as PIL Image in RGB format.

        Raises:
            RuntimeError: If camera is not open or frame capture fails.
        """
        frame_rgb = self.capture_rgb()
        return Image.fromarray(frame_rgb)

    # ------------------------------------------------------------------
    # Context manager / destructor
    # ------------------------------------------------------------------

    def __enter__(self) -> "Camera":
        """Context manager entry - opens the camera."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the camera."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures camera connection is released."""
        self.close()


# ----------------------------------------------------------------------------
# CameraSnapshot — one instant's sensors + pose, with mask/bbox → 3D lifting
# ----------------------------------------------------------------------------
# Generalizes the perception loop's frame snapshot so any caller (the graphs
# service, HRI subtasks, future tasks) can capture an RGB-D frame *together with*
# the camera pose and intrinsics of that instant, and later lift pixels from it
# into map-frame points. The detection / LLM round-trips that follow a capture
# are slow and the robot keeps moving through them, so deprojecting against the
# camera's *current* depth (what ``walkie.tools.bboxes_to_positions`` does)
# describes a different instant than the image being reasoned about — the
# snapshot pins lift pose, intrinsics, and robot heading to the moment the frame
# was actually taken.
#
# Lifting reuses the exact ``interfaces.perception.geometry.deproject_mask``
# pipeline the scene graph runs (depth-edge filter, mask erode, SOR, voxel),
# parameterized by the same ``WALKIE_EXPLORE_*`` env vars by default. Those
# geometry helpers live in ``interfaces.perception`` (a pure-leaf package, not the
# ``walkie_graphs`` service), so they're imported at module top level — there is no
# cycle to dodge: walkie_graphs imports this module, never the other way around.


def _noop(_msg: str) -> None:
    pass


def _envf(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _envi(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rotation matrix for intrinsic X→Y→Z euler angles (radians): ``Rx @ Ry @ Rz``."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return Rx @ Ry @ Rz


def tilt_offset_matrix(
    *,
    rx: float | None = None,
    ry: float | None = None,
    rz: float | None = None,
) -> np.ndarray | None:
    """Camera-local rotation correction for the tilt servo's backlash, or ``None``.

    The head's tilt servo has mechanical backlash, so the optical frame the TF tree
    reports is rotated slightly from where the camera actually points. These three
    euler offsets (radians, in the camera *optical* frame) are baked into the camera
    pose right before lifting so the correction applies to every deprojection path.
    ``None`` args default from ``WALKIE_CAMERA_TILT_OFFSET_{X,Y,Z}_RAD``. Returns
    ``None`` when all three are zero (so callers can skip the matmul entirely).
    """
    rx = rx if rx is not None else _envf("WALKIE_CAMERA_TILT_OFFSET_X_RAD", "0.0")
    ry = ry if ry is not None else _envf("WALKIE_CAMERA_TILT_OFFSET_Y_RAD", "0.0")
    rz = rz if rz is not None else _envf("WALKIE_CAMERA_TILT_OFFSET_Z_RAD", "0.0")
    if rx == 0.0 and ry == 0.0 and rz == 0.0:
        return None
    return _euler_xyz_to_matrix(rx, ry, rz)


# Intrinsics are static per camera, so cache the scaled result per resolution.
# Keyed weakly by the walkie object (not module-global) so two interfaces — or
# successive test fakes — never share entries.
_INTR_CACHE: "weakref.WeakKeyDictionary[object, dict[tuple[int, int], Intrinsics]]" = (
    weakref.WeakKeyDictionary()
)


def camera_pose(
    walkie,
    *,
    map_frame: str | None = None,
    cam_frame: str | None = None,
    timeout: float | None = None,
    tilt_offset: np.ndarray | None = None,
    apply_tilt_offset: bool = True,
    log=_noop,
) -> CameraPose | None:
    """Camera optical-frame pose in the map frame, from the SDK transform tree.

    ``transform.lookup(MAP_FRAME, CAMERA_FRAME)`` with the camera *optical* frame
    returns a pose whose rotation maps camera-optical points straight into the map
    (lift, head tilt, and mount offsets already baked in), so deprojection needs no
    further composition. Returns ``None`` when the lookup fails.

    The tilt servo has backlash, so the reported optical frame is rotated slightly
    from where the camera really points. When *apply_tilt_offset* is true the
    camera-local correction from :func:`tilt_offset_matrix` (or an explicit
    *tilt_offset* matrix) is composed in as ``R @ R_offset`` — rotating points in
    the optical frame before the TF rotation maps them to the map.
    """
    map_frame = map_frame or os.getenv("WALKIE_EXPLORE_TF_MAP_FRAME", "map")
    cam_frame = cam_frame or os.getenv(
        "WALKIE_EXPLORE_TF_CAMERA_FRAME", "zed_head_left_camera_frame_optical"
    )
    timeout = timeout if timeout is not None else _envf("WALKIE_EXPLORE_TF_TIMEOUT_SEC", "1.0")
    try:
        tf = walkie.robot.transform.lookup(map_frame, cam_frame, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        log(f"transform lookup error ({map_frame} -> {cam_frame}): {e}")
        return None
    if tf is None:
        log(f"transform lookup returned None ({map_frame} -> {cam_frame})")
        return None

    from walkie_sdk.utils.converters import quaternion_to_matrix

    q, p = tf["quaternion"], tf["position"]
    R = quaternion_to_matrix(float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
    t = np.array([float(p["x"]), float(p["y"]), float(p["z"])])
    if apply_tilt_offset:
        R_off = tilt_offset if tilt_offset is not None else tilt_offset_matrix()
        if R_off is not None:
            R = R @ R_off
    return CameraPose(R=R, t=t)


def intrinsics_for(walkie, width: int, height: int, *, log=_noop) -> Intrinsics | None:
    """Real pinhole intrinsics from the SDK, scaled to the given resolution.

    ``camera.get_intrinsics()`` reads the ZED's ``CameraInfo`` (cached by the SDK —
    intrinsics are static) and applies to the registered depth too. Returns ``None``
    when no camera info is available yet.
    """
    try:
        per_walkie = _INTR_CACHE.setdefault(walkie, {})
    except TypeError:  # non-weakref-able walkie (rare): skip caching
        per_walkie = {}
    cached = per_walkie.get((width, height))
    if cached is not None:
        return cached
    try:
        raw = walkie.robot.camera.get_intrinsics()
    except Exception as e:  # noqa: BLE001
        log(f"intrinsics unavailable: {e}")
        return None
    if not raw:
        log("intrinsics unavailable (camera_info not published yet)")
        return None

    intr = Intrinsics(
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        width=int(raw.get("width") or width),
        height=int(raw.get("height") or height),
    ).scaled_to(width, height)
    per_walkie[(width, height)] = intr
    return intr


@dataclass
class CameraSnapshot:
    """Every sensor + pose read for one instant, taken together at capture.

    ``cam``/``intr`` may be ``None`` (pose or camera-info unavailable) — lifting
    then degrades to empty results; ``depth`` and ``img`` are always present when
    built via :meth:`capture` (a missing one returns ``None`` instead of a
    snapshot).
    """

    ts: float
    img: object  # PIL RGB frame
    depth: np.ndarray | None  # aligned depth, H×W metres (NaN invalid)
    cam: CameraPose | None  # camera optical-frame pose in the map frame
    intr: Intrinsics | None  # pinhole intrinsics at the depth resolution
    robot_pose: dict | None  # status.get_position() — heading at capture time
    _edge_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    _edge_mask_done: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def capture(cls, walkie, *, log=_noop) -> "CameraSnapshot | None":
        """Read depth, RGB, camera pose, intrinsics, and robot pose back-to-back.

        Everything later lifting consumes is read here, before any slow round-trip,
        so it all describes the same instant. Returns ``None`` when depth or the
        image is unavailable (both mandatory); ``cam``/``intr`` may be ``None`` and
        degrade lifting to "no geometry".
        """
        # Per-stage timing (WALKIE_SNAPSHOT_TIMING=1): get_depth/get_frame are
        # cached reads, but camera_pose() does a synchronous ROS TF service
        # round-trip every call — this prints which stage actually dominates a
        # snapshot, e.g. when a follow loop's sample rate is capped.
        timing = os.getenv("WALKIE_SNAPSHOT_TIMING", "0").lower() in ("1", "true", "yes")
        t0 = time.monotonic()
        try:
            depth = walkie.robot.camera.get_depth()
        except Exception as e:  # noqa: BLE001
            log(f"depth unavailable: {e}")
            return None
        if depth is None:
            log("depth unavailable — no snapshot")
            return None
        t_depth = time.monotonic()
        try:
            img = walkie.camera.capture_pil()  # PIL RGB
        except Exception as e:  # noqa: BLE001
            log(f"capture failed: {e}")
            return None
        t_rgb = time.monotonic()
        ts = time.time()
        cam = camera_pose(walkie, log=log)  # synchronous TF service round-trip
        t_tf = time.monotonic()
        intr = (
            intrinsics_for(walkie, depth.shape[1], depth.shape[0], log=log)
            if cam is not None
            else None
        )
        try:
            robot_pose = walkie.status.get_position()
        except Exception as e:  # noqa: BLE001
            log(f"robot pose unavailable: {e}")
            robot_pose = None
        if timing:
            print(f"[CameraSnapshot] depth={1e3 * (t_depth - t0):.0f}ms "
                  f"rgb={1e3 * (t_rgb - t_depth):.0f}ms tf={1e3 * (t_tf - t_rgb):.0f}ms "
                  f"intr+odom={1e3 * (time.monotonic() - t_tf):.0f}ms")
        return cls(ts=ts, img=img, depth=depth, cam=cam, intr=intr, robot_pose=robot_pose)

    @property
    def has_geometry(self) -> bool:
        """True when this frame can be lifted to 3D (pose + intrinsics + depth)."""
        return self.cam is not None and self.intr is not None and self.depth is not None

    # ------------------------------------------------------------------
    # Lifting
    # ------------------------------------------------------------------
    def _edges(self) -> np.ndarray | None:
        """Depth-discontinuity map for this frame, computed once and cached."""
        if not self._edge_mask_done:
            self._edge_mask = depth_discontinuity_mask(
                self.depth,
                _envf("WALKIE_EXPLORE_DEPTH_EDGE_THRESH_M", "0.05"),
                rel_thresh=_envf("WALKIE_EXPLORE_DEPTH_EDGE_REL", "0.0"),
            )
            self._edge_mask_done = True
        return self._edge_mask

    def mask_to_points(
        self,
        mask: np.ndarray,
        *,
        frame: str = "map",
        voxel: float | None = None,
        max_points: int | None = None,
        erode_px: int | None = None,
        max_depth: float | None = None,
        sor_k: int | None = None,
        sor_std_ratio: float | None = None,
        use_edge_filter: bool = True,
    ) -> np.ndarray:
        """Lift a pixel mask to an ``(N, 3)`` cloud — the walkie_graphs way.

        Runs :func:`deproject_mask` against the *snapshot's* depth/pose/intrinsics
        with the same flying-pixel cleanup the scene graph uses. ``None`` params
        default from the corresponding ``WALKIE_EXPLORE_*`` env vars. Returns an
        empty ``(0, 3)`` array when the snapshot has no geometry.

        ``frame`` selects the output frame:

        - ``"map"`` (default) — points in the map/world frame (``P = P_opt @ R.T + t``),
          what the scene graph stores.
        - ``"optical"`` — points in the camera **optical** frame (X-right, Y-down,
          Z-forward), lifted with an identity pose. This is what GraspNet expects;
          map them back with ``p_map = self.cam.R @ p_opt + self.cam.t``. The cleanup
          (voxel/SOR/edge filter) is rigid-invariant, so the cloud is identical to the
          map-frame one up to that transform.
        """
        if not self.has_geometry:
            return np.zeros((0, 3), dtype=np.float32)

        if frame == "map":
            pose = self.cam
        elif frame == "optical":
            pose = CameraPose(R=np.eye(3, dtype=float), t=np.zeros(3, dtype=float))
        else:
            raise ValueError(f"frame must be 'map' or 'optical', got {frame!r}")

        return deproject_mask(
            mask,
            self.depth,
            self.intr,
            pose,
            voxel=voxel if voxel is not None else _envf("WALKIE_EXPLORE_VOXEL_M", "0.02"),
            max_points=max_points
            if max_points is not None
            else _envi("WALKIE_EXPLORE_MAX_POINTS_PER_OBJ", "2000"),
            erode_px=erode_px if erode_px is not None else _envi("WALKIE_EXPLORE_MASK_ERODE_PX", "2"),
            edge_mask=self._edges() if use_edge_filter else None,
            max_depth=max_depth
            if max_depth is not None
            else _envf("WALKIE_EXPLORE_MAX_DEPTH_M", "0"),
            sor_k=sor_k if sor_k is not None else _envi("WALKIE_EXPLORE_SOR_K", "0"),
            sor_std_ratio=sor_std_ratio
            if sor_std_ratio is not None
            else _envf("WALKIE_EXPLORE_SOR_STD_RATIO", "2.0"),
        )

    def bbox_to_points(self, bbox_xyxy, *, shrink: float = 1.0, **kw) -> np.ndarray:
        """Lift a bbox region to a cloud via a rectangular mask.

        A bbox has no segmentation mask, so its rim is mostly background —
        ``shrink`` scales the rectangle toward its center (0.6 keeps the central
        60% per axis) before deprojection. Extra kwargs go to :meth:`mask_to_points`.
        """
        if not self.has_geometry:
            return np.zeros((0, 3), dtype=np.float32)
        h, w = self.depth.shape[:2]
        # Bbox pixel coords are in the RGB image; scale to the depth resolution
        # (deproject_mask would resize a full-frame mask the same way).
        try:
            iw, ih = self.img.size
        except Exception:  # noqa: BLE001 — synthetic snapshots may lack an image
            iw, ih = w, h
        sx, sy = w / iw, h / ih
        x1, y1, x2, y2 = bbox_xyxy
        cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
        hw, hh = (x2 - x1) / 2 * sx * shrink, (y2 - y1) / 2 * sy * shrink
        ix1, iy1 = max(0, int(cx - hw)), max(0, int(cy - hh))
        ix2, iy2 = min(w, int(np.ceil(cx + hw))), min(h, int(np.ceil(cy + hh)))
        if ix2 <= ix1 or iy2 <= iy1:
            return np.zeros((0, 3), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[iy1:iy2, ix1:ix2] = 1
        # No erode: the shrink already trimmed the rim, and eroding a thin
        # shrunken box can wipe it out entirely.
        kw.setdefault("erode_px", 0)
        # No voxel: voxelizing re-weights points by world surface area, so a far
        # background (large footprint per pixel) would outvote the near object in
        # the median even when the object dominates the bbox PIXELS. Keep
        # per-pixel weighting; max_points still caps the cloud.
        kw.setdefault("voxel", 0.0)
        return self.mask_to_points(mask, **kw)

    def bbox_world_point(
        self, bbox_xyxy, *, shrink: float = 0.6, **kw
    ) -> tuple[float, float, float] | None:
        """Map-frame (x, y, z) of a bbox's content: the MEDIAN of its lifted cloud.

        Median, not mean: a rectangle inevitably includes some background depth
        (unlike the scene graph's true detector masks), and the median stays on
        the dominant surface instead of averaging object and wall.
        """
        pts = self.bbox_to_points(bbox_xyxy, shrink=shrink, **kw)
        if len(pts) == 0:
            return None
        m = np.median(pts, axis=0)
        return float(m[0]), float(m[1]), float(m[2])

    def bbox_world_xy(self, bbox_xyxy, *, shrink: float = 0.6, **kw) -> tuple[float, float] | None:
        """Map-frame (x, y) of a bbox's content — see :meth:`bbox_world_point`."""
        p = self.bbox_world_point(bbox_xyxy, shrink=shrink, **kw)
        return None if p is None else (p[0], p[1])
