"""Rerun-backed visualization session + a no-op stub, behind one small interface.

This is the *generic* drawing layer. It owns the process-global Rerun recording
(``rr.init`` + ``rr.spawn``/``rr.serve_grpc``) and exposes a menu of primitives —
``axes`` (an XYZ triad), ``points``, ``box``, ``arrow``, ``lines``, ``clear``, plus
the reusable ``robot``/``camera`` markers. It knows nothing about scene graphs or
tasks; callers compose these primitives under their own entity-path namespaces
(``world/...`` for the scene graph, ``grasp/...`` for a manipulation task, ...).

``rerun`` is imported lazily inside :class:`RerunSession` so importing this module
(or anything that builds a :class:`~tasks.base.TaskContext`) never pulls rerun on a
GPU-/display-less box. When viz is disabled, callers get a :class:`NoOpViz` whose
methods all do nothing, so no call site ever needs a null check.

Env knobs (canonical ``WALKIE_VIZ*`` names, with the old ``WALKIE_EXPLORE_*`` names
honored as fallbacks when the new one is unset):

- ``WALKIE_VIZ_RECORDING``      recording name shown in the viewer (default ``walkie``)
- ``WALKIE_VIZ_SERVE``          0 = native window on the robot; 1 = serve to the LAN
- ``WALKIE_VIZ_{WEB,GRPC}_PORT``  browser + gRPC ports when serving
- ``WALKIE_VIZ_HOST``           IP remote browsers use (blank = auto-detect LAN IP)
- ``WALKIE_VIZ_CORS``           CORS origins for the gRPC server
- ``WALKIE_VIZ_{ROBOT,CAMERA}`` toggle the robot / camera markers
"""

from __future__ import annotations

import math
import os
import socket
import urllib.parse
from typing import Protocol, runtime_checkable

import numpy as np

# Axis triad colors: column 0 -> X (red), column 1 -> Y (green), column 2 -> Z (blue).
_AXIS_COLORS = [(230, 40, 40), (40, 200, 40), (40, 90, 230)]


def _env(new: str, old: str | None, default: str) -> str:
    """Read ``new``, else legacy ``old``, else ``default`` (first set wins)."""
    val = os.getenv(new)
    if val is not None:
        return val
    if old is not None:
        val = os.getenv(old)
        if val is not None:
            return val
    return default


def _flag(new: str, old: str | None, default: str) -> bool:
    return _env(new, old, default).strip().lower() in ("1", "true", "yes")


def _lan_ip() -> str:
    """Best-effort primary LAN IP, so a remote viewer URL is actually reachable."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # selects the egress interface; sends nothing
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


@runtime_checkable
class VizSession(Protocol):
    """Structural interface satisfied by both :class:`RerunSession` and :class:`NoOpViz`."""

    def axes(self, entity, position, rotation=None, length: float = 0.1, labels: bool = False) -> None: ...
    def points(self, entity, positions, *, colors=None, radii=None, labels=None) -> None: ...
    def box(self, entity, center, half_sizes, *, color=None, label=None) -> None: ...
    def arrow(self, entity, origin, vector, *, color=None, label=None) -> None: ...
    def lines(self, entity, strips, *, labels=None, colors=None) -> None: ...
    def clear(self, entity, *, recursive: bool = True) -> None: ...
    def robot(self, robot_pose) -> None: ...
    def camera(self, cam_pose) -> None: ...


class RerunSession:
    """Streams primitives to a Rerun viewer over one process-global recording.

    Two modes (``WALKIE_VIZ_SERVE``):

    - ``0``: ``rr.spawn()`` — a native viewer window on the robot. Needs a display
      and is **only visible on the robot itself**.
    - ``1`` (default): serve a browser viewer over HTTP + a gRPC data stream, both
      bound to ``0.0.0.0``, so **any computer on the LAN** can watch via a browser
      (no install) or a native viewer. The printed URL uses the robot's LAN IP.
    """

    def __init__(self) -> None:
        import rerun as rr  # lazy: only needed when viz is enabled

        self._rr = rr
        rr.init(_env("WALKIE_VIZ_RECORDING", None, "walkie"))
        # The robot's map frame is gravity-aligned **+Z up**; declare it so Rerun's 3D
        # view shows the scene upright instead of its default Y-up (which tips every point
        # cloud / marker on its side — looks like a bad transform). Logged on the root so
        # the view inherits it regardless of where it's rooted, and on "world" for viewers
        # that root the 3D space there.
        try:
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        except Exception:  # noqa: BLE001 — never let a viz convention call break startup
            pass

        self._show_robot = _flag("WALKIE_VIZ_ROBOT", "WALKIE_EXPLORE_VIZ_ROBOT", "1")
        self._show_camera = _flag("WALKIE_VIZ_CAMERA", "WALKIE_EXPLORE_VIZ_CAMERA", "1")

        if not _flag("WALKIE_VIZ_SERVE", "WALKIE_EXPLORE_RERUN_SERVE", "1"):
            rr.spawn()  # local native window on the robot
            return

        grpc_port = int(_env("WALKIE_VIZ_GRPC_PORT", "WALKIE_EXPLORE_RERUN_GRPC_PORT", "9876"))
        web_port = int(_env("WALKIE_VIZ_WEB_PORT", "WALKIE_EXPLORE_RERUN_WEB_PORT", "9090"))
        host = _env("WALKIE_VIZ_HOST", "WALKIE_EXPLORE_RERUN_HOST", "").strip() or _lan_ip()
        cors = [o.strip() for o in _env("WALKIE_VIZ_CORS", "WALKIE_EXPLORE_RERUN_CORS", "*").split(",") if o.strip()]

        # gRPC data sink + HTTP browser app (both already bind 0.0.0.0). serve_grpc
        # advertises 127.0.0.1, but a REMOTE browser must reach the robot's LAN IP,
        # so build the connect URI with `host`. CORS must allow the page's origin
        # (the browser app at :web_port talks cross-port to the gRPC at :grpc_port).
        rr.serve_grpc(grpc_port=grpc_port, cors_allow_origin=cors)
        grpc_uri = f"rerun+http://{host}:{grpc_port}/proxy"
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=grpc_uri)

        viewer_url = f"http://{host}:{web_port}/?url={urllib.parse.quote(grpc_uri, safe='')}"
        print(
            "[viz] Rerun viewer live on the LAN — open from any computer:\n"
            f"          browser: {viewer_url}\n"
            f"          native : rerun {grpc_uri}\n"
            "          Can't connect from another machine? The servers bind 0.0.0.0,\n"
            "          so it's the robot's host firewall dropping the ports. Open BOTH\n"
            f"          (web page + data stream): "
            f"sudo ufw allow {web_port}/tcp && sudo ufw allow {grpc_port}/tcp"
        )

    # ------------------------------------------------------------------
    # Path handling
    # ------------------------------------------------------------------
    def _path(self, entity):
        """Build an escaped Rerun entity path from a ``"a/b"`` string or ``[a, b]`` list.

        A list lets callers keep raw segments that contain spaces (e.g. object ids
        carrying a class name) properly escaped; a string is split on ``/`` for the
        common ``"grasp/ee"`` convenience case.
        """
        parts = list(entity) if isinstance(entity, (list, tuple)) else [p for p in entity.split("/") if p]
        return self._rr.new_entity_path(parts)

    # ------------------------------------------------------------------
    # Generic primitives
    # ------------------------------------------------------------------
    def axes(self, entity, position, rotation=None, length: float = 0.1, labels: bool = False) -> None:
        """Draw an XYZ axis triad at ``position`` oriented by ``rotation`` (3x3, columns = axes).

        Rendered as three colored arrows (X red, Y green, Z blue), each the matching
        column of ``rotation`` scaled by ``length``. ``rotation`` defaults to identity
        (world-aligned axes). Use this to visualize a pose/frame — e.g. a grasp
        end-effector. (rerun 0.33 has no ``Transform3D(axis_length=)``; arrows are the
        reliable triad.)
        """
        rr = self._rr
        R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        o = [float(position[0]), float(position[1]), float(position[2])]
        vecs = (R * float(length)).T.tolist()  # rows = scaled columns of R = the 3 axes
        rr.log(
            self._path(entity),
            rr.Arrows3D(origins=[o, o, o], vectors=vecs, colors=_AXIS_COLORS,
                        labels=["x", "y", "z"] if labels else None),
        )

    def points(self, entity, positions, *, colors=None, radii=None, labels=None) -> None:
        rr = self._rr
        kw = {}
        if colors is not None:
            kw["colors"] = colors
        if radii is not None:
            kw["radii"] = radii
        if labels is not None:
            kw["labels"] = labels
        rr.log(self._path(entity), rr.Points3D(positions, **kw))

    def box(self, entity, center, half_sizes, *, color=None, label=None) -> None:
        rr = self._rr
        kw = {}
        if color is not None:
            kw["colors"] = [color]
        if label is not None:
            kw["labels"] = [label]
        rr.log(self._path(entity), rr.Boxes3D(centers=[list(center)], half_sizes=[list(half_sizes)], **kw))

    def arrow(self, entity, origin, vector, *, color=None, label=None) -> None:
        rr = self._rr
        kw = {}
        if color is not None:
            kw["colors"] = [color]
        if label is not None:
            kw["labels"] = [label]
        rr.log(self._path(entity), rr.Arrows3D(origins=[list(origin)], vectors=[list(vector)], **kw))

    def lines(self, entity, strips, *, labels=None, colors=None) -> None:
        rr = self._rr
        kw = {}
        if labels is not None:
            kw["labels"] = labels
        if colors is not None:
            kw["colors"] = colors
        rr.log(self._path(entity), rr.LineStrips3D(strips, **kw))

    def clear(self, entity, *, recursive: bool = True) -> None:
        rr = self._rr
        rr.log(self._path(entity), rr.Clear(recursive=recursive))

    # ------------------------------------------------------------------
    # Reusable map markers
    # ------------------------------------------------------------------
    def robot(self, robot_pose) -> None:
        """Mark the robot's position + heading on the map (z=0 ground plane)."""
        rr = self._rr
        if not self._show_robot or not robot_pose:
            return
        x = float(robot_pose.get("x", 0.0))
        y = float(robot_pose.get("y", 0.0))
        h = float(robot_pose.get("heading", 0.0))
        green = (0, 200, 0)
        rr.log("world/robot", rr.Points3D([[x, y, 0.0]], radii=[0.12], colors=[green], labels=["robot"]))
        length = 0.5
        rr.log(
            "world/robot/heading",
            rr.Arrows3D(origins=[[x, y, 0.0]],
                        vectors=[[math.cos(h) * length, math.sin(h) * length, 0.0]], colors=[green]),
        )

    def camera(self, cam_pose) -> None:
        """Mark the camera's 3D world position + viewing direction.

        ``cam_pose`` is a :class:`~interfaces.perception.geometry.CameraPose`: ``t`` is
        the optical center in world coords and ``R`` maps the optical frame into the
        world, where the camera looks along the optical +z axis, so ``R @ [0,0,1]`` is
        the world-frame look direction.
        """
        rr = self._rr
        if not self._show_camera or cam_pose is None:
            return
        t = np.asarray(cam_pose.t, dtype=float)
        forward = np.asarray(cam_pose.R, dtype=float) @ np.array([0.0, 0.0, 1.0])
        cyan = (0, 200, 220)
        rr.log("world/camera", rr.Points3D([t.tolist()], radii=[0.08], colors=[cyan], labels=["camera"]))
        rr.log("world/camera/forward",
               rr.Arrows3D(origins=[t.tolist()], vectors=[(forward * 0.5).tolist()], colors=[cyan]))


class NoOpViz:
    """A :class:`VizSession` whose every method does nothing.

    Returned by :func:`services.viz.get_viz` when viz is disabled (or the backend
    failed to build), so callers can draw unconditionally — ``ctx.viz.axes(...)`` is
    always safe and simply renders nothing.
    """

    def axes(self, *args, **kwargs) -> None: ...
    def points(self, *args, **kwargs) -> None: ...
    def box(self, *args, **kwargs) -> None: ...
    def arrow(self, *args, **kwargs) -> None: ...
    def lines(self, *args, **kwargs) -> None: ...
    def clear(self, *args, **kwargs) -> None: ...
    def robot(self, *args, **kwargs) -> None: ...
    def camera(self, *args, **kwargs) -> None: ...
