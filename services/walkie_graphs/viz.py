"""Real-time 3D visualization of the scene graph with Rerun (rerun.io).

Optional: ``rerun`` is imported lazily and only when ``WALKIE_GRAPHS_VIZ=rerun``,
so the core module and tests never need it. Install with ``uv sync --extra graphs``.

Each :meth:`RerunViz.update` logs, in the ``world`` space:
- one colored point cloud per object (colored by class),
- one AABB per object (``WALKIE_GRAPHS_VIZ_BOXES``), labelled with the class name
  (``WALKIE_GRAPHS_VIZ_LABELS``); when boxes are off but labels on, the class name
  is anchored to the object centroid as a standalone marker instead,
- the geometric relations as labelled line segments between object centroids,
- the robot position + heading, and the camera's 3D position + look direction.
"""

from __future__ import annotations

import math
import os
import socket
import urllib.parse

import numpy as np


def build_viz(backend: str):
    """Factory: return a visualizer for ``backend`` ('rerun'), else None."""
    if backend == "rerun":
        return RerunViz()
    return None


def _class_color(name: str) -> tuple[int, int, int]:
    """Deterministic per-class RGB so a class keeps one color across frames."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


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


class RerunViz:
    """Streams the graph to a Rerun viewer.

    Two modes (``WALKIE_GRAPHS_RERUN_SERVE``):

    - ``0`` (default): ``rr.spawn()`` — a native viewer window on the robot. Needs a
      display and is **only visible on the robot itself**.
    - ``1``: serve a browser viewer over HTTP + a gRPC data stream, both bound to
      ``0.0.0.0``, so **any computer on the LAN** can watch via a browser (no install)
      or a native viewer. The printed URL uses the robot's LAN IP.
    """

    def __init__(self) -> None:
        import rerun as rr  # lazy: only needed when viz is enabled

        self._rr = rr
        rr.init("walkie_graphs")

        self._show_boxes = os.getenv("WALKIE_GRAPHS_VIZ_BOXES", "1").lower() in ("1", "true", "yes")
        self._show_labels = os.getenv("WALKIE_GRAPHS_VIZ_LABELS", "1").lower() in ("1", "true", "yes")
        self._show_robot = os.getenv("WALKIE_GRAPHS_VIZ_ROBOT", "1").lower() in ("1", "true", "yes")
        self._show_camera = os.getenv("WALKIE_GRAPHS_VIZ_CAMERA", "1").lower() in ("1", "true", "yes")

        if os.getenv("WALKIE_GRAPHS_RERUN_SERVE", "0").lower() not in ("1", "true", "yes"):
            rr.spawn()  # local native window on the robot
            return

        grpc_port = int(os.getenv("WALKIE_GRAPHS_RERUN_GRPC_PORT", "9876"))
        web_port = int(os.getenv("WALKIE_GRAPHS_RERUN_WEB_PORT", "9090"))
        host = os.getenv("WALKIE_GRAPHS_RERUN_HOST", "").strip() or _lan_ip()
        cors = [o.strip() for o in os.getenv("WALKIE_GRAPHS_RERUN_CORS", "*").split(",") if o.strip()]

        # gRPC data sink + HTTP browser app (both already bind 0.0.0.0). serve_grpc
        # advertises 127.0.0.1, but a REMOTE browser must reach the robot's LAN IP,
        # so build the connect URI with `host`. CORS must allow the page's origin
        # (the browser app at :web_port talks cross-port to the gRPC at :grpc_port).
        rr.serve_grpc(grpc_port=grpc_port, cors_allow_origin=cors)
        grpc_uri = f"rerun+http://{host}:{grpc_port}/proxy"
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=grpc_uri)

        viewer_url = f"http://{host}:{web_port}/?url={urllib.parse.quote(grpc_uri, safe='')}"
        print(
            "[graphs] Rerun viewer live on the LAN — open from any computer:\n"
            f"          browser: {viewer_url}\n"
            f"          native : rerun {grpc_uri}\n"
            "          Can't connect from another machine? The servers bind 0.0.0.0,\n"
            "          so it's the robot's host firewall dropping the ports. Open BOTH\n"
            f"          (web page + data stream): "
            f"sudo ufw allow {web_port}/tcp && sudo ufw allow {grpc_port}/tcp"
        )

    def update(self, memory, robot_pose=None, cam_pose=None) -> None:
        rr = self._rr
        nodes = memory.all_objects()
        for n in nodes:
            color = _class_color(n.class_name)
            pts = memory.load_pcd(n.id)
            if len(pts):
                rr.log(f"world/objects/{n.id}/points", rr.Points3D(pts, colors=[color]))
            if self._show_boxes:
                half = [e / 2.0 for e in n.extent]
                rr.log(
                    f"world/objects/{n.id}/box",
                    rr.Boxes3D(centers=[list(n.centroid)], half_sizes=[half],
                               labels=[n.class_name] if self._show_labels else None,
                               colors=[color]),
                )
            elif self._show_labels:
                # Boxes hidden but labels wanted: anchor the class name to the
                # object's centroid as a standalone marker (tiny point + label).
                rr.log(
                    f"world/objects/{n.id}/label",
                    rr.Points3D([list(n.centroid)], radii=[0.01],
                                labels=[n.class_name], colors=[color]),
                )

        self._log_robot(robot_pose)
        self._log_camera(cam_pose)

        centroids = {n.id: n.centroid for n in nodes}
        strips, labels = [], []
        for r in memory.all_relations():
            if r.src_id in centroids and r.dst_id in centroids:
                strips.append([list(centroids[r.src_id]), list(centroids[r.dst_id])])
                labels.append(r.predicate)
        if strips:
            rr.log("world/relations", rr.LineStrips3D(strips, labels=labels))
        else:
            rr.log("world/relations", rr.Clear(recursive=True))

    def _log_robot(self, robot_pose) -> None:
        """Mark the robot's position + heading on the map (z=0 ground plane)."""
        rr = self._rr
        if not self._show_robot or not robot_pose:
            return
        x = float(robot_pose.get("x", 0.0))
        y = float(robot_pose.get("y", 0.0))
        h = float(robot_pose.get("heading", 0.0))
        green = (0, 200, 0)
        rr.log(
            "world/robot",
            rr.Points3D([[x, y, 0.0]], radii=[0.12], colors=[green], labels=["robot"]),
        )
        length = 0.5
        rr.log(
            "world/robot/heading",
            rr.Arrows3D(
                origins=[[x, y, 0.0]],
                vectors=[[math.cos(h) * length, math.sin(h) * length, 0.0]],
                colors=[green],
            ),
        )

    def _log_camera(self, cam_pose) -> None:
        """Mark the camera's 3D world position + viewing direction.

        ``cam_pose`` is a :class:`~walkie_graphs.geometry.CameraPose`: ``t`` is the
        optical center in world coords and ``R`` maps the optical frame into the world,
        where the camera looks along the optical +z axis, so ``R @ [0,0,1]`` is the
        world-frame look direction.
        """
        rr = self._rr
        if not self._show_camera or cam_pose is None:
            return
        t = np.asarray(cam_pose.t, dtype=float)
        forward = np.asarray(cam_pose.R, dtype=float) @ np.array([0.0, 0.0, 1.0])
        cyan = (0, 200, 220)
        rr.log(
            "world/camera",
            rr.Points3D([t.tolist()], radii=[0.08], colors=[cyan], labels=["camera"]),
        )
        rr.log(
            "world/camera/forward",
            rr.Arrows3D(origins=[t.tolist()], vectors=[(forward * 0.5).tolist()], colors=[cyan]),
        )
