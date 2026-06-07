"""Real-time 3D visualization of the scene graph with Rerun (rerun.io).

Optional: ``rerun`` is imported lazily and only when ``WALKIE_GRAPHS_VIZ=rerun``,
so the core module and tests never need it. Install with ``uv sync --extra graphs``.

Each :meth:`RerunViz.update` logs, in the ``world`` space:
- one colored point cloud per object (colored by class),
- one labelled AABB per object (class + caption),
- the geometric relations as labelled line segments between object centroids.
"""

from __future__ import annotations

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

    def update(self, memory) -> None:
        rr = self._rr
        nodes = memory.all_objects()
        seen_ids = set()
        for n in nodes:
            seen_ids.add(n.id)
            color = _class_color(n.class_name)
            pts = memory.load_pcd(n.id)
            if len(pts):
                rr.log(f"world/objects/{n.id}/points", rr.Points3D(pts, colors=[color]))
            half = [e / 2.0 for e in n.extent]
            label = n.class_name
            rr.log(
                f"world/objects/{n.id}/box",
                rr.Boxes3D(centers=[list(n.centroid)], half_sizes=[half],
                           labels=[label], colors=[color]),
            )

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
