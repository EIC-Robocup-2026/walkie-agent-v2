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


class RerunViz:
    """Streams the graph to a Rerun viewer (spawned, or served for remote view)."""

    def __init__(self) -> None:
        import rerun as rr  # lazy: only needed when viz is enabled

        self._rr = rr
        rr.init("walkie_graphs", spawn=False)
        if os.getenv("WALKIE_GRAPHS_RERUN_SERVE", "0").lower() in ("1", "true", "yes"):
            # Headless/remote: serve a web viewer instead of spawning a window.
            try:
                rr.serve_web()
            except Exception:
                rr.spawn()
        else:
            rr.spawn()

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
            label = n.class_name + (f": {n.best_caption}" if n.best_caption else "")
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
