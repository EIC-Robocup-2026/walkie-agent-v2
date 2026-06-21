"""Render the 3D scene graph into a shared viz session (see :mod:`services.viz`).

This module is the scene-graph-specific *layer*: it knows how to turn a
:class:`~services.walkie_graphs.memory.GraphMemory` into draw calls, but owns no
Rerun session of its own — it draws through a :class:`~services.viz.session.VizSession`
(real or no-op) under the ``world/...`` namespace, so it coexists with task-level
visualization (e.g. a grasp triad under ``grasp/...``) in one viewer.

Each :meth:`SceneGraphViz.update` logs, in the ``world`` space:
- one colored point cloud per object (colored by class),
- one AABB per object (``WALKIE_GRAPHS_VIZ_BOXES``), labelled with the class name
  (``WALKIE_GRAPHS_VIZ_LABELS``); when boxes are off but labels on, the class name is
  anchored to the object centroid as a standalone marker instead,
- the geometric relations as labelled line segments between object centroids,
- the robot position + heading and the camera 3D position + look direction (reusable
  markers provided by the session, gated by ``WALKIE_VIZ_ROBOT``/``WALKIE_VIZ_CAMERA``).
"""

from __future__ import annotations

import os

import numpy as np


def _class_color(name: str) -> tuple[int, int, int]:
    """Deterministic per-class RGB so a class keeps one color across frames."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


class SceneGraphViz:
    """Translates a ``GraphMemory`` into primitive draw calls on a shared viz session."""

    def __init__(self, viz) -> None:
        self._viz = viz  # a services.viz session (RerunSession or NoOpViz)
        self._show_boxes = os.getenv("WALKIE_GRAPHS_VIZ_BOXES", "1").lower() in ("1", "true", "yes")
        self._show_labels = os.getenv("WALKIE_GRAPHS_VIZ_LABELS", "1").lower() in ("1", "true", "yes")
        self._show_background = os.getenv("WALKIE_GRAPHS_VIZ_BACKGROUND", "1").lower() in ("1", "true", "yes")

    def update(self, memory, robot_pose=None, cam_pose=None) -> None:
        viz = self._viz

        if self._show_background and getattr(memory, "background", None) is not None:
            bg = memory.background.points()
            if len(bg):
                if len(bg) > 100_000:  # bound the per-update stream size
                    bg = bg[:: len(bg) // 100_000 + 1]
                viz.points("world/background", bg, colors=[(128, 128, 128)], radii=0.008)

        nodes = memory.all_objects()
        for n in nodes:
            color = _class_color(n.class_name)
            pts = memory.load_pcd(n.id)
            # n.id can contain spaces (it carries the class name); pass the path as a
            # list so each segment is escaped properly (a raw "a/b" string is split on
            # "/", which would also work, but the list keeps ids with odd chars safe).
            if len(pts):
                viz.points(["world", "objects", n.id, "points"], pts, colors=[color])
            if self._show_boxes:
                half = [e / 2.0 for e in n.extent]
                viz.box(["world", "objects", n.id, "box"], list(n.centroid), half,
                        color=color, label=n.class_name if self._show_labels else None)
            elif self._show_labels:
                # Boxes hidden but labels wanted: anchor the class name to the object's
                # centroid as a standalone marker (tiny point + label).
                viz.points(["world", "objects", n.id, "label"], [list(n.centroid)],
                           radii=[0.01], labels=[n.class_name], colors=[color])

        viz.robot(robot_pose)
        viz.camera(cam_pose)

        centroids = {n.id: n.centroid for n in nodes}
        strips, labels = [], []
        for r in memory.all_relations():
            if r.src_id in centroids and r.dst_id in centroids:
                strips.append([list(centroids[r.src_id]), list(centroids[r.dst_id])])
                labels.append(r.predicate)
        if strips:
            viz.lines("world/relations", strips, labels=labels)
        else:
            viz.clear("world/relations", recursive=True)
