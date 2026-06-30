"""Render the scene graph into the shared viz session (see :mod:`services.viz`).

The scene-graph-specific *layer*: it knows how to turn a
:class:`~walkie_world.scene.store.SceneStore` into draw calls, but owns no Rerun
session of its own — it draws through a :class:`~services.viz.session.VizSession` (real
or no-op) under the ``world/...`` namespace, so it coexists with task-level
visualization (e.g. a grasp triad under ``grasp/...``) in one viewer.

:meth:`SceneGraphViz.update` logs, in the ``world`` space:
- one colored point cloud per confirmed object (colored by class),
- one AABB per object (``WALKIE_EXPLORE_VIZ_BOXES``), labelled with the class name
  (``WALKIE_EXPLORE_VIZ_LABELS``); when boxes are off but labels on, the class name is
  anchored to the object centroid as a standalone marker instead,
- the geometric relations as labelled line segments between object centroids,
- the robot position + heading and the camera 3D position + look direction (reusable
  markers from the session, gated by ``WALKIE_VIZ_ROBOT`` / ``WALKIE_VIZ_CAMERA``),
- optionally the TSDF structural cloud as a faint "background"
  (``WALKIE_EXPLORE_VIZ_BACKGROUND``), passed in by the build.

:meth:`update_markers` is the cheap subset (just robot + camera) the capture thread
calls every tick so those markers stay live between the occasional batch builds.
"""

from __future__ import annotations

import os

import numpy as np


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _class_color(name: str) -> tuple[int, int, int]:
    """Deterministic per-class RGB so a class keeps one color across builds."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


class SceneGraphViz:
    """Translates a ``SceneStore`` into primitive draw calls on a shared viz session."""

    def __init__(self, viz) -> None:
        self._viz = viz  # a services.viz session (RerunSession or NoOpViz)
        self._show_boxes = _flag("WALKIE_EXPLORE_VIZ_BOXES")
        self._show_labels = _flag("WALKIE_EXPLORE_VIZ_LABELS")
        self._show_background = _flag("WALKIE_EXPLORE_VIZ_BACKGROUND")

    def update(self, store, *, robot_pose=None, cam_pose=None, structural=None, rooms=None) -> None:
        """Redraw the whole scene from the current ``store`` (called after each build).

        ``rooms`` (an optional ``{name: Room}`` from the world map) draws each room's
        boundary polygon as walls. Object point clouds + AABB boxes come from ``store``:
        a map-seeded object (no cloud yet) shows as its bounding box, and once perception
        promotes it to a real cloud the cloud is drawn — the bbox->point-cloud swap.
        """
        viz = self._viz

        # Room boundary polygons as walls (closed line strips on the floor plane).
        self._draw_rooms(rooms)

        # Faint structural cloud (TSDF map) as background, when present.
        if self._show_background and structural is not None and len(structural):
            bg = np.asarray(structural)
            if len(bg) > 100_000:  # bound the per-update stream size
                bg = bg[:: len(bg) // 100_000 + 1]
            viz.points("world/background", bg, colors=[(128, 128, 128)], radii=0.008)

        nodes = store.all_objects()
        # Clear the object subtree first so pruned/merged nodes vanish, then redraw all.
        viz.clear("world/objects", recursive=True)
        for n in nodes:
            color = _class_color(n.class_name)
            pts = getattr(n, "points", None)
            # n.id can contain spaces (it carries the class name); pass the path as a
            # list so each segment is escaped properly.
            if pts is not None and len(pts):
                viz.points(["world", "objects", n.id, "points"], np.asarray(pts), colors=[color])
            if self._show_boxes:
                half = [e / 2.0 for e in n.extent]
                viz.box(["world", "objects", n.id, "box"], list(n.centroid), half,
                        color=color, label=n.class_name if self._show_labels else None)
            elif self._show_labels:
                viz.points(["world", "objects", n.id, "label"], [list(n.centroid)],
                           radii=[0.01], labels=[n.class_name], colors=[color])

        self.update_markers(robot_pose=robot_pose, cam_pose=cam_pose)

        # Geometric relations as labelled segments between object centroids.
        centroids = {n.id: n.centroid for n in nodes}
        strips, labels = [], []
        for r in store.all_relations():
            if r.src_id in centroids and r.dst_id in centroids:
                strips.append([list(centroids[r.src_id]), list(centroids[r.dst_id])])
                labels.append(r.predicate)
        if strips:
            viz.lines("world/relations", strips, labels=labels)
        else:
            viz.clear("world/relations", recursive=True)

    def _draw_rooms(self, rooms) -> None:
        """Draw each room's boundary polygon as a closed wall line strip (z=0 floor)."""
        if not rooms:
            return
        strips = []
        for room in rooms.values():
            poly = getattr(room, "polygon", ()) or ()
            if len(poly) >= 2:
                ring = [[float(x), float(y), 0.0] for (x, y) in poly]
                ring.append(ring[0])  # close the loop back to the first vertex
                strips.append(ring)
        if strips:
            self._viz.lines("world/rooms", strips)
        else:
            self._viz.clear("world/rooms", recursive=True)

    def update_markers(self, *, robot_pose=None, cam_pose=None) -> None:
        """Cheap live markers — robot position/heading + camera position/look direction."""
        self._viz.robot(robot_pose)
        self._viz.camera(cam_pose)

    def update_live(self, frame, detections, *, exclude=()) -> None:
        """Live feed: draw the CURRENT frame's lifted detections under ``world/live``.

        Refreshed every capture tick (when ``WALKIE_EXPLORE_VIZ_LIVE=1``) so you can watch
        the scene fill in as walkie_graphs takes each snapshot — independent of the slower
        batch build that produces the persistent ``world/objects``. Lifts via the canonical
        :meth:`CameraSnapshot.mask_to_points`, so the live clouds sit exactly where the
        built object clouds will.
        """
        viz = self._viz
        viz.clear("world/live", recursive=True)  # show only the current frame
        if not getattr(frame, "has_geometry", False):
            return
        for i, d in enumerate(detections):
            cls = getattr(d, "class_name", "") or ""
            if cls.lower() in exclude or getattr(d, "mask", None) is None:
                continue
            try:
                pts = frame.mask_to_points(d.mask)
            except Exception:  # noqa: BLE001
                continue
            if len(pts):
                viz.points(["world", "live", str(i)], np.asarray(pts), colors=[_class_color(cls)])


def build_scene_viz():
    """Wrap the process-wide :mod:`services.viz` session in a :class:`SceneGraphViz`.

    Returns ``None`` when viz is disabled (the session is a ``NoOpViz``) so the service
    can keep its ``self.viz is None`` fast path. Calling this is also what **starts** the
    shared Rerun session (it's lazy on the first ``get_viz()``).
    """
    try:
        from services.viz import NoOpViz, get_viz

        viz = get_viz()
        if isinstance(viz, NoOpViz):
            return None
        return SceneGraphViz(viz)
    except Exception as e:  # noqa: BLE001 — viz is best-effort, never block the robot
        print(f"[graphs] visualizer unavailable: {e}")
        return None
