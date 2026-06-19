"""Manual, on-robot smoke test for the GraspNet pick pipeline.

Needs the robot + walkie-ai-server (with the grasp service) up. The cloud/pos
sources also need the object already scanned into the walkie_graphs store; the
mask source needs only the live camera. It resolves the object, prints the
GraspNet result for the selected source, and (optionally) executes one real pick.

Run as a module so the repo root is on sys.path:

    # A/B the three GraspNet inputs against the same object:
    SOURCE=cloud uv run python -m manual_tests.test_pick_and_place "water bottle"
    SOURCE=mask  uv run python -m manual_tests.test_pick_and_place "water bottle"
    SOURCE=pos   uv run python -m manual_tests.test_pick_and_place "water bottle"
    # Then perform the pick with the chosen source. With EXECUTE the tester steps
    # through every arm/base motion (Enter=do, s=skip, q=abort); CONFIRM=0 disables.
    SOURCE=pos EXECUTE=1 uv run python -m manual_tests.test_pick_and_place "water bottle"

RViz markers are published by default (VIZ=0 to disable): the ranked grasp
candidates as arrows (best green -> red), the best approach waypoint as a sphere
with a score label (in base_footprint), and the table collision box as a
translucent cube (in map). Open RViz and add a MarkerArray display on
'walkie/viz_markers'; MoveIt's own displays then show the planned arm motion and
the attached/table collision objects when EXECUTE=1.

Deliberately outside tests/ (pyproject testpaths) so pytest never collects it —
it drives real hardware.
"""

from __future__ import annotations

import math
import os
import sys

from dotenv import load_dotenv

from client import WalkieAIClient
from tasks.base import TaskContext
from tasks.common import initialize_graphs, initialize_llm_model, initialize_robot, load_task_config
from tasks.manipulation import db, grasp, pick_object
from tasks.manipulation.types import DetectedObject

# RViz marker shape constants (re-exported by the SDK).
from walkie_sdk import ARROW, CUBE, SPHERE, TEXT_VIEW_FACING

_VIZ_NS = "pnp_test"


def _yaw_to_quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _rank_color(i: int, n: int) -> list[float]:
    """Best grasp (i=0) green, fading to red down the ranking. RGBA."""
    t = 0.0 if n <= 1 else i / (n - 1)
    return [t, 1.0 - t, 0.0, 1.0]


def draw_grasps(walkie, result: dict, *, max_show: int = 5) -> None:
    """Draw GraspNet candidates as arrows (ranked color) + the best approach pose.

    Grasps come back in ``planning_frame`` (base_footprint). The best grasp is
    drawn green with a SPHERE at its approach waypoint and a score label; lower
    -ranked grasps fade to red. Best-effort — viz failures never abort the test.
    """
    viz = walkie.robot.viz
    frame = result.get("planning_frame") or "base_footprint"
    grasps = result.get("grasps", [])[:max_show]
    try:
        viz.clear_markers()
    except Exception:  # noqa: BLE001
        pass
    for i, g in enumerate(grasps):
        color = _rank_color(i, len(grasps))
        try:
            viz.draw_marker(
                position=[float(v) for v in g["position"]],
                quaternion=[float(v) for v in g["orientation"]],
                frame_id=frame, marker_type=ARROW,
                scale=[0.10, 0.02, 0.02], color=color,
                marker_id=100 + i, ns=f"{_VIZ_NS}/grasps",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[viz] grasp arrow {i} failed ({exc})")
        ap = g.get("approach_position")
        if i == 0 and ap is not None:
            try:
                viz.draw_marker(
                    position=[float(v) for v in ap], frame_id=frame,
                    marker_type=SPHERE, scale=[0.04, 0.04, 0.04],
                    color=[0.1, 0.4, 1.0, 0.9], marker_id=200, ns=f"{_VIZ_NS}/approach",
                )
                px, py, pz = (float(v) for v in g["position"])
                viz.draw_marker(
                    position=[px, py, pz + 0.08], frame_id=frame,
                    marker_type=TEXT_VIEW_FACING, scale=[0.0, 0.0, 0.04],
                    color=[1.0, 1.0, 1.0, 1.0], marker_id=201, ns=f"{_VIZ_NS}/label",
                    text=f"score={g.get('score')}",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[viz] approach/label failed ({exc})")
    print(f"[viz] drew {len(grasps)} grasp(s) in '{frame}' on walkie/viz_markers")


def draw_table_box(walkie, node) -> None:
    """Draw the surface collision box (floor -> top_z) as a translucent CUBE in map."""
    if node is None:
        return
    box = db.node_table_box(node)
    if box is None:
        return
    (cx, cy, top_z, yaw), (dx, dy) = box
    try:
        walkie.robot.viz.draw_marker(
            position=[cx, cy, top_z / 2.0], quaternion=_yaw_to_quat(yaw),
            frame_id="map", marker_type=CUBE,
            scale=[max(dx, 0.01), max(dy, 0.01), max(top_z, 0.01)],
            color=[0.6, 0.6, 0.6, 0.3], marker_id=300, ns=f"{_VIZ_NS}/table",
        )
        print(f"[viz] drew table box center=({cx:.2f},{cy:.2f},{top_z/2:.2f}) "
              f"size=({dx:.2f},{dy:.2f},{top_z:.2f}) in 'map'")
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] table box failed ({exc})")


def _obj_from_node(node) -> DetectedObject:
    return DetectedObject(
        bbox_xyxy=(0, 0, 0, 0),
        class_name=node.class_name,
        confidence=1.0,
        world_xy=(node.centroid[0], node.centroid[1]),
        world_xyz=tuple(node.centroid),
        node_id=node.id,
    )


def _raw_result(ctx, graphs, node, obj, source):
    """Call the source-specific GraspNet entry directly, for rich printing."""
    if source == "mask":
        return grasp._grasp_from_mask(ctx, obj)
    if source == "pos":
        return grasp._grasp_from_pos(ctx, graphs, node)
    return grasp._grasp_from_cloud(ctx, graphs, node)


def main() -> None:
    load_dotenv()
    load_task_config(os.path.dirname(__file__))

    query = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    source = os.getenv("SOURCE", os.getenv("WALKIE_GRASP_SOURCE", "cloud")).strip().lower()
    os.environ["WALKIE_GRASP_SOURCE"] = source  # so plan_grasp/pick_object agree
    execute = os.getenv("EXECUTE", "0").lower() in ("1", "true", "yes")
    viz = os.getenv("VIZ", "1").lower() in ("1", "true", "yes")
    # CONFIRM=1 -> step through every arm/base motion (Enter=do, s=skip, q=abort).
    # Default on when EXECUTE so a tester gates each real move; CONFIRM=0 disables.
    if os.getenv("CONFIRM", "1" if execute else "0").lower() in ("1", "true", "yes"):
        os.environ["WALKIE_MANIP_CONFIRM"] = "1"

    walkie = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    graphs = initialize_graphs(model, walkie_ai, walkie)
    ctx = TaskContext(walkie=walkie, walkieAI=walkie_ai, model=model, graphs=graphs)

    try:
        node = db.resolve_object_node(graphs, query)
        if node is None and source in ("cloud", "pos"):
            print(f"[test] no DB node for {query!r}; scan it first (or use SOURCE=mask).")
            return
        if node is not None:
            print(f"[test] node {node.id} class={node.class_name} centroid={node.centroid} "
                  f"aabb={node.aabb_min}..{node.aabb_max}")
        obj = _obj_from_node(node) if node is not None else DetectedObject(
            bbox_xyxy=(0, 0, 0, 0), class_name=query, confidence=1.0,
        )

        print(f"[test] GraspNet source = {source!r}")
        result = _raw_result(ctx, graphs, node, obj, source)
        if not result or not result.get("grasps"):
            print(f"[test] GraspNet returned no grasps: {result}")
            return
        print(f"[test] planning_frame={result.get('planning_frame')} "
              f"n_grasps={len(result['grasps'])} object_size={result.get('object_size')}")
        for i, g in enumerate(result["grasps"][:5]):
            print(f"  grasp[{i}] score={g.get('score')} width={g.get('width')} "
                  f"antipodal={g.get('antipodal_score')} pos={g['position']} "
                  f"quat={g['orientation']} approach={g.get('approach_position')}")

        if viz:
            # Resolve the surface node the executor would use, so the drawn table
            # box matches what gets added to the MoveIt planning scene.
            surface = db.resolve_surface_node(
                graphs, os.getenv("WALKIE_SURFACE_CLASS", "table"),
                near=obj.world_xy,
            ) if graphs is not None else None
            draw_grasps(walkie, result)
            draw_table_box(walkie, surface)
            print("[test] markers published — open RViz, add MarkerArray on "
                  "'walkie/viz_markers' (grasps in base_footprint, table in map).")

        if not execute:
            print("[test] dry run (set EXECUTE=1 to perform the pick).")
            return

        grasped = pick_object(ctx, obj)
        print(f"[test] pick_object -> grasped={grasped}")
    finally:
        if graphs is not None:
            graphs.stop()
        walkie.close()


if __name__ == "__main__":
    main()
