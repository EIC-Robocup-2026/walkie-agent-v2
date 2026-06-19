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
    # Then perform the pick with the chosen source:
    SOURCE=pos EXECUTE=1 uv run python -m manual_tests.test_pick_and_place "water bottle"

Deliberately outside tests/ (pyproject testpaths) so pytest never collects it —
it drives real hardware.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from client import WalkieAIClient
from tasks.base import TaskContext
from tasks.common import initialize_graphs, initialize_llm_model, initialize_robot, load_task_config
from tasks.manipulation import db, grasp, pick_object
from tasks.manipulation.types import DetectedObject


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
