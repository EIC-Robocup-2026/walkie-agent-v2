"""Manual, on-robot smoke test for the GraspNet pick pipeline.

Needs the robot + walkie-ai-server (with the grasp service) up, and at least one
object already scanned into the walkie_graphs store. It resolves the object node,
prints the GraspNet `from_cloud` result, and (optionally) executes one real pick.

Run as a module so the repo root is on sys.path:

    uv run python -m manual_tests.test_pick_and_place "water bottle"
    EXECUTE=1 uv run python -m manual_tests.test_pick_and_place "water bottle"

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
from tasks.manipulation import db, pick_object
from tasks.manipulation.types import DetectedObject


def main() -> None:
    load_dotenv()
    load_task_config(os.path.dirname(__file__))

    query = sys.argv[1] if len(sys.argv) > 1 else "bottle"
    execute = os.getenv("EXECUTE", "0").lower() in ("1", "true", "yes")

    walkie = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    graphs = initialize_graphs(model, walkie_ai, walkie)
    ctx = TaskContext(walkie=walkie, walkieAI=walkie_ai, model=model, graphs=graphs)

    try:
        node = db.resolve_object_node(graphs, query)
        if node is None:
            print(f"[test] no DB node for {query!r}; scan it first.")
            return
        print(f"[test] node {node.id} class={node.class_name} centroid={node.centroid} "
              f"aabb={node.aabb_min}..{node.aabb_max}")

        cloud = db.object_cloud_pc2(graphs, node)
        if cloud is None:
            print("[test] no stored cloud for node.")
            return
        print(f"[test] cloud points={cloud['width']} frame={cloud['header']['frame_id']}")

        result = walkie.robot.grasp.from_cloud(
            cloud,
            score_threshold=float(os.getenv("WALKIE_GRASP_FROM_CLOUD_SCORE_TH", "0.0")),
            max_grasps=int(os.getenv("WALKIE_GRASP_FROM_CLOUD_MAX", "20")),
        )
        if not result or not result.get("grasps"):
            print(f"[test] GraspNet returned no grasps: {result}")
            return
        print(f"[test] planning_frame={result.get('planning_frame')} "
              f"n_grasps={len(result['grasps'])} object_size={result.get('object_size')}")
        for i, g in enumerate(result["grasps"][:5]):
            print(f"  grasp[{i}] score={g.get('score'):.3f} width={g.get('width')} "
                  f"pos={g['position']} quat={g['orientation']} approach={g.get('approach_position')}")

        if not execute:
            print("[test] dry run (set EXECUTE=1 to perform the pick).")
            return

        obj = DetectedObject(
            bbox_xyxy=(0, 0, 0, 0),
            class_name=node.class_name,
            confidence=1.0,
            world_xy=(node.centroid[0], node.centroid[1]),
            world_xyz=tuple(node.centroid),
            node_id=node.id,
        )
        grasped = pick_object(ctx, obj)
        print(f"[test] pick_object -> grasped={grasped}")
    finally:
        if graphs is not None:
            graphs.stop()
        walkie.close()


if __name__ == "__main__":
    main()
