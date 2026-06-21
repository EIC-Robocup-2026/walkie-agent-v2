"""On-robot grasp bring-up harness (relocated from tasks/Restaurant/subtasks.py).

This is the teammate's grasp scratchpad — originally the ``TestTask`` SubTask added
to ``tasks/Restaurant/subtasks.py`` in commit 8fc1064 ("test: grasp"). It is *not*
Restaurant-challenge logic; it's a quick manual loop for bringing up the arm +
GraspNet (``tasks.skills.grasp_object``) on the real robot. It was moved here intact
when the full Phase 0-3 Restaurant implementation superseded the initial scaffold —
so the harness survives without sitting in the challenge's serial task.

Run on the robot (needs walkie-ai-server + the arm; NOT collected by pytest, it lives
in manual_tests/ and is __main__-guarded):

    uv run python -m manual_tests.test_restaurant_grasp                 # default "red can"
    uv run python -m manual_tests.test_restaurant_grasp "blue bottle"   # custom target
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from scipy.spatial.transform import Rotation

from client import WalkieAIClient
from tasks.base import StepResult, SubTask, TaskContext
from tasks.common import initialize_robot, load_task_config
from tasks.skills import grasp_object


class TestTask(SubTask):
    """A simple test subtask, for quick manual testing of the infrastructure."""

    critical = True

    def __init__(self, target: str = "red can"):
        super().__init__()
        self.target = target

    def run(self, ctx: TaskContext) -> StepResult:
        # print("[test] running test subtask")
        ctx.walkie.arm.left.gripper(1.0, blocking=True)  # open
        ctx.walkie.arm.left.go_to_home(pose_name="standby", blocking=False)
        grasp_pos = grasp_object(ctx, prompts=[self.target], standoff_m=0.2)
        if grasp_pos is None:
            print("[test] no grasp found")
            return StepResult.RETRY
        print(f"[test] grasp: {grasp_pos.grasp_xyz} score={grasp_pos.score:.3f}")

        # grasp_pos is in the map frame (positions + 3x3 rotation); the arm wants
        # RPY euler radians, so convert. frame_id="map" because the pose is map-frame.
        roll, pitch, yaw = Rotation.from_matrix(grasp_pos.rotation).as_euler("xyz")
        print(f"[test] grasp RPY (rad): {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
        ee = ctx.walkie.arm.get_ee_pose("left_arm", frame_id="map")  # warm up the transform cache
        result = ctx.walkie.arm.go_to_pose(
            ee["x"], ee["y"], ee["z"], roll, pitch, yaw,
            group_name="left_arm", frame_id="map", blocking=True,
        )
        print(result)
        return StepResult.DONE


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "red can"
    load_dotenv()
    # Reuse the Restaurant task config (RESTAURANT_* / manipulation knobs).
    load_task_config(Path(__file__).resolve().parent.parent / "tasks" / "Restaurant")

    walkie = initialize_robot()
    walkie_ai = WalkieAIClient()
    # The grasp path doesn't use the LLM, so skip model construction (no key needed).
    ctx = TaskContext(walkie=walkie, walkieAI=walkie_ai, model=None)
    try:
        print(TestTask(target).run(ctx))
    finally:
        walkie.close()


if __name__ == "__main__":
    main()
