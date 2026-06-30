"""Entrypoint for the Finals task (rulebook chapter 6).

    uv run python -m tasks.Final.run
    DISABLE_LISTENING=1 uv run python -m tasks.Final.run   # type instead of speak

Builds the shared world + the Walkie agent stack (WalkieBrain) and runs the Finals
envelope (subtasks.py): enter the arena → welcome the guest, move the laundry basket,
close the dishwasher (fixed high-value problems) → patrol the rooms, letting the agent
find + solve open-ended problems → finish. RobotContext is initialized (stage="ready")
so the agents' perception middleware sees the live perception.json the producer writes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import WalkieBrain, initialize_llm_model, initialize_robot, load_task_config
from ..scoring import ScoreTracker
from .scoring import FINAL_SHEET
from .subtasks import build_final_task
from agents.core.robot_context import RobotContext
from client import WalkieAIClient
from walkie_world import WalkieWorld

import open3d as o3d

# Quiet Open3D (matches GPSR/run.py).
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    disable_listening = os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes")

    # Initialize RobotContext (perception path + ready stage) BEFORE the brain so the
    # producer writes the live perception.json and the agents' perception middleware
    # injects the `## Current perception` block.
    rc = RobotContext.init(perception_path=os.getenv("PERCEPTION_PATH", "perception.json"))
    rc.stage = "ready"

    # One shared world: arena nouns the parser grounds against, the scene memory the
    # patrol/find recall from, and the people store for welcomed guests.
    world = WalkieWorld(
        embed_text=(lambda q: walkie_ai.image.embed_text(q)),
        enable_people=True,
    )

    scorecard = ScoreTracker(
        FINAL_SHEET, path=os.getenv("FINAL_SCORECARD_PATH", "final_scorecard.json")
    )
    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=disable_listening,
        world=world,
        people=world.people,
        scorer=scorecard,
    )

    # The agent stack binds to ctx (its tools reach the ctx-based skills); the patrol
    # drives brain.walkie_agent. Built after ctx so the agents can reach ctx.
    brain = WalkieBrain(ctx, disable_listening=disable_listening)
    ctx.data["brain"] = brain
    if os.getenv("FINAL_START_PERCEPTION", "1").lower() in ("1", "true", "yes"):
        try:
            brain.explore.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[final] perception loop failed to start ({exc}); continuing without it")

    try:
        build_final_task(ctx).run()
    finally:
        try:
            brain.explore.stop()
        except Exception:  # noqa: BLE001
            pass
        print(scorecard.summary())  # attempted/claimed points — NOT referee-awarded
        scorecard.write()
        world.persist()
        walkie_interface.close()


if __name__ == "__main__":
    main()
