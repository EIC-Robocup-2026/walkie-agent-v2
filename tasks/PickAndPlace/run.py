"""Entrypoint for the Pick and Place task (rulebook 5.2).

    uv run python -m tasks.PickAndPlace.run
    DISABLE_LISTENING=1 uv run python -m tasks.PickAndPlace.run   # type instead of speak

PLACEHOLDER: builds the task scaffold and runs it. No people memory is needed
(no people are involved in this test), so ctx.people stays None.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from .subtasks import build_pick_and_place_task
from client import WalkieAIClient


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes"),
    )

    try:
        build_pick_and_place_task(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
