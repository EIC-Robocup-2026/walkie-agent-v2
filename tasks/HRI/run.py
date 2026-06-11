import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import (
    load_task_config,
    initialize_llm_model,
    initialize_robot,
)
from .subtasks import build_hri_task
from client import WalkieAIClient


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    # Initialize
    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes"),
    )

    # Flow start here
    try:
        build_hri_task(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
