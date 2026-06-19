"""Entrypoint for the Restaurant task (rulebook 5.5).

    uv run python -m tasks.Restaurant.run
    DISABLE_LISTENING=1 uv run python -m tasks.Restaurant.run   # type instead of speak

PLACEHOLDER: builds the task scaffold and runs it. Person handling here is
gesture detection, not face re-ID, so no PeopleStore is wired (ctx.people=None).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import (
    initialize_graphs,
    initialize_llm_model,
    initialize_robot,
    load_task_config,
)
from .subtasks import build_restaurant_task
from client import WalkieAIClient


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    graphs = initialize_graphs(model, walkie_ai, walkie_interface)

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        graphs=graphs,
        disable_listening=os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes"),
    )

    try:
        build_restaurant_task(ctx).run()
    finally:
        if graphs is not None:
            graphs.stop()
        walkie_interface.close()


if __name__ == "__main__":
    main()
