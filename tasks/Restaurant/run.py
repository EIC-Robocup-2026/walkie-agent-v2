"""Entrypoint for the Restaurant task (rulebook 5.5).

    uv run python -m tasks.Restaurant.run
    DISABLE_LISTENING=1 uv run python -m tasks.Restaurant.run   # type instead of speak
    RESTAURANT_PHASE0=1 uv run python -m tasks.Restaurant.run   # just scan -> approach

Phase 0 (scan -> detect waving customer -> approach to stand-off) is implemented;
order-taking + relay are real; pick/serve are Phase-2 stubs. Person handling is
gesture detection, not face re-ID, so no PeopleStore is wired (ctx.people=None).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from .subtasks import build_phase0_slice, build_restaurant_task
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

    phase0 = os.getenv("RESTAURANT_PHASE0", "0").lower() in ("1", "true", "yes")
    build = build_phase0_slice if phase0 else build_restaurant_task

    try:
        build(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
