"""Entrypoint for the GPSR task (rulebook 5.3).

    uv run python -m tasks.GPSR.run
    DISABLE_LISTENING=1 uv run python -m tasks.GPSR.run   # type instead of speak

PLACEHOLDER: builds the full Walkie agent stack (WalkieBrain) and hands it to the
task on ctx.data["brain"]; ExecuteCommands delegates each operator command to it.
The walkie_graphs perception loop is started in the background (GPSR_START_PERCEPTION)
so "find/bring object" commands can use long-term spatial memory.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import WalkieBrain, initialize_llm_model, initialize_robot, load_task_config
from .subtasks import build_gpsr_task
from client import WalkieAIClient


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    disable_listening = os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes")

    # The agent stack IS the GPSR planner/executor (see subtasks.ExecuteCommands).
    brain = WalkieBrain(walkie_ai, walkie_interface, model, disable_listening=disable_listening)
    if os.getenv("GPSR_START_PERCEPTION", "1").lower() in ("1", "true", "yes"):
        try:
            brain.graphs.start()
        except Exception as exc:
            print(f"[gpsr] perception loop failed to start ({exc}); continuing without it")

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=disable_listening,
    )
    ctx.data["brain"] = brain

    try:
        build_gpsr_task(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
