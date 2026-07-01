"""Entrypoint for the Inspection task (RoboCup@Home robot inspection).

    uv run python -m tasks.Inspection.run
    DISABLE_LISTENING=1 uv run python -m tasks.Inspection.run   # type instead of speak
    INSPECTION_AGENT_MODE=1 ...                                 # full agent for referee Q&A

Runs the deterministic inspection scaffold (subtasks.py): notice the entry door open
and drive through → stop for a person who steps in front → visit the inspection points
→ declare external devices → loudness test → move to the exit on the referee's signal.

Points of interest live in tasks/Inspection/config.toml (the referee publishes them
before the run). By DEFAULT this is a light process: no heavy agent stack, no
perception loop, no scorer (an inspection is functional / pass-fail, not point-scored)
— just the robot, the LLM for a couple of small calls, and the AI server for STT/TTS +
depth/pose. ``INSPECTION_AGENT_MODE=1`` additionally builds the full WalkieBrain so the
referee Q&A is handled by the Walkie orchestrator agent; navigation and the scripted
demonstrations stay deterministic either way.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from .subtasks import build_inspection_task
from client import WalkieAIClient


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    disable_listening = _truthy("DISABLE_LISTENING")
    agent_mode = _truthy("INSPECTION_AGENT_MODE")

    # Default (recommended for the inspection): a light objects-only world is enough
    # — the scaffold never queries scene/people memory. Agent mode needs the full
    # brain, so wire an embed_text-backed world + RobotContext (perception middleware)
    # exactly as the Final task does.
    world = None
    if agent_mode:
        from agents.core.robot_context import RobotContext
        from walkie_world import WalkieWorld

        rc = RobotContext.init(perception_path=os.getenv("PERCEPTION_PATH", "perception.json"))
        rc.stage = "ready"
        world = WalkieWorld(
            embed_text=(lambda q: walkie_ai.image.embed_text(q)),
            enable_people=False,
        )

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=disable_listening,
        world=world,
    )

    if agent_mode:
        from ..common import WalkieBrain

        brain = WalkieBrain(ctx, disable_listening=disable_listening)
        ctx.data["brain"] = brain
        print("[inspection] agent mode ON — referee Q&A handled by the Walkie agent")

    try:
        build_inspection_task(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
