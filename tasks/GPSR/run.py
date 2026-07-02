"""Entrypoint for the GPSR task (rulebook 5.3).

    uv run python -m tasks.GPSR.run
    DISABLE_LISTENING=1 uv run python -m tasks.GPSR.run   # type instead of speak

Builds the world model (arena nouns) + the Walkie agent stack (WalkieBrain, the
Tier-2 execution fallback) and runs the four-step GPSR envelope (subtasks.py):
go to the instruction point → receive + plan + speak each command → execute →
return. The walkie_graphs perception loop runs in the background
(GPSR_START_PERCEPTION) so find/bring commands can use long-term spatial memory.

A no-robot dry run of just the parser + spoken plan is available offline via
`python -m tasks.GPSR.parse` (needs OPENROUTER_API_KEY, no robot).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import (
    WalkieBrain,
    initialize_llm_model,
    initialize_parser_model,
    initialize_robot,
    load_task_config,
)
from ..scoring import ScoreTracker
from .scoring import GPSR_SHEET
from .subtasks import build_gpsr_task
from client import WalkieAIClient
from walkie_world import WalkieWorld

import open3d as o3d

# Set the global verbosity level to ignore warnings and only show critical errors
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()
    disable_listening = os.getenv("DISABLE_LISTENING", "0").lower() in ("1", "true", "yes")

    # One shared world model for the whole task: arena nouns the parser grounds
    # against (rooms/locations/objects/names/gestures), the spatial scene memory the
    # find/bring commands recall from (CLIP search via embed_text), and the face+attire
    # people store GPSR's person commands (meet/greet/guide/follow, §5.4) reuse.
    world = WalkieWorld(
        embed_text=(lambda q: walkie_ai.image.embed_text(q)),
        enable_people=True,
    )

    scorecard = ScoreTracker(GPSR_SHEET, path=os.getenv("GPSR_SCORECARD_PATH", "gpsr_scorecard.json"))
    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        parser_model=initialize_parser_model(),  # GPSR_PARSER_MODEL; None = share `model`
        disable_listening=disable_listening,
        world=world,
        people=world.people,  # back-compat: ctx.people is the world's people store
        scorer=scorecard,  # live tally of attempted/claimed points (ctx.score)
    )

    # The agent stack is the Tier-2 execution fallback (see subtasks.ExecuteCommands);
    # it binds to the SAME ctx (shared world/people/scorer/blackboard) so its tools act
    # on the same robot state. Built after ctx so the agents can reach the ctx-based skills.
    brain = WalkieBrain(ctx, disable_listening=disable_listening)
    ctx.data["brain"] = brain
    if os.getenv("GPSR_START_PERCEPTION", "1").lower() in ("1", "true", "yes"):
        try:
            brain.explore.start()
        except Exception as exc:
            print(f"[gpsr] perception loop failed to start ({exc}); continuing without it")

    # while True:
    #     pass

    try:
        build_gpsr_task(ctx).run()
    finally:
        print(scorecard.summary())  # attempted/claimed points — NOT referee-awarded
        scorecard.write()
        walkie_interface.close()


if __name__ == "__main__":
    main()
