"""Entrypoint for the Pick and Place task (rulebook 5.2).

    uv run python -m tasks.PickAndPlace.run                       # whole flow
    DISABLE_LISTENING=1 uv run python -m tasks.PickAndPlace.run   # type instead of speak
    PNP_SKIP_PREFLIGHT=1 ...                                      # bypass the pre-flight gate

Step-by-step on-robot bring-up — pick an isolated slice with PNP_SLICE (validate
each subtask before running the whole flow, like the Restaurant runner):

    PNP_SLICE=nav ...        # waypoint tour: visit every PNP_*_POSE, announce arrival
    PNP_SLICE=perceive ...   # drive to the table, perceive + announce each object
    PNP_SLICE=sort ...       # perceive + sort + indicate the correct placement (no arm)
    PNP_SLICE=breakfast ...  # recognize breakfast items at their sources + announce plan
    PNP_SLICE=full ...       # whole pick-and-place flow (default)

The arm is a separate skill under development, so every grasp/place is gated by
PNP_ARM_CALIBRATED (default 0): the flow navigates, recognizes objects, and
*indicates* the correct placement (banking the non-arm score budget) without
moving the arm. Flip PNP_ARM_CALIBRATED=1 once the arm skill lands.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from ..scoring import ScoreTracker
from .scoring import PNP_SHEET
from .subtasks import (
    build_breakfast_slice,
    build_nav_slice,
    build_perceive_slice,
    build_pick_and_place_task,
    build_sort_slice,
)
from client import WalkieAIClient


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# Isolated slices for step-by-step on-robot bring-up, selected by PNP_SLICE.
# Order = rough bring-up order: prove the waypoints, then perception, then the
# perception+sort+indicate scoring path, then the whole flow.
_SLICES = {
    "nav": build_nav_slice,            # waypoint tour (no perception/arm)
    "perceive": build_perceive_slice,  # perceive + announce each recognized object
    "sort": build_sort_slice,          # perceive + sort + indicate placement (arm-gated)
    "breakfast": build_breakfast_slice,  # recognize breakfast items + announce layout
    "full": build_pick_and_place_task,  # whole pick-and-place flow
}


def _select_build():
    """Resolve PNP_SLICE (falling back to the whole flow)."""
    slice_name = os.getenv("PNP_SLICE", "").strip().lower() or "full"
    build = _SLICES.get(slice_name)
    if build is None:
        valid = ", ".join(_SLICES)
        raise SystemExit(f"[pnp] unknown PNP_SLICE={slice_name!r}; pick one of: {valid}")
    return slice_name, build


def preflight(walkie, walkie_ai) -> bool:
    """Quick checks before a robot run — fail fast instead of mid-task.

    Returns True if the HARD prerequisites pass (LLM, AI server, localization);
    arm/config lines are informational. Bypass with PNP_SKIP_PREFLIGHT=1.
    """
    hard_ok = True

    def hard(label: str, passed: bool, hint: str = "") -> None:
        nonlocal hard_ok
        print(f"  [{'OK' if passed else 'XX'}] {label}" + ("" if passed or not hint else f" — {hint}"))
        hard_ok = hard_ok and passed

    print("[pnp] pre-flight:")

    # 1. LLM (destination sorting) — skipped when using a local model.
    hard("LLM configured",
         _truthy("LLM_USE_LOCAL") or bool(os.getenv("OPENROUTER_API_KEY")),
         "set OPENROUTER_API_KEY in .env (object sorting won't run without it)")

    # 2. walkie-ai-server reachable (detection / caption / embed / STT / TTS).
    try:
        walkie_ai.stt.available_providers()
        server_ok, why = True, ""
    except Exception as exc:
        server_ok, why = False, f"{os.getenv('WALKIE_AI_BASE_URL', 'http://localhost:5000')} ({exc})"
    hard("walkie-ai-server reachable", server_ok, why)

    # 3. Robot localizing (Nav2/SLAM fix) — go_to + every 3D lift depend on it.
    try:
        pose = walkie.status.get_position()
    except Exception:
        pose = None
    hard("robot localizing (odom/SLAM fix)", pose is not None, "bring up Nav2 / SLAM first")

    # --- informational (won't block) ---
    print(f"  [i ] arm: " + ("CALIBRATED — pick/place WILL move the arm" if _truthy("PNP_ARM_CALIBRATED")
                             else "gated — recognize + indicate placement only, no arm motion"))
    print(f"  [i ] trash category = {os.getenv('PNP_TRASH_CATEGORY', '') or '(unset — set PNP_TRASH_CATEGORY)'}")
    slice_name, _ = _select_build()
    print(f"  [i ] slice: {slice_name}")
    return hard_ok


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    if not preflight(walkie_interface, walkie_ai) and not _truthy("PNP_SKIP_PREFLIGHT"):
        print("[pnp] pre-flight FAILED — fix the [XX] items above, or set "
              "PNP_SKIP_PREFLIGHT=1 to run anyway.", file=sys.stderr)
        walkie_interface.close()
        return

    scorecard = ScoreTracker(PNP_SHEET, path=os.getenv("PNP_SCORECARD_PATH", "pnp_scorecard.json"))
    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=_truthy("DISABLE_LISTENING"),
        scorer=scorecard,  # live tally of attempted/claimed points (ctx.score)
    )

    slice_name, build = _select_build()
    print(f"[pnp] running slice: {slice_name}")

    try:
        build(ctx).run()
    finally:
        print(scorecard.summary())  # attempted/claimed points — NOT referee-awarded
        scorecard.write()
        walkie_interface.close()


if __name__ == "__main__":
    main()
