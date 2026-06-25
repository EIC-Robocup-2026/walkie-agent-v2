"""Entrypoint for the Restaurant task (rulebook 5.5).

    uv run python -m tasks.Restaurant.run                       # full serve flow
    DISABLE_LISTENING=1 uv run python -m tasks.Restaurant.run   # type instead of speak
    RESTAURANT_SKIP_PREFLIGHT=1 ...                             # bypass the pre-flight gate

Step-by-step on-robot bring-up — pick an isolated slice with RESTAURANT_SLICE
(like GPSR's runbook; validate each subtask before running the whole flow):

    RESTAURANT_SLICE=surfaces ...   # read-only: list detected surfaces (no motion)
    RESTAURANT_SLICE=pick ...       # pick only (grasp skill, no placement)
    RESTAURANT_SLICE=place ...      # surface scan -> full pick -> place
    RESTAURANT_SLICE=phase0 ...     # scan -> detect waving customer -> approach
    RESTAURANT_SLICE=full ...       # whole serve loop (default; RESTAURANT_PHASE0=1 alias for phase0)

Phase 0 (scan -> detect waving customer -> approach to stand-off) is implemented;
order-taking + relay are real; pick/serve are Phase-2 stubs. Person handling is
gesture detection, not face re-ID, so no PeopleStore is wired (ctx.people=None).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from ..scoring import ScoreTracker
from .scoring import RESTAURANT_SHEET
from .subtasks import (
    build_phase0_slice,
    build_pick_demo,
    build_place_demo,
    build_restaurant_task,
    build_surface_demo,
)
from client import WalkieAIClient


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# Isolated slices for step-by-step on-robot bring-up, selected by RESTAURANT_SLICE
# (RESTAURANT_PHASE0=1 stays as a back-compat alias for "phase0"). Order = rough
# bring-up order: prove perception, then pick, then place, then HRI, then the lot.
_SLICES = {
    "surfaces": build_surface_demo,   # read-only surface detection (no motion)
    "pick": build_pick_demo,          # grasp skill only
    "place": build_place_demo,        # surface scan -> pick -> place
    "phase0": build_phase0_slice,     # scan -> detect waving customer -> approach
    "full": build_restaurant_task,    # whole serve loop
}


def _select_build():
    """Resolve RESTAURANT_SLICE (falling back to RESTAURANT_PHASE0 / full)."""
    slice_name = os.getenv("RESTAURANT_SLICE", "").strip().lower()
    if not slice_name:
        slice_name = "phase0" if _truthy("RESTAURANT_PHASE0") else "full"
    build = _SLICES.get(slice_name)
    if build is None:
        valid = ", ".join(_SLICES)
        raise SystemExit(f"[restaurant] unknown RESTAURANT_SLICE={slice_name!r}; pick one of: {valid}")
    return slice_name, build


def preflight(walkie, walkie_ai) -> bool:
    """Quick checks before a robot run — fail fast instead of mid-task.

    Returns True if the HARD prerequisites pass (LLM, AI server, localization);
    config/arm lines are informational. Bypass with RESTAURANT_SKIP_PREFLIGHT=1.
    """
    hard_ok = True

    def hard(label: str, passed: bool, hint: str = "") -> None:
        nonlocal hard_ok
        print(f"  [{'OK' if passed else 'XX'}] {label}" + ("" if passed or not hint else f" — {hint}"))
        hard_ok = hard_ok and passed

    print("[restaurant] pre-flight:")

    # 1. LLM (order parsing) — skipped when using a local model.
    hard("LLM configured",
         _truthy("LLM_USE_LOCAL") or bool(os.getenv("OPENROUTER_API_KEY")),
         "set OPENROUTER_API_KEY in .env (orders won't parse without it)")

    # 2. walkie-ai-server reachable (pose / detection / caption / STT / TTS all on
    #    one server; STT exposes a cheap providers ping in the unified client).
    try:
        walkie_ai.stt.available_providers()
        server_ok, why = True, ""
    except Exception as exc:
        server_ok, why = False, f"{os.getenv('WALKIE_AI_BASE_URL', 'http://localhost:5000')} ({exc})"
    hard("walkie-ai-server reachable", server_ok, why)

    # 3. Robot localizing (Nav2/SLAM fix) — go_to + every lift depend on it.
    try:
        pose = walkie.status.get_position()
    except Exception:
        pose = None
    hard("robot localizing (odom/SLAM fix)", pose is not None, "bring up Nav2 / SLAM first")

    # --- informational (won't block) ---
    print(f"  [i ] bar start pose = {os.getenv('RESTAURANT_KITCHEN_BAR_POSE', 'current')} "
          "('current' = stay where the robot is; or 'x,y,heading_rad' for a fixed map pose)")
    print(f"  [i ] arm: " + ("CALIBRATED — pick/serve WILL move" if _truthy("RESTAURANT_ARM_CALIBRATED")
                             else "uncalibrated — pick/serve log only, no motion"))
    slice_name, _ = _select_build()
    detail = ("batched serve" if _truthy("RESTAURANT_BATCH") else "serial serve") if slice_name == "full" else slice_name
    print(f"  [i ] slice: {slice_name} ({detail})")
    return hard_ok


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    if not preflight(walkie_interface, walkie_ai) and not _truthy("RESTAURANT_SKIP_PREFLIGHT"):
        print("[restaurant] pre-flight FAILED — fix the [XX] items above, or set "
              "RESTAURANT_SKIP_PREFLIGHT=1 to run anyway.", file=sys.stderr)
        walkie_interface.close()
        return

    scorecard = ScoreTracker(RESTAURANT_SHEET, path=os.getenv("RESTAURANT_SCORECARD_PATH", "restaurant_scorecard.json"))
    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=_truthy("DISABLE_LISTENING"),
        scorer=scorecard,  # live tally of attempted/claimed points (ctx.score)
    )

    slice_name, build = _select_build()
    print(f"[restaurant] running slice: {slice_name}")

    try:
        build(ctx).run()
    finally:
        print(scorecard.summary())  # attempted/claimed points — NOT referee-awarded
        scorecard.write()
        walkie_interface.close()


if __name__ == "__main__":
    main()
