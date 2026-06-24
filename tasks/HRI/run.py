"""Entrypoint for the HRI / Receptionist task (rulebook 5.1).

    uv run python -m tasks.HRI.run                       # full Receptionist flow
    DISABLE_LISTENING=1 uv run python -m tasks.HRI.run   # type instead of speak
    HRI_SKIP_PREFLIGHT=1 ...                             # bypass the pre-flight gate

Step-by-step on-robot bring-up — pick an isolated slice with HRI_SLICE (like the
Restaurant / PickAndPlace runners; validate each piece before the whole flow):

    HRI_SLICE=seats ...        # loop seat + people detection (tune seat detection)
    HRI_SLICE=greet ...        # greet + learn one guest at the door
    HRI_SLICE=follow_host ...  # remember the host, then follow + drop the bag
    HRI_SLICE=full ...         # whole 12-step Receptionist flow (default)

Back-compat: HRI_TEST_FOLLOW_HOST=1 / HRI_TEST_SCAN_SEATS=1 still select the
follow_host / seats slices. The bag handover/drop (the arm step) stays gated by
HRI_ENABLE_BAG.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ..base import TaskContext
from ..common import initialize_llm_model, initialize_robot, load_task_config
from .subtasks import (
    build_follow_host_slice,
    build_greet_slice,
    build_hri_task,
    build_seats_slice,
    prepare_run,
)
from client import WalkieAIClient
from perception import PeopleStore


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# Isolated slices for step-by-step on-robot bring-up, selected by HRI_SLICE.
# Order = rough bring-up order: perception, then one guest, then follow, then all.
_SLICES = {
    "seats": build_seats_slice,             # seat + people detection tuning
    "greet": build_greet_slice,             # greet + learn one guest
    "follow_host": build_follow_host_slice,  # follow-host re-ID + bag drop
    "full": build_hri_task,                 # whole Receptionist flow
}


def _select_build():
    """Resolve HRI_SLICE (falling back to the HRI_TEST_* aliases, then full)."""
    slice_name = os.getenv("HRI_SLICE", "").strip().lower()
    if not slice_name:
        if _truthy("HRI_TEST_FOLLOW_HOST"):
            slice_name = "follow_host"
        elif _truthy("HRI_TEST_SCAN_SEATS"):
            slice_name = "seats"
        else:
            slice_name = "full"
    build = _SLICES.get(slice_name)
    if build is None:
        valid = ", ".join(_SLICES)
        raise SystemExit(f"[HRI] unknown HRI_SLICE={slice_name!r}; pick one of: {valid}")
    return slice_name, build


def preflight(walkie, walkie_ai) -> bool:
    """Quick checks before a robot run — fail fast instead of mid-task.

    Returns True if the HARD prerequisites pass (LLM, AI server, localization);
    bag/arm + slice lines are informational. Bypass with HRI_SKIP_PREFLIGHT=1.
    """
    hard_ok = True

    def hard(label: str, passed: bool, hint: str = "") -> None:
        nonlocal hard_ok
        print(f"  [{'OK' if passed else 'XX'}] {label}" + ("" if passed or not hint else f" — {hint}"))
        hard_ok = hard_ok and passed

    print("[HRI] pre-flight:")

    # 1. LLM (seat picks + guest-intro speeches) — skipped when using a local model.
    hard("LLM configured",
         _truthy("LLM_USE_LOCAL") or bool(os.getenv("OPENROUTER_API_KEY")),
         "set OPENROUTER_API_KEY in .env (intros/seat picks won't run without it)")

    # 2. walkie-ai-server reachable (STT/TTS/detection/pose/face/appearance).
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
    print("  [i ] bag (arm): " + ("ENABLED — ReceiveBag will move the arm" if _truthy("HRI_ENABLE_BAG")
                                  else "gated — bag handover/drop skipped (HRI_ENABLE_BAG=0)"))
    slice_name, _ = _select_build()
    print(f"  [i ] slice: {slice_name}")
    return hard_ok


def main() -> None:
    load_dotenv()
    load_task_config(Path(__file__).resolve().parent)

    walkie_interface = initialize_robot()
    model = initialize_llm_model()
    walkie_ai = WalkieAIClient()

    if not preflight(walkie_interface, walkie_ai) and not _truthy("HRI_SKIP_PREFLIGHT"):
        print("[HRI] pre-flight FAILED — fix the [XX] items above, or set "
              "HRI_SKIP_PREFLIGHT=1 to run anyway.", file=sys.stderr)
        walkie_interface.close()
        return

    ctx = TaskContext(
        walkie=walkie_interface,
        walkieAI=walkie_ai,
        model=model,
        disable_listening=_truthy("DISABLE_LISTENING"),
        people=PeopleStore.from_env(),
    )

    prepare_run(ctx)  # fresh identities + positions for this run (all slices)
    slice_name, build = _select_build()
    print(f"[HRI] running slice: {slice_name}")

    try:
        build(ctx).run()
    finally:
        walkie_interface.close()


if __name__ == "__main__":
    main()
