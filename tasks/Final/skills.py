"""Deterministic handlers for the Finals' fixed, position-known problems.

These score the high-value rulebook lines reliably (open the apartment door 600,
move the laundry basket 600, close the dishwasher 300); the open-ended problems are
left to the agent-driven patrol (tasks/Final/subtasks.py). All ``tasks.skills`` /
``tasks.GPSR`` imports are LAZY (inside the functions) so this module stays
offline-importable and pulls the heavy grasp/Open3D stack only when run on the robot.
"""

from __future__ import annotations

import os

from tasks.base import TaskContext

from . import prompts


def _arm_enabled() -> bool:
    """Arm-motion gate (same var the agents' manipulation tools read)."""
    val = os.getenv("FINAL_ARM_CALIBRATED") or os.getenv("WALKIE_ARM_ENABLED", "0")
    return val.lower() in ("1", "true", "yes")


def _resolve_pose(ctx: TaskContext, name: str):
    """(canonical, pose) for a room/location name from the arena map, or (None, None)."""
    world = ctx.world
    canon = world.location(name) or world.room(name)
    if not canon:
        return None, None
    return canon, world.location_pose(canon)


def drive_to(ctx: TaskContext, name: str) -> bool:
    """Navigate to a named place, opening a door only if the route is blocked.

    Returns False (without driving) when the name has no surveyed pose.
    """
    from tasks.skills import go_to_through_door

    canon, pose = _resolve_pose(ctx, name)
    if pose is None:
        return False
    return go_to_through_door(ctx, *pose, ask_even_if_open=ctx.world.is_barrier(canon))


def fulfil_spoken_request(ctx: TaskContext, utterance: str | None = None) -> bool:
    """Take a person's spoken request, repeat it, and execute it via the GPSR pipeline.

    Mirrors the Walkie agent's ``handle_person_request`` tool, for the deterministic
    handlers (e.g. the welcomed guest). Returns True if at least one parsed command did
    not fail.
    """
    from tasks.GPSR.dispatch import execute_plan
    from tasks.GPSR.parse import parse_commands
    from tasks.GPSR.plan import CmdStatus, render_plan_speech

    text = (utterance or "").strip() or ctx.listen()
    if not text:
        return False
    world = ctx.world.vocab
    parsed = parse_commands(ctx.model, text, world)
    if not parsed:
        ctx.say("I am sorry, I did not understand the request.")
        return False
    brain = ctx.data.get("brain")
    manip = _arm_enabled() or os.getenv("GPSR_ENABLE_MANIPULATION", "0").lower() in (
        "1", "true", "yes"
    )
    ok_any = False
    for _src, plan in parsed:
        try:  # repeat the command back (Finals: repeating the command scores)
            ctx.say(render_plan_speech(plan))
        except Exception:  # noqa: BLE001 — never let TTS/render abort execution
            pass
        status = execute_plan(ctx, plan, world, brain, manip_enabled=manip)
        ok_any = ok_any or (status != CmdStatus.FAILED)
    return ok_any


def welcome_guest(ctx: TaskContext) -> bool:
    """Drive to the exit door, open it autonomously, welcome the guest, take their request.

    Scores ``open_apartment_door`` on a successful open and ``solve_problem`` if the
    guest's request is carried out. The guest's position is known (rulebook: no points
    for locating them), so we just drive to the surveyed door pose.
    """
    from tasks.skills import go_to_through_door

    ctx.say(prompts.WELCOME_ANNOUNCE)
    canon, pose = _resolve_pose(ctx, os.getenv("FINAL_EXIT_DOOR", "exit"))
    if pose is None:
        ctx.say(prompts.WELCOME_NO_DOOR)
        return False
    opened = go_to_through_door(ctx, *pose, ask_even_if_open=True)
    if opened:
        ctx.score("open_apartment_door")
    ctx.say(prompts.WELCOME_GREETING)
    if fulfil_spoken_request(ctx):
        ctx.score("solve_problem")
    return opened


def move_laundry_basket(ctx: TaskContext) -> bool:
    """Pick up the laundry basket and carry it to the washing machine (arm-gated)."""
    ctx.say(prompts.LAUNDRY_ANNOUNCE)
    if not _arm_enabled():
        ctx.say(prompts.LAUNDRY_NO_ARM)
        return False
    from tasks.skills import pick_object, place_object

    basket = os.getenv("FINAL_LAUNDRY_BASKET", "laundry basket")
    if not pick_object(ctx, prompts=[basket], approach_preference="side"):
        return False
    if not drive_to(ctx, os.getenv("FINAL_WASHING_MACHINE", "washing_machine")):
        ctx.say(prompts.LAUNDRY_NO_TARGET)
        return False
    if place_object(ctx):
        ctx.score("move_laundry_basket")
        return True
    return False


def _push_close(ctx: TaskContext) -> bool:
    """Best-effort arm push to close a door. Returns whether a motion executed.

    The exact close trajectory needs on-robot calibration; this is a guarded
    placeholder so the step is wired end-to-end. Only reached when the arm is enabled.
    """
    arm = getattr(ctx.walkie, "arm", None)
    if arm is None:
        return False
    for meth in ("do", "execute", "command"):
        fn = getattr(arm, meth, None)
        if callable(fn):
            try:
                fn("close the dishwasher door")
                return True
            except Exception as exc:  # noqa: BLE001
                print(f"[final] dishwasher push failed ({exc})")
                return False
    return False


def close_dishwasher(ctx: TaskContext) -> bool:
    """Drive to the dishwasher and close its door (arm-gated)."""
    ctx.say(prompts.DISHWASHER_ANNOUNCE)
    if not drive_to(ctx, os.getenv("FINAL_DISHWASHER", "dishwasher")):
        ctx.say(prompts.DISHWASHER_NO_TARGET)
        return False
    if not _arm_enabled():
        ctx.say(prompts.DISHWASHER_NO_ARM)
        return False
    if _push_close(ctx):
        ctx.score("close_dishwasher")
        return True
    return False
