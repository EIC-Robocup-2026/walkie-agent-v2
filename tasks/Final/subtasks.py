"""Finals task steps (rulebook chapter 6): hybrid scaffold + agent.

The scaffold runs the fixed, position-known, high-value problems deterministically
(welcome the guest through the door, move the laundry basket, close the dishwasher),
then patrols the rooms and hands each to the Walkie agent to find + solve one
open-ended problem (trash, a misplaced object, a person's request) with its tools.
The whole run is bounded by ``FINAL_TIME_BUDGET_SEC`` (the 10-minute Finals cap).

All heavy / robot imports (``tasks.skills``, the agent invoke) happen inside ``run``,
so ``build_final_task`` and this module import on a GPU-less box.
"""

from __future__ import annotations

import os
import time

from langchain_core.messages import HumanMessage

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts, skills


def _budget_sec() -> float:
    return float(os.getenv("FINAL_TIME_BUDGET_SEC", "600"))


def _past_deadline(ctx: TaskContext) -> bool:
    deadline = ctx.data.get("final_deadline")
    return deadline is not None and time.monotonic() > deadline


class EnterArena(SubTask):
    """Wait for the arena door to open, enter, and start the run clock."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        from tasks.skills import request_open_door

        ctx.say(prompts.ENTER_ANNOUNCE)
        # The run clock starts when we begin (the door wait is part of the test).
        ctx.data["final_deadline"] = time.monotonic() + _budget_sec()
        ctx.data.setdefault("solved_categories", [])
        try:
            request_open_door(ctx, prompt=prompts.ENTER_DOOR_PROMPT)
        except Exception as exc:  # noqa: BLE001 — proceed even if the door check fails
            print(f"[final] arena-door wait failed ({exc}); entering anyway")
        start = os.getenv("FINAL_START_POSE", "current").strip().lower()
        if start and start != "current":
            try:
                x, y, h = (float(v) for v in start.split(","))
                ctx.goto(x, y, h)
            except ValueError:
                print(f"[final] bad FINAL_START_POSE {start!r}; staying put")
        return StepResult.DONE


class WelcomeGuest(SubTask):
    """Open the apartment door and welcome the waiting guest (fixed 600-pt problem)."""

    def run(self, ctx: TaskContext) -> StepResult:
        if _past_deadline(ctx):
            return StepResult.DONE
        skills.welcome_guest(ctx)
        return StepResult.DONE


class MoveLaundryBasket(SubTask):
    """Carry the laundry basket to the washing machine (fixed 600-pt problem)."""

    def run(self, ctx: TaskContext) -> StepResult:
        if _past_deadline(ctx):
            return StepResult.DONE
        skills.move_laundry_basket(ctx)
        return StepResult.DONE


class CloseDishwasher(SubTask):
    """Close the dishwasher door (fixed 300-pt problem)."""

    def run(self, ctx: TaskContext) -> StepResult:
        if _past_deadline(ctx):
            return StepResult.DONE
        skills.close_dishwasher(ctx)
        return StepResult.DONE


class PatrolAndSolve(SubTask):
    """Patrol the rooms; hand each to the agent to find + solve one open-ended problem.

    Cycles the configured rooms until the time budget runs out or ``FINAL_MAX_PATROL_
    PROBLEMS`` turns are taken. Each turn drives to the room, then invokes the Walkie
    orchestrator with the per-room mission prompt; the agent uses its tools
    (find_person_raising_hand / handle_person_request, get_default_location,
    pick_up_object / place_object_down) to do the work and SPEAK each problem it finds.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        brain = ctx.data.get("brain")
        if brain is None:
            print("[final] no brain on ctx; skipping patrol")
            return StepResult.DONE
        rooms = [r.strip() for r in os.getenv("FINAL_PATROL_ROOMS", "").split(",") if r.strip()]
        if not rooms:
            print("[final] FINAL_PATROL_ROOMS empty; skipping patrol")
            return StepResult.DONE
        max_turns = max(1, int(os.getenv("FINAL_MAX_PATROL_PROBLEMS", "8")))
        solved = ctx.data.setdefault("solved_categories", [])

        turn = 0
        while turn < max_turns and not _past_deadline(ctx):
            room = rooms[turn % len(rooms)]
            skills.drive_to(ctx, room)
            mission = prompts.FINAL_PATROL_MISSION.format(
                room=room.replace("_", " "),
                solved=", ".join(solved) if solved else "none yet",
            )
            try:
                brain.walkie_agent.invoke(
                    {"messages": [HumanMessage(content=mission)]},
                    config={"configurable": {"thread_id": "final"}},
                )
            except Exception as exc:  # noqa: BLE001 — one bad room must not end the patrol
                print(f"[final] patrol turn in {room!r} failed ({exc}); continuing")
            turn += 1
        return StepResult.DONE


class Finish(SubTask):
    """Announce completion."""

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.FINISH_ANNOUNCE)
        return StepResult.DONE


def build_final_task(ctx: TaskContext) -> Task:
    """Assemble the Finals task. Pure: no hardware touched at build time."""
    return Task(
        "Final",
        [
            EnterArena(),
            WelcomeGuest(),
            MoveLaundryBasket(),
            CloseDishwasher(),
            PatrolAndSolve(),
            Finish(),
        ],
        ctx,
    )
