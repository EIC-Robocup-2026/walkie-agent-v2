"""GPSR subtasks + the build_gpsr_task factory (rulebook 5.3).

The fixed outer envelope (rulebook procedure): go to the instruction point,
receive the operator's command(s), **plan + demonstrate** each, execute, then
return. The reactive/arbitrary part is inside ReceiveAndPlanCommands /
ExecuteCommands, per docs/GPSR_DESIGN.md.

Phase 0 (this commit) lands the draw-independent 540: parse each command into a
typed `Plan` (parse.py) and **speak the plan** ("demonstrate a plan has been
generated", 3×100). Execution is the Tier-2 agent fallback for now; Phase 1 adds
the deterministic Tier-1 skill dispatch.

Blackboard layout (ctx.data):
    brain:    WalkieBrain          # the agent stack (Tier-2 fallback), set by run.py
    world:    WorldModel           # arena nouns, set by run.py
    commands: list[Command]        # parsed + planned operator commands
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto

from langchain.messages import HumanMessage

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .parse import parse_commands
from .plan import Plan, render_plan_speech


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _one_by_one() -> bool:
    return os.getenv("GPSR_ISSUE_MODE", "consecutive").strip().lower() == "one_by_one"


def _speak_plan_enabled() -> bool:
    return os.getenv("GPSR_SPEAK_PLAN", "1").lower() in ("1", "true", "yes")


class CmdStatus(Enum):
    RECEIVED = auto()
    PLANNED = auto()
    IN_PROGRESS = auto()
    DONE = auto()
    PARTIAL = auto()
    FAILED = auto()


@dataclass
class Command:
    """One operator command + its parsed plan + progress (docs/GPSR_DESIGN.md §6)."""

    id: int
    utterance: str
    plan: Plan
    status: CmdStatus = CmdStatus.RECEIVED
    result_note: str | None = None


class GoToInstructionPoint(SubTask):
    """Navigate to the instruction point when the arena door opens."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
        return StepResult.DONE if ctx.goto(x, y, h) else StepResult.RETRY


class ReceiveAndPlanCommands(SubTask):
    """Take the operator's command(s), parse each into a typed plan, speak it.

    Speaking the plan is what scores "demonstrate a plan has been generated"
    (3×100) and doubles as the operator's confirmation. The robot *requests all
    three at once* (§5.5) to keep the interleave bonus reachable and save
    round-trips; in one-by-one mode ExecuteCommands returns between commands.
    Re-asks only on an empty parse (§5.2 — rephrasings cost −30 each).
    """

    max_retries = 2  # the rulebook allows asking the operator to repeat

    def run(self, ctx: TaskContext) -> StepResult:
        world = ctx.data.get("world")
        if world is None:
            print("[gpsr] no world model on ctx.data['world'] — cannot plan")
            ctx.say(prompts.PLAN_NOT_UNDERSTOOD)
            return StepResult.ABORT
        ctx.say(prompts.GREET_OPERATOR)
        answer = ctx.ask(prompts.ASK_FOR_COMMANDS)
        if not answer:
            ctx.say(prompts.ASK_REPEAT)
            return StepResult.RETRY

        parsed = parse_commands(ctx.model, answer, world)
        if not parsed:
            ctx.say(prompts.ASK_REPEAT)
            return StepResult.RETRY

        commands: list[Command] = []
        for i, (text, plan) in enumerate(parsed, 1):
            cmd = Command(id=i, utterance=text, plan=plan)
            if plan:
                cmd.status = CmdStatus.PLANNED
                if _speak_plan_enabled():
                    ctx.say(render_plan_speech(plan, preamble=prompts.PLAN_PREAMBLE.format(n=i)))
                if not plan.is_complete:
                    print(f"[gpsr] command {i} plan has ungrounded steps: {plan.source!r}")
            else:
                ctx.say(prompts.PLAN_NOT_UNDERSTOOD)
            commands.append(cmd)
            print(f"[gpsr] command {i}: {text!r} -> {[s.primitive.value for s in plan.steps]}")

        ctx.data["commands"] = commands
        ctx.say(prompts.CONFIRM_RECEIVED)
        return StepResult.DONE


class ExecuteCommands(SubTask):
    """Execute each planned command.

    Phase 0: Tier-2 only — hand each command to the agent stack (planning +
    execution belong to the agent for the long tail). Phase 1 replaces this with
    deterministic Tier-1 dispatch of each PlanStep to a skill, agent-fallback on
    miss. Degrades to a spoken note if the brain was not wired.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        brain = ctx.data.get("brain")
        commands: list[Command] = ctx.data.get("commands", [])
        if brain is None:
            print("[gpsr] no agent stack on ctx.data['brain'] — cannot execute")
            ctx.say("My execution system is not connected, so I cannot carry out the commands.")
            return StepResult.DONE
        for cmd in commands:
            cmd.status = CmdStatus.IN_PROGRESS
            ctx.say(prompts.COMMAND_ANNOUNCE.format(n=cmd.id, command=cmd.utterance))
            try:
                brain.walkie_agent.invoke(
                    {"messages": [HumanMessage(content=cmd.utterance)]},
                    config={"configurable": {"thread_id": "gpsr"}},
                )
                cmd.status = CmdStatus.DONE
            except Exception as exc:
                print(f"[gpsr] command {cmd.id} failed ({exc})")
                cmd.status = CmdStatus.FAILED
            if _one_by_one() and cmd.id < len(commands):
                x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
                ctx.goto(x, y, h)  # return to operator for the next command
        return StepResult.DONE


class ReturnToInstructionPoint(SubTask):
    """Go back to the instruction point after all commands (rulebook procedure 3)."""

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.RETURN_ANNOUNCE)
        x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
        ctx.goto(x, y, h)
        ctx.say(prompts.ALL_DONE)
        return StepResult.DONE


def build_gpsr_task(ctx: TaskContext) -> Task:
    """Construct the GPSR task. Pure: the agent stack + world are read from ctx.data."""
    return Task(
        "GPSR",
        [
            GoToInstructionPoint(),
            ReceiveAndPlanCommands(),
            ExecuteCommands(),
            ReturnToInstructionPoint(),
        ],
        ctx,
    )
