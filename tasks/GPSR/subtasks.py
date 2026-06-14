"""GPSR subtasks + the build_gpsr_task factory (rulebook 5.3).

PLACEHOLDER scaffold. Unlike the other tasks, GPSR is not a fixed procedure:
the operator issues three arbitrary commands and the robot must plan and execute
each. That is precisely what the existing Walkie agent stack
(tasks.common.WalkieBrain -> walkie_agent + actuator/vision/database sub-agents)
already does, so ExecuteCommands hands each command straight to it rather than
faking a linear sequence.

run.py builds the WalkieBrain and stashes it on ctx.data["brain"]; the subtasks
read it from there. Flow (rulebook procedure): go to the instruction point,
receive the command(s), execute them (consecutively or one-by-one, returning to
the instruction point between commands when one-by-one), then return.

Blackboard layout (ctx.data):
    brain:    WalkieBrain        # the agent stack, set by run.py
    commands: list[str]          # from ReceiveCommands
"""

from __future__ import annotations

import os

from langchain.messages import HumanMessage

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts


def _pose(env_key: str, default: str = "0.0,0.0,0.0") -> tuple[float, float, float]:
    parts = [p.strip() for p in os.getenv(env_key, default).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{env_key}: expected 'x,y,heading_rad', got {parts!r}")
    x, y, h = (float(p) for p in parts)
    return x, y, h


def _one_by_one() -> bool:
    return os.getenv("GPSR_ISSUE_MODE", "consecutive").strip().lower() == "one_by_one"


class GoToInstructionPoint(SubTask):
    """Navigate to the instruction point when the arena door opens."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
        return StepResult.DONE if ctx.goto(x, y, h) else StepResult.RETRY


class ReceiveCommands(SubTask):
    """Greet the operator and capture up to N commands, split into a list.

    Real glue today: ask -> STT -> LLM split. In one-by-one mode only the first
    command is taken here; ExecuteCommands loops back for the rest.
    """

    max_retries = 2  # the rulebook allows asking the operator to repeat

    def run(self, ctx: TaskContext) -> StepResult:
        ctx.say(prompts.GREET_OPERATOR)
        answer = ctx.ask(prompts.ASK_FOR_COMMANDS)
        if not answer:
            ctx.say(prompts.ASK_REPEAT)
            return StepResult.RETRY
        parsed = ctx.extract(prompts.CommandList, prompts.SPLIT_COMMANDS_INSTRUCTIONS, answer)
        commands = parsed.commands if parsed else []
        if not commands:
            ctx.say(prompts.ASK_REPEAT)
            return StepResult.RETRY
        max_n = int(os.getenv("GPSR_MAX_COMMANDS", "3"))
        ctx.data["commands"] = commands[:max_n]
        ctx.say(prompts.CONFIRM_RECEIVED)
        print(f"[gpsr] received commands: {ctx.data['commands']}")
        return StepResult.DONE


class ExecuteCommands(SubTask):
    """Hand each command to the Walkie agent stack; return between in one-by-one.

    This is the honest GPSR core: planning + execution belong to the agent, not
    to a hardcoded SubTask. Degrades to a spoken note if the brain was not wired.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        brain = ctx.data.get("brain")
        commands: list[str] = ctx.data.get("commands", [])
        if brain is None:
            print("[gpsr] TODO no agent stack on ctx.data['brain'] — cannot execute")
            ctx.say("My planning system is not connected, so I cannot execute the commands.")
            return StepResult.DONE
        for i, command in enumerate(commands, 1):
            ctx.say(prompts.COMMAND_ANNOUNCE.format(n=i, command=command))
            try:
                brain.walkie_agent.invoke(
                    {"messages": [HumanMessage(content=command)]},
                    config={"configurable": {"thread_id": "gpsr"}},
                )
            except Exception as exc:
                print(f"[gpsr] command {i} failed ({exc})")
            if _one_by_one() and i < len(commands):
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
    """Construct the GPSR task. Pure: the agent stack is read from ctx.data at run time."""
    return Task(
        "GPSR",
        [
            GoToInstructionPoint(),
            ReceiveCommands(),
            ExecuteCommands(),
            ReturnToInstructionPoint(),
        ],
        ctx,
    )
