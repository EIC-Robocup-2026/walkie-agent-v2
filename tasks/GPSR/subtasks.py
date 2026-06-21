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
from dataclasses import dataclass

from tasks.base import StepResult, SubTask, Task, TaskContext

from . import prompts
from .dispatch import execute_interleaved, execute_plan
from .parse import parse_commands
from .plan import CmdStatus, Plan, render_plan_speech
from .schedule import interleave


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


def _manip_enabled() -> bool:
    return os.getenv("GPSR_ENABLE_MANIPULATION", "0").lower() in ("1", "true", "yes")


def _interleave_enabled() -> bool:
    return os.getenv("GPSR_INTERLEAVE", "0").lower() in ("1", "true", "yes")


@dataclass
class Command:
    """One operator command + its parsed plan + progress (docs/GPSR_DESIGN.md §6)."""

    id: int
    utterance: str
    plan: Plan
    status: CmdStatus = CmdStatus.RECEIVED
    result_note: str | None = None


class GoToInstructionPoint(SubTask):
    """Enter the arena (asking for the door if needed) and drive to the
    instruction point."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
        # The arena door may be closed — ask a human to open it before driving in.
        # Reusable across challenges: tasks.skills.{request_open_door,go_to_through_door}.
        if os.getenv("GPSR_REQUEST_DOOR", "0").lower() in ("1", "true", "yes"):
            from tasks.skills import go_to_through_door, request_open_door
            # Fully-closed door: ask once, then self-watch the depth and walk in the
            # moment it reads open (no spoken confirmation needed).
            request_open_door(ctx)
            # Then drive — and if a *partly*-open door blocks nav (the doorway reads
            # open but the gap is too narrow), ask for it to be opened wider and retry.
            reached = go_to_through_door(ctx, x, y, h, ask_even_if_open=True, door_attempts=3)
        else:
            reached = ctx.goto(x, y, h)
        return StepResult.DONE if reached else StepResult.RETRY


class ReceiveAndPlanCommands(SubTask):
    """Take the operator's command(s), parse each into a typed plan, speak it.

    Speaking the plan is what scores "demonstrate a plan has been generated"
    (3×100) and doubles as the operator's confirmation. The robot *requests all
    three at once* (§5.5) to keep the interleave bonus reachable and save
    round-trips; in one-by-one mode ExecuteCommands returns between commands.

    Recovery (rulebook 5.3): re-ask only on an empty parse (§5.2 — each
    rephrasing costs −30), a bounded number of times, then **request a custom
    operator** (a clearer human; −20/command but recovers the command) before
    giving up. The receive loop is self-managed (not the SubTask retry counter)
    so it can escalate AND always leave ``ctx.data["commands"]`` set — the old
    behaviour silently left it unset and forfeited the whole run. On total
    failure it returns DONE (not ABORT) so the robot still returns to the
    instruction point and stays "attending".
    """

    max_retries = 0  # recovery is handled in-loop by _receive_commands

    def run(self, ctx: TaskContext) -> StepResult:
        world = ctx.data.get("world")
        if world is None:
            print("[gpsr] no world model on ctx.data['world'] — cannot plan")
            ctx.say(prompts.PLAN_NOT_UNDERSTOOD)
            return StepResult.ABORT
        ctx.say(prompts.GREET_OPERATOR)
        commands = self._receive_commands(ctx, world)
        ctx.data["commands"] = commands
        if commands:
            ctx.say(prompts.CONFIRM_RECEIVED)
        else:
            ctx.say(prompts.GIVE_UP_ON_COMMANDS)
        return StepResult.DONE

    def _receive_commands(self, ctx: TaskContext, world) -> list[Command]:
        """Ask for the command(s), escalating on failure; return the planned list.

        Returns as soon as one round yields at least one usable plan (a partial
        batch is accepted — re-asking the whole batch would re-pay −30, §5.2).
        Escalates: up to ``GPSR_MAX_REPHRASINGS`` rephrasing requests, then (if
        ``GPSR_USE_CUSTOM_OPERATOR``) a custom-operator request plus up to
        ``GPSR_CUSTOM_OPERATOR_ATTEMPTS`` further listens. ``[]`` once exhausted.
        """
        max_rephrasings = int(os.getenv("GPSR_MAX_REPHRASINGS", "2"))
        use_custom = os.getenv("GPSR_USE_CUSTOM_OPERATOR", "1").lower() in ("1", "true", "yes")
        max_custom = int(os.getenv("GPSR_CUSTOM_OPERATOR_ATTEMPTS", "3")) if use_custom else 0
        rephrasings = 0
        custom_attempts = 0
        requested_custom = False
        while True:
            # retries=0 so each ask is ONE say+listen — this loop owns all
            # re-prompting, so GPSR_MAX_REPHRASINGS is the true re-prompt bound
            # (ctx.ask's default retries=1 would re-prompt internally and inflate
            # the −30 count + the clock).
            answer = ctx.ask(prompts.ASK_FOR_COMMANDS, retries=0)
            if answer:
                parsed = parse_commands(ctx.model, answer, world)
                if any(plan for _, plan in parsed):  # at least one usable plan
                    return self._build_commands(ctx, parsed)
            # Nothing usable this round — escalate (rephrase, then custom operator).
            if rephrasings < max_rephrasings:
                rephrasings += 1
                ctx.say(prompts.ASK_REPHRASE)
            elif custom_attempts < max_custom:
                if not requested_custom:
                    requested_custom = True
                    ctx.say(prompts.REQUEST_CUSTOM_OPERATOR)
                custom_attempts += 1
            else:
                return []  # rephrasings + custom operator exhausted

    def _build_commands(self, ctx: TaskContext, parsed) -> list[Command]:
        """Turn parsed (text, Plan) pairs into Commands, speaking each plan."""
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
        return commands


class ExecuteCommands(SubTask):
    """Execute the planned commands via the two-tier dispatcher.

    Each PlanStep runs through its deterministic Tier-1 skill, falling back to the
    agent stack (Tier-2) for ungrounded/gated/failed steps; the per-command
    CmdStatus reflects partial scoring. Two modes:

    - **Serial (default, the MVP):** run each command's plan fully, in order. In
      one-by-one mode the robot returns to the operator between commands.
    - **Interleaved (`GPSR_INTERLEAVE`, the bonus):** merge all commands into one
      room-batched order (schedule.interleave) and walk it with a shared nav cache
      so each room is visited once — the "reduce unnecessary movements" the bonus
      rewards. Only when ≥2 commands were planned and not one-by-one; falls back
      to serial on any error. Degrades to a note if the brain (Tier-2) was unwired.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        brain = ctx.data.get("brain")
        world = ctx.data.get("world")
        commands: list[Command] = ctx.data.get("commands", [])
        if world is None:
            print("[gpsr] no world model on ctx.data['world'] — cannot execute")
            return StepResult.DONE
        manip = _manip_enabled()
        planned = [c for c in commands if c.plan]
        if _interleave_enabled() and not _one_by_one() and len(planned) >= 2:
            if self._run_interleaved(ctx, commands, world, brain, manip):
                return StepResult.DONE  # else fell through to serial below
        self._run_serial(ctx, commands, world, brain, manip)
        return StepResult.DONE

    def _run_serial(self, ctx, commands: list[Command], world, brain, manip: bool) -> None:
        for cmd in commands:
            cmd.status = CmdStatus.IN_PROGRESS
            ctx.say(prompts.COMMAND_ANNOUNCE.format(n=cmd.id, command=cmd.utterance))
            try:
                cmd.status = execute_plan(ctx, cmd.plan, world, brain, manip_enabled=manip)
            except Exception as exc:
                print(f"[gpsr] command {cmd.id} execution raised ({exc})")
                cmd.status = CmdStatus.FAILED
            print(f"[gpsr] command {cmd.id} -> {cmd.status.name}")
            if _one_by_one() and cmd.id < len(commands):
                x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
                ctx.goto(x, y, h)  # return to operator for the next command

    def _run_interleaved(self, ctx, commands: list[Command], world, brain, manip: bool) -> bool:
        """Execute all planned commands in one room-batched interleave. Returns
        False — with NO side effects yet — only if scheduling fails, so the caller
        can fall back to serial cleanly; once execution starts it never falls back
        (which would double-drive the robot)."""
        indexed = [(c.id, c.plan) for c in commands if c.plan]
        try:
            order = interleave(indexed, world)  # pure; the only pre-side-effect failure point
        except Exception as exc:
            print(f"[gpsr] interleave scheduling failed ({exc}); falling back to serial")
            return False
        ctx.say(prompts.INTERLEAVE_ANNOUNCE)
        for c in commands:
            c.status = CmdStatus.IN_PROGRESS if c.plan else CmdStatus.FAILED
        statuses = execute_interleaved(ctx, indexed, world, brain, manip_enabled=manip, order=order)
        for c in commands:
            if c.id in statuses:
                c.status = statuses[c.id]
            print(f"[gpsr] command {c.id} -> {c.status.name}")
        return True


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
    # Guests differ every run — stale identities must never match today's.
    if ctx.people is not None and os.getenv("GPSR_PEOPLE_RESET", "1").lower() in ("1", "true", "yes"):
        try:
            ctx.people.clear()
            print("[GPSR] people memory cleared for a fresh run")
        except Exception as exc:
            print(f"[GPSR] people memory reset failed ({exc})")
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
