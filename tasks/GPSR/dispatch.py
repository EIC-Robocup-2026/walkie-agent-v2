"""GPSR plan executor: walk a Plan, run each step Tier-1, fall back to Tier-2.

For each `PlanStep`: if it's eligible for deterministic dispatch
(`plan.prefer_tier1` — grounded, and not a gated manipulation step) and a skill
exists, run the skill; otherwise — or if the skill hard-fails — hand the step's
clause to the agent stack (Tier-2). The command's `CmdStatus` aggregates per-step
success so partial scoring is reflected.

The routing/status *policy* is pure and offline-tested (plan.prefer_tier1 /
plan.summarize_status); this module is the robot-side wiring (it imports skills,
which import tasks.base) and is verified on the robot.
"""

from __future__ import annotations

import uuid

from langchain.messages import HumanMessage

from tasks.base import TaskContext

from .plan import CmdStatus, Plan, prefer_tier1, summarize_status
from .skills import SKILLS
from .world import WorldModel


# Prepended to every Tier-2 clause. The Walkie agent's general prompt lets it ask
# the operator for help / to choose between candidates — correct for interactive
# use, but wrong in GPSR: the command is executed autonomously, nobody answers, and
# the executor does not wait for a reply (it marks the step done and moves on, so a
# question silently does nothing). Force it to commit to one choice and act.
_TIER2_DIRECTIVE = (
    "You are carrying out a single RoboCup GPSR command autonomously. There is NO "
    "operator available to answer follow-up questions and you will not receive any "
    "reply, so never ask the user to choose or to clarify. If several candidates "
    "match, silently pick the single most likely one (most recently / most "
    "confidently seen) and act on it. Complete the command end to end, then report "
    "the outcome with `speak` in one short sentence.\n\nCommand: "
)


def _tier2(ctx: TaskContext, brain, clause: str) -> bool:
    """Delegate one clause to the Walkie agent stack (the long-tail fallback)."""
    if brain is None or not clause:
        return False
    print(f"[gpsr.dispatch] Tier-2 fallback: {clause!r}")
    try:
        brain.walkie_agent.invoke(
            {"messages": [HumanMessage(content=_TIER2_DIRECTIVE + clause)]},
            config={"configurable": {"thread_id": f"gpsr-{uuid.uuid4()}"}},
        )
        return True
    except Exception as exc:
        print(f"[gpsr.dispatch] Tier-2 fallback failed ({exc})")
        return False


def execute_step(
    ctx: TaskContext,
    step,
    world: WorldModel,
    brain,
    *,
    manip_enabled: bool,
    state: dict,
    source: str = "",
) -> bool:
    """Run one PlanStep: Tier-1 skill if eligible, else the Tier-2 agent fallback.

    Shared by the serial (`execute_plan`) and interleaved (`execute_interleaved`)
    executors so both route identically; `state` is the per-run scratch dict
    (nav-dedup etc.). `source` is the command clause used if the step has no raw.
    """
    ok = False
    if prefer_tier1(step, manip_enabled=manip_enabled):
        skill = SKILLS.get(step.primitive.value)
        if skill is not None:
            try:
                ok = skill(ctx, step, world, state)
            except Exception as exc:
                print(f"[gpsr.dispatch] skill {step.primitive.value} raised ({exc})")
                ok = False
    if not ok:  # ungrounded, gated, no skill, or skill error -> agent fallback
        ok = _tier2(ctx, brain, step.raw or source)
        # The agent may have driven the robot (delegate_to_actuator), so the
        # deterministic nav cache (state["at"]) can no longer be trusted — drop it
        # so a following navigate to a "known" place still actually drives.
        state.pop("at", None)
    print(f"[gpsr.dispatch] step {step.primitive.value}: {'ok' if ok else 'failed'}")
    return ok


def execute_plan(
    ctx: TaskContext,
    plan: Plan,
    world: WorldModel,
    brain,
    *,
    manip_enabled: bool,
    state: dict | None = None,
) -> CmdStatus:
    """Execute every step of *plan* in order; return the aggregate command status."""
    if not plan.steps:
        return CmdStatus.FAILED
    state = state if state is not None else {}
    oks = [
        execute_step(ctx, s, world, brain, manip_enabled=manip_enabled, state=state, source=plan.source)
        for s in plan.steps
    ]
    return summarize_status(oks)


def execute_interleaved(
    ctx: TaskContext,
    indexed: list[tuple[int, Plan]],
    world: WorldModel,
    brain,
    *,
    manip_enabled: bool,
    order: list[tuple[int, int]],
) -> dict[int, CmdStatus]:
    """Walk *order* (from schedule.interleave) across commands.

    Only the robot's physical location (``state["at"]``, the nav-dedup) is
    GLOBAL — a room entered for one command is not re-entered for another, which
    is the point of interleaving. Everything else is **per-command** scratch:
    each command keeps its own ``state`` dict, so an interleaved step of command B
    cannot clobber the ``target_xy``/``found_object`` command A stashed for its
    own next step. Returns each command's aggregate status (partial scoring).
    """
    plans = {cid: plan for cid, plan in indexed}
    oks_by_cmd: dict[int, list[bool]] = {cid: [] for cid, _ in indexed}
    states: dict[int, dict] = {cid: {} for cid, _ in indexed}  # per-command scratch
    at: str | None = None  # robot's location — the ONLY state shared across commands
    for cid, idx in order:
        st = states[cid]
        if at is not None:
            st["at"] = at
        else:
            st.pop("at", None)
        ok = execute_step(
            ctx, plans[cid].steps[idx], world, brain,
            manip_enabled=manip_enabled, state=st, source=plans[cid].source,
        )
        oks_by_cmd[cid].append(ok)
        at = st.get("at")  # propagate a nav move (or Tier-2's pop -> None) globally
    return {
        cid: (summarize_status(oks) if oks else CmdStatus.FAILED)
        for cid, oks in oks_by_cmd.items()
    }
