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


def _tier2(ctx: TaskContext, brain, clause: str) -> bool:
    """Delegate one clause to the Walkie agent stack (the long-tail fallback)."""
    if brain is None or not clause:
        return False
    print(f"[gpsr.dispatch] Tier-2 fallback: {clause!r}")
    try:
        brain.walkie_agent.invoke(
            {"messages": [HumanMessage(content=clause)]},
            config={"configurable": {"thread_id": f"gpsr-{uuid.uuid4()}"}},
        )
        return True
    except Exception as exc:
        print(f"[gpsr.dispatch] Tier-2 fallback failed ({exc})")
        return False


def execute_plan(
    ctx: TaskContext,
    plan: Plan,
    world: WorldModel,
    brain,
    *,
    manip_enabled: bool,
    state: dict | None = None,
) -> CmdStatus:
    """Execute every step of *plan*; return the aggregate command status."""
    if not plan.steps:
        return CmdStatus.FAILED
    state = state if state is not None else {}
    oks: list[bool] = []
    for step in plan.steps:
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
            ok = _tier2(ctx, brain, step.raw or plan.source)
            # The agent may have driven the robot (delegate_to_actuator), so the
            # deterministic nav cache (state["at"]) can no longer be trusted — drop
            # it so a following navigate to a "known" place still actually drives.
            state.pop("at", None)
        oks.append(ok)
        print(f"[gpsr.dispatch] step {step.primitive.value}: {'ok' if ok else 'failed'}")
    return summarize_status(oks)
