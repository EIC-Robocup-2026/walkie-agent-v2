"""GPSR plan executor: walk a Plan, run each step Tier-1, fall back to Tier-2.

For each `PlanStep`: if it's eligible for deterministic dispatch
(`plan.prefer_tier1` — grounded, and not a gated manipulation step) and a skill
exists, run the skill, **retrying it a few times** before giving up. Most Tier-1
failures are small and transient (a flickered detection, a nav goal that needs a
second attempt), so re-running the deterministic skill is cheaper and far more
reliable than escalating to an LLM agent on the first stumble. Only once the
retries are exhausted — or the step was never Tier-1 eligible (ungrounded / gated)
— is the clause handed to a **scoped** agent fallback (Tier-2): the SINGLE
sub-agent whose domain matches the failed primitive (movement → actuator,
perception → vision), NOT the full Walkie orchestrator. That confines a recovery
attempt to the failed step's own kind of work — a nav failure is retried with
movement only, never by improvising extra actions. The command's `CmdStatus`
aggregates per-step success so partial scoring is reflected.

The routing/status *policy* is pure and offline-tested (plan.prefer_tier1 /
plan.summarize_status); this module is the robot-side wiring (it imports skills,
which import tasks.base) and is verified on the robot.
"""

from __future__ import annotations

import os
import time
import uuid

from langchain.messages import HumanMessage

from tasks.base import TaskContext

from .plan import CmdStatus, Plan, prefer_tier1, summarize_status
from .skills import SKILLS
from walkie_world.map.vocab import WorldModel


# --- scoped Tier-2 fallback -------------------------------------------------
# Which sub-agent recovers a failed step, keyed by primitive. The fallback invokes
# ONLY this one sub-agent (on `brain`), not the full Walkie orchestrator, so it is
# structurally confined to that agent's own tools: a navigate/guide/follow/pick/
# place/deliver failure can only be retried with movement+arm (actuator); a find/
# count/info failure only with the camera (vision). There is deliberately NO
# escalation to the full agent stack — if the scoped agent can't recover, the step
# is left FAILED (partial scoring) rather than letting a general agent act beyond
# the one failed step. (`pick`/`place`/`deliver` reach here either gated-off or
# after the grasp skill exhausted its own retries; routing them to the actuator
# matches the documented gate semantics — "fall through to the agent" — without
# widening the scope to non-arm work.)
_FALLBACK_ROUTE: dict[str, str] = {
    "navigate": "actuator",
    "guide": "actuator",
    "follow": "actuator",
    "pick": "actuator",
    "place": "actuator",
    "deliver": "actuator",
    "find_object": "vision",
    "find_person": "vision",
    "count": "vision",
    "greet": "vision",
    "get_person_info": "vision",
    "get_object_property": "vision",
    "say": "vision",
}

# Prepended to every scoped clause. A sub-agent's own prompt lets it ask the operator
# / improvise — correct for interactive use, wrong in GPSR: the command runs
# autonomously, nobody answers, and the executor does not wait for a reply (a question
# silently does nothing). Force it to commit to one choice, do ONLY the failed step,
# and not overreach into work the step never asked for.
_SCOPE_DIRECTIVE: dict[str, str] = {
    "actuator": (
        "You are autonomously recovering ONE failed step of a RoboCup GPSR command. "
        "There is NO operator to answer questions and you get no reply, so never ask "
        "to clarify — if several options match, silently pick the most likely and act. "
        "Do ONLY this one movement/arm sub-task; do not add any step beyond it. "
        "Report the outcome with `speak` in one short sentence.\n\nSub-task: "
    ),
    "vision": (
        "You are autonomously recovering ONE failed step of a RoboCup GPSR command. "
        "There is NO operator to answer questions and you get no reply, so never ask "
        "to clarify. Do ONLY this one perception sub-task using the camera and report "
        "what you see; do not move the robot or add any step beyond it. "
        "Report the outcome with `speak` in one short sentence.\n\nSub-task: "
    ),
}


def _scoped_fallback(ctx: TaskContext, brain, step, *, state: dict, source: str) -> bool:
    """Hand one failed step to the single sub-agent that owns its kind of work.

    Routed by primitive (`_FALLBACK_ROUTE`) to one sub-agent on `brain`; that agent
    only holds its own tools, so recovery stays scoped to the failed step (no full
    orchestrator, no escalation). Returns ``False`` (step stays failed) when there's
    no brain, no clause, or no route for the primitive.
    """
    clause = step.raw or source
    route = _FALLBACK_ROUTE.get(step.primitive.value)
    if brain is None or not clause or route is None:
        return False
    agent = getattr(brain, route, None)
    if agent is None:
        print(f"[gpsr.dispatch] no '{route}' sub-agent on brain; {step.primitive.value} failed")
        return False
    print(f"[gpsr.dispatch] scoped fallback -> {route}: {clause!r}")
    try:
        agent.invoke(
            {"messages": [HumanMessage(content=_SCOPE_DIRECTIVE[route] + clause)]},
            config={"configurable": {"thread_id": f"gpsr-{route}-{uuid.uuid4()}"}},
        )
    except Exception as exc:
        print(f"[gpsr.dispatch] scoped fallback ({route}) failed ({exc})")
        return False
    # An actuator fallback may have driven the robot, so the deterministic nav cache
    # (state["at"]) can no longer be trusted — drop it so a following navigate to a
    # "known" place still actually drives. A vision-only fallback never moves, so the
    # cache stays valid and is kept (avoids a spurious re-navigation).
    if route == "actuator":
        state.pop("at", None)
    return True


def _run_skill_with_retry(ctx, skill, step, world: WorldModel, state: dict) -> bool:
    """Run a Tier-1 skill, retrying a few times before deferring to the fallback.

    Most Tier-1 failures are transient (a flickered detection, a nav goal that needs
    a second attempt); retrying the deterministic skill is cheaper and more reliable
    than escalating to an LLM agent on the first stumble. Attempts and the pause
    between them are configurable (``GPSR_TIER1_RETRY_ATTEMPTS`` /
    ``GPSR_TIER1_RETRY_PAUSE_SEC``); a skill that raises is caught and counts as a
    failed attempt. ``attempts`` is the TOTAL number of tries (1 = today's behaviour,
    no retry).
    """
    attempts = max(1, int(os.getenv("GPSR_TIER1_RETRY_ATTEMPTS", "1")))
    pause = float(os.getenv("GPSR_TIER1_RETRY_PAUSE_SEC", "1.0"))
    prim = step.primitive.value
    for i in range(attempts):
        try:
            if skill(ctx, step, world, state):
                return True
        except Exception as exc:
            print(f"[gpsr.dispatch] skill {prim} raised ({exc})")
        if i + 1 < attempts:
            print(f"[gpsr.dispatch] Tier-1 {prim} attempt {i + 1}/{attempts} failed; retrying")
            if pause > 0:
                time.sleep(pause)
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
    """Run one PlanStep: Tier-1 skill (with retries) if eligible, else scoped fallback.

    Shared by the serial (`execute_plan`) and interleaved (`execute_interleaved`)
    executors so both route identically; `state` is the per-run scratch dict
    (nav-dedup etc.). `source` is the command clause used if the step has no raw.
    """
    ok = False
    if prefer_tier1(step, manip_enabled=manip_enabled):
        skill = SKILLS.get(step.primitive.value)
        if skill is not None:
            ok = _run_skill_with_retry(ctx, skill, step, world, state)
    if not ok:  # ungrounded, gated, no skill, or retries exhausted -> scoped fallback
        ok = _scoped_fallback(ctx, brain, step, state=state, source=source)
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
