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
    world:    WalkieWorld          # ctx.world: arena nouns + scene graph + people, set by run.py
    commands: list[Command]        # parsed + planned operator commands

Live scoring (ctx.score, GPSR_SHEET): unlike PnP/Restaurant/HRI — whose tallies are
pure-optimistic positives — GPSR *also* tallies penalties (pen_rephrasing,
pen_bypass_stt, pen_custom_operator), because they are deterministic and owned by
this flow. So a GPSR scorecard number below another challenge's is not a regression:
it nets the re-ask / typed-bypass / custom-operator costs the receive loop incurs.
Still attempted/claimed, NOT referee-awarded (see tasks/scoring.py).
"""

from __future__ import annotations

import os
import re
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


def _return_between_commands() -> bool:
    """Whether the serial executor ALWAYS drives back to the instruction point
    after finishing each command, re-stationing there before starting the next
    (GPSR_RETURN_BETWEEN_COMMANDS) — regardless of GPSR_ISSUE_MODE (one-by-one
    mode already implied it). In-code default OFF (the pre-change behaviour, and
    what the unit tests exercise); tasks/GPSR/config.toml turns it ON for real
    runs. The final return after the LAST command stays owned by the envelope's
    ReturnToInstructionPoint."""
    return os.getenv("GPSR_RETURN_BETWEEN_COMMANDS", "0").lower() in ("1", "true", "yes")


def _speak_plan_enabled() -> bool:
    return os.getenv("GPSR_SPEAK_PLAN", "1").lower() in ("1", "true", "yes")


def _manip_enabled() -> bool:
    return os.getenv("GPSR_ENABLE_MANIPULATION", "0").lower() in ("1", "true", "yes")


def _interleave_enabled() -> bool:
    return os.getenv("GPSR_INTERLEAVE", "0").lower() in ("1", "true", "yes")


def _confirm_plan_enabled() -> bool:
    """Whether to ask a human to approve each plan before executing it (off by
    default — the rulebook GPSR run is autonomous; on for supervised demos)."""
    return os.getenv("GPSR_CONFIRM_PLAN", "0").lower() in ("1", "true", "yes")


def _confirm_default_proceed() -> bool:
    """On an UNCLEAR/silent confirmation answer, proceed (default) or skip? Keeps an
    STT mishear from silently forfeiting (or wrongly running) a command. Set
    GPSR_CONFIRM_DEFAULT="skip" to require a clear yes."""
    return os.getenv("GPSR_CONFIRM_DEFAULT", "proceed").strip().lower() != "skip"


def _verify_enabled() -> bool:
    """Whether to read each heard command back for a yes/no + re-capture on "no"
    (GPSR_VERIFY_COMMANDS). In-code default OFF (the pre-change behaviour, and
    what the unit tests exercise); tasks/GPSR/config.toml turns it ON for real
    runs — the referee reads all three commands in one halting stream, and a
    mishear caught here costs one −30 re-ask instead of a wasted errand."""
    return os.getenv("GPSR_VERIFY_COMMANDS", "0").lower() in ("1", "true", "yes")


def _verify_exhausted_skip() -> bool:
    """When a command stays unconfirmed after the re-capture budget: execute the
    best understanding anyway (in-code default — preserves the pre-knob
    behaviour) or skip it (GPSR_VERIFY_EXHAUSTED="skip", what config.toml sets:
    the operator explicitly rejected that text N times, so executing it is a
    likely-wrong errand — the sure-points play is spending those minutes on the
    confirmed commands instead)."""
    return os.getenv("GPSR_VERIFY_EXHAUSTED", "execute").strip().lower() == "skip"


def _batch_listen_kwargs() -> dict:
    """ctx.ask kwargs for the ONE continuous 3-command capture. The referee reads
    all three in a row, often haltingly, so the capture needs a wide window and a
    bigger end-of-speech silence than the mic's 1 s default (which would cut the
    recording at the first mid-command stumble)."""
    return {
        "timeout": float(os.getenv("GPSR_LISTEN_TIMEOUT_SEC", "90")),
        "min_silence_ms": int(os.getenv("GPSR_LISTEN_MIN_SILENCE_MS", "2500")),
    }


def _single_listen_kwargs() -> dict:
    """ctx.ask kwargs for re-capturing ONE corrected command (shorter than the
    batch window, still stumble-tolerant)."""
    return {
        "timeout": float(os.getenv("GPSR_RELISTEN_TIMEOUT_SEC", "30")),
        "min_silence_ms": int(os.getenv("GPSR_RELISTEN_MIN_SILENCE_MS", "1500")),
    }


_AFFIRM_WORDS = {
    "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "correct", "right",
    "proceed", "affirmative", "confirm", "confirmed", "alright", "fine", "perfect",
    "approved", "approve",
}
_AFFIRM_PHRASES = ("go ahead", "do it", "sounds good", "please do", "carry on", "go for it")
_NEGATE_WORDS = {
    "no", "nope", "nah", "dont", "stop", "cancel", "skip", "negative", "wrong",
    "incorrect", "abort",
}
_NEGATE_PHRASES = ("do not", "don't", "never mind", "nevermind", "not now")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", (text or "").lower()))


def _is_negative(text: str) -> bool:
    t = (text or "").lower()
    return bool(_tokens(t) & _NEGATE_WORDS) or any(p in t for p in _NEGATE_PHRASES)


def _is_affirmative(text: str) -> bool:
    t = (text or "").lower()
    return bool(_tokens(t) & _AFFIRM_WORDS) or any(p in t for p in _AFFIRM_PHRASES)


@dataclass
class Command:
    """One operator command + its parsed plan + progress (docs/GPSR_DESIGN.md §6)."""

    id: int
    utterance: str
    plan: Plan
    status: CmdStatus = CmdStatus.RECEIVED
    result_note: str | None = None
    # Operator approval of the plan (the GPSR_CONFIRM_PLAN gate). Defaults True so
    # that with the gate OFF every command runs exactly as before; set False only
    # when a human declines the spoken plan, and ExecuteCommands then skips it.
    confirmed: bool = True


class GoToInstructionPoint(SubTask):
    """Enter the arena (asking for the door if needed) and drive to the
    instruction point."""

    critical = True

    def run(self, ctx: TaskContext) -> StepResult:
        # results = ctx.world.resolve_place("fridge")
        # print(results)
        # for result in results:
        #     print(result.id, result.centroid)

        # while True:
        #     pass


        x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
        # The arena door may be closed — ask a human to open it before driving in.
        # Reusable across challenges: tasks.skills.{request_open_door,go_to_through_door}.
        if os.getenv("GPSR_REQUEST_DOOR", "0").lower() in ("1", "true", "yes"):
            from tasks.skills import go_to_through_door, request_open_door
            # Fully-closed door: ask once, then self-watch the depth and walk in the
            # moment it reads open (no spoken confirmation needed).
            # ctx.walkie.robot.head(0.4)
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
    three at once* (§5.5) to keep the interleave bonus reachable and save the
    listen round-trips; ExecuteCommands still re-stations at the instruction
    point between commands (GPSR_RETURN_BETWEEN_COMMANDS / one-by-one mode).

    Recovery (rulebook 5.3): re-ask only on an empty parse (§5.2 — each
    rephrasing costs −30), a bounded number of times, then **request a custom
    operator** (a clearer human; −20/command but recovers the command) before
    giving up. The receive loop is self-managed (not the SubTask retry counter)
    so it can escalate AND always leave ``ctx.data["commands"]`` set — the old
    behaviour silently left it unset and forfeited the whole run. On total
    failure it returns DONE (not ABORT) so the robot still returns to the
    instruction point and stays "attending".

    Verification (GPSR_VERIFY_COMMANDS, the sure-points gate): the referee reads
    all three commands in ONE continuous, halting stream, so the batch capture
    uses a wide listen window (GPSR_LISTEN_TIMEOUT_SEC / _MIN_SILENCE_MS) and,
    after parsing, each command is read back for a yes/no — a "no" re-captures
    JUST that command (−30 each, bounded), and a merged-away command is
    recovered by asking whether all commands were heard. Only after every
    command is confirmed does planning/execution proceed.
    """

    max_retries = 0  # recovery is handled in-loop by _receive_commands

    def run(self, ctx: TaskContext) -> StepResult:
        world = getattr(ctx, "world", None)
        if world is None:
            print("[gpsr] no world model on ctx.world — cannot plan")
            ctx.say(prompts.PLAN_NOT_UNDERSTOOD)
            return StepResult.ABORT
        ctx.say(prompts.GREET_OPERATOR_VERIFY if _verify_enabled() else prompts.GREET_OPERATOR)
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
            # the −30 count + the clock). The wide batch-listen window keeps a
            # halting referee's mid-command pauses from cutting the capture.
            answer = ctx.ask(prompts.ASK_FOR_COMMANDS, retries=0, **_batch_listen_kwargs())
            print(f"[GPSR] heard: {answer}")
            if answer:
                parsed = parse_commands(ctx.model, answer, world)
                if any(plan for _, plan in parsed):  # at least one usable plan
                    declined: set[int] = set()
                    if _verify_enabled():
                        parsed, declined = self._verify_commands(ctx, world, parsed)
                    return self._build_commands(ctx, parsed, declined=declined)
            # Nothing usable this round — escalate (rephrase, then custom operator).
            if rephrasings < max_rephrasings:
                rephrasings += 1
                ctx.say(prompts.ASK_REPHRASE)
                ctx.score("pen_rephrasing")  # −30 per requested rephrasing (claimed)
            elif custom_attempts < max_custom:
                if not requested_custom:
                    requested_custom = True
                    ctx.say(prompts.REQUEST_CUSTOM_OPERATOR)
                    # −20 *per command* in the rulebook; we record one unit at the
                    # request (the optimistic-ceiling choice — see module docstring).
                    ctx.score("pen_custom_operator")
                custom_attempts += 1
            else:
                return []  # rephrasings + custom operator exhausted

    def _verify_commands(self, ctx: TaskContext, world, parsed) -> tuple[list, set[int]]:
        """Read each heard command back for a yes/no, then recover any command
        the split merged away (GPSR_VERIFY_COMMANDS — the sure-points gate).

        Execution starts only after every command is confirmed: the referee
        reads all three in one halting stream, so per-command verification
        converts a mishear/mis-split into one cheap re-capture instead of a
        wasted multi-minute errand. Returns ``(pairs, declined)`` — the verified
        (text, plan) pairs plus the 1-based positions that stayed UNCONFIRMED
        after the re-capture budget and must not execute
        (GPSR_VERIFY_EXHAUSTED="skip"); empty in "execute" mode.
        """
        verified: list = []
        declined: set[int] = set()
        for i, (text, plan) in enumerate(parsed, 1):
            text, plan, ok = self._verify_one(ctx, world, i, text, plan)
            verified.append((text, plan))
            if not ok:
                declined.add(i)
        self._recover_missing(ctx, world, verified, declined)
        return verified, declined

    def _verify_one(self, ctx: TaskContext, world, n: int, text: str, plan: Plan):
        """Confirm command *n* with the operator; on "no" re-capture JUST it.

        A clear "no" asks the operator to repeat only that command (−30
        rephrasing each, bounded by GPSR_VERIFY_MAX_RECAPTURES), re-parses and
        re-confirms; a clear "yes" keeps it as heard. Negative is checked FIRST
        (mirrors _confirm_plan): on a mixed answer ("okay, no, that's wrong"),
        wrongly keeping a misheard command wastes a multi-minute errand while
        wrongly re-capturing a correct one costs only −30 — bias toward the
        cheap mistake. An unclear/silent verdict follows GPSR_CONFIRM_DEFAULT:
        "proceed" (default) keeps the command as heard — a mumbled "yes" must
        not forfeit it — while "skip" demands a clear yes. A re-capture is
        accepted only when it parses to a USABLE plan; reading back an
        unplannable text would waste a confirmation round on a guaranteed
        forfeit. Returns ``(text, plan, confirmed)`` — ``confirmed`` False only
        when the budget ran out in GPSR_VERIFY_EXHAUSTED="skip" mode.
        """
        max_recaptures = int(os.getenv("GPSR_VERIFY_MAX_RECAPTURES", "2"))
        recaptures = 0
        while True:
            answer = ctx.ask(prompts.ASK_VERIFY_COMMAND.format(n=n, command=text), retries=1)
            if _is_negative(answer):
                pass  # fall through to the re-capture below
            elif _is_affirmative(answer):
                ctx.say(prompts.VERIFY_CONFIRMED.format(n=n))
                return text, plan, True
            elif _confirm_default_proceed():
                print(f"[gpsr] command {n}: unclear verification {answer!r} -> keep as heard")
                return text, plan, True
            # Clear "no" (or strict mode on an unclear answer) -> re-capture.
            if recaptures >= max_recaptures:
                if _verify_exhausted_skip():
                    ctx.say(prompts.VERIFY_GIVE_UP.format(n=n))
                    return text, plan, False
                ctx.say(prompts.VERIFY_BEST_EFFORT.format(n=n))
                return text, plan, True
            recaptures += 1
            ctx.score("pen_rephrasing")  # asking to re-say costs −30 (§5.2)
            heard = ctx.ask(prompts.ASK_REPEAT_ONE.format(n=n), retries=0,
                            **_single_listen_kwargs())
            if heard:
                reparsed = parse_commands(ctx.model, heard, world)
                if reparsed and reparsed[0][1]:  # need a usable plan, not just text
                    if len(reparsed) > 1:
                        print(f"[gpsr] re-capture of command {n} split into "
                              f"{len(reparsed)} commands; keeping the first")
                    text, plan = reparsed[0]
                    continue  # loop re-confirms the corrected version
            ctx.say(prompts.VERIFY_RECAPTURE_MISSED)  # then re-confirm what we had

    def _recover_missing(self, ctx: TaskContext, world, commands: list, declined: set[int]) -> None:
        """Recover commands the batch split merged away (halting speech blurs the
        seam between two commands, leaving fewer than the operator gave).

        When fewer than GPSR_MAX_COMMANDS were heard, ask; a "no" (not all)
        captures the missing command(s) one at a time, each verified like the
        rest. A "yes"/unclear answer moves on — operators legitimately give
        fewer than three in practice runs. Any capture/parse failure breaks out
        rather than looping (the clock outweighs a maybe-recoverable command).
        Mutates ``commands``/``declined`` in place.
        """
        max_n = int(os.getenv("GPSR_MAX_COMMANDS", "3"))
        while len(commands) < max_n:
            answer = ctx.ask(prompts.ASK_GOT_ALL.format(n=len(commands)), retries=1)
            if not _is_negative(answer):
                break  # "yes" or unclear: that was all of them
            ctx.score("pen_rephrasing")  # asking to re-say costs −30 (§5.2)
            heard = ctx.ask(prompts.ASK_SAY_MISSING, retries=0, **_single_listen_kwargs())
            reparsed = parse_commands(ctx.model, heard, world) if heard else []
            if not reparsed:
                ctx.say(prompts.VERIFY_RECAPTURE_MISSED)
                break
            for text, plan in reparsed[: max_n - len(commands)]:
                text, plan, ok = self._verify_one(ctx, world, len(commands) + 1, text, plan)
                commands.append((text, plan))
                if not ok:
                    declined.add(len(commands))

    def _build_commands(
        self, ctx: TaskContext, parsed, declined: set[int] = frozenset()
    ) -> list[Command]:
        """Turn parsed (text, Plan) pairs into Commands, speaking each plan.

        ``declined`` (1-based positions from _verify_commands) marks commands the
        operator could not confirm in GPSR_VERIFY_EXHAUSTED="skip" mode: they
        keep their plan (status PLANNED) but are built unconfirmed so
        ExecuteCommands skips them — the existing decline machinery. Their plan
        is NOT spoken (the robot just announced it is skipping that command;
        speaking a plan for it would contradict that) and no speak_plan point is
        claimed.
        """
        commands: list[Command] = []
        for i, (text, plan) in enumerate(parsed, 1):
            cmd = Command(id=i, utterance=text, plan=plan)
            if i in declined:  # verify budget spent in "skip" mode -> never execute
                cmd.confirmed = False
                cmd.result_note = "skipped: operator could not confirm the command"
            # The command reached us as text → STT understood it (80 each). Typing it
            # instead (DISABLE_LISTENING) bypasses STT: forfeits the +80, costs −50.
            if ctx.disable_listening:
                ctx.score("pen_bypass_stt")
            else:
                ctx.score("understand_stt")
            if plan:
                cmd.status = CmdStatus.PLANNED
                if _speak_plan_enabled() and i not in declined:
                    ctx.say(render_plan_speech(plan, preamble=prompts.PLAN_PREAMBLE.format(n=i)))
                    ctx.score("speak_plan")  # demonstrated a generated plan (100 each)
                # Optional human approval gate: "plan is this — okay to do it?"
                # (moot for a command verification already declined)
                if _confirm_plan_enabled() and cmd.confirmed:
                    cmd.confirmed = self._confirm_plan(ctx, i)
                if not plan.is_complete:
                    print(f"[gpsr] command {i} plan has ungrounded steps: {plan.source!r}")
            else:
                ctx.say(prompts.PLAN_NOT_UNDERSTOOD)
            commands.append(cmd)
            print(f"[gpsr] command {i}: {text!r} -> {[s.primitive.value for s in plan.steps]}"
                  f"{'' if cmd.confirmed else ' (declined)'}")
        return commands

    def _confirm_plan(self, ctx: TaskContext, n: int) -> bool:
        """Ask the human to approve command *n*'s spoken plan before it executes.

        A clear "no" skips the command; a clear "yes" runs it. An unclear/silent
        answer (even after ctx.ask's one re-prompt) falls back to GPSR_CONFIRM_DEFAULT
        so an STT mishear doesn't silently forfeit (or wrongly run) the command.
        """
        answer = ctx.ask(prompts.ASK_CONFIRM_PLAN.format(n=n), retries=1)
        if _is_negative(answer):
            ctx.say(prompts.PLAN_REJECTED)
            return False
        if _is_affirmative(answer):
            ctx.say(prompts.PLAN_CONFIRMED)
            return True
        proceed = _confirm_default_proceed()
        print(f"[gpsr] command {n}: unclear confirmation {answer!r} -> "
              f"{'proceed' if proceed else 'skip'} (GPSR_CONFIRM_DEFAULT)")
        ctx.say(prompts.PLAN_CONFIRMED if proceed else prompts.PLAN_REJECTED)
        return proceed


class ExecuteCommands(SubTask):
    """Execute the planned commands via the two-tier dispatcher.

    Each PlanStep runs through its deterministic Tier-1 skill, falling back to the
    agent stack (Tier-2) for ungrounded/gated/failed steps; the per-command
    CmdStatus reflects partial scoring. Two modes:

    - **Serial (default, the MVP):** run each command's plan fully, in order.
      After finishing each command the robot drives back and re-stations at the
      instruction point before starting the next (GPSR_RETURN_BETWEEN_COMMANDS,
      ON in config.toml for real runs; one-by-one mode always returns). The
      final return after the last command stays with ReturnToInstructionPoint.
    - **Interleaved (`GPSR_INTERLEAVE`, the bonus):** merge all commands into one
      room-batched order (schedule.interleave) and walk it with a shared nav cache
      so each room is visited once — the "reduce unnecessary movements" the bonus
      rewards. Only when ≥2 commands were planned and not one-by-one; falls back
      to serial on any error. Degrades to a note if the brain (Tier-2) was unwired.
    """

    def run(self, ctx: TaskContext) -> StepResult:
        brain = ctx.data.get("brain")
        world = getattr(ctx, "world", None)
        commands: list[Command] = ctx.data.get("commands", [])
        if world is None:
            print("[gpsr] no world model on ctx.world — cannot execute")
            return StepResult.DONE
        manip = _manip_enabled()
        # Only confirmed, planned commands are eligible to run (the GPSR_CONFIRM_PLAN
        # gate; confirmed is always True when the gate is off).
        planned = [c for c in commands if c.plan and c.confirmed]
        ran_interleaved = False
        if _interleave_enabled() and not _one_by_one() and len(planned) >= 2:
            ran_interleaved = self._run_interleaved(ctx, commands, world, brain, manip)
        if not ran_interleaved:  # interleave disabled, unfit, or failed to schedule
            self._run_serial(ctx, commands, world, brain, manip)
        # interleave_bonus needs *meaningful* interleaving, not merely all-3-at-once
        # (docs/GPSR_DESIGN.md §5.5) — so it scores only when the room-batched
        # scheduler actually executed (GPSR_INTERLEAVE=1), never on the serial MVP.
        if ran_interleaved:
            ctx.score("interleave_bonus")
        # One solve unit (250) per command that made progress. PARTIAL is NOT
        # pro-rated here — partial-ness lives in the estimate's capture %, while the
        # live tally is the optimistic claimed ceiling (see tasks/scoring.py).
        for cmd in commands:
            if cmd.status in (CmdStatus.DONE, CmdStatus.PARTIAL):
                ctx.score("solve_command")
        return StepResult.DONE

    def _run_serial(self, ctx, commands: list[Command], world, brain, manip: bool) -> None:
        for cmd in commands:
            if cmd.plan and not cmd.confirmed:  # operator declined the plan -> skip
                # (or-guard: don't clobber a verify-gate skip reason set at build)
                cmd.result_note = cmd.result_note or "skipped: plan not approved"
                ctx.say(prompts.COMMAND_SKIPPED.format(n=cmd.id))
                print(f"[gpsr] command {cmd.id} skipped (plan not approved)")
                continue
            cmd.status = CmdStatus.IN_PROGRESS
            ctx.say(prompts.COMMAND_ANNOUNCE.format(n=cmd.id, command=cmd.utterance))
            state: dict = {}  # per-command scratch; also collects Tier-2 fallback notes
            try:
                cmd.status = execute_plan(ctx, cmd.plan, world, brain, manip_enabled=manip, state=state)
            except Exception as exc:
                print(f"[gpsr] command {cmd.id} execution raised ({exc})")
                cmd.status = CmdStatus.FAILED
            # Surface what any scoped Tier-2 fallback did (dispatch stashes each
            # sub-agent's spoken line in state["_notes"]) without clobbering a note
            # already set (e.g. a skip reason).
            notes = state.get("_notes")
            if notes and not cmd.result_note:
                cmd.result_note = "; ".join(notes)
            print(f"[gpsr] command {cmd.id} -> {cmd.status.name}")
            # Re-station at the instruction point before the NEXT command (never
            # after the last — ReturnToInstructionPoint owns the final return).
            if (_one_by_one() or _return_between_commands()) and cmd.id < len(commands):
                ctx.say(prompts.RETURN_BETWEEN_ANNOUNCE.format(n=cmd.id))
                x, y, h = _pose("GPSR_INSTRUCTION_POINT_POSE")
                ctx.goto(x, y, h)

    def _run_interleaved(self, ctx, commands: list[Command], world, brain, manip: bool) -> bool:
        """Execute all planned commands in one room-batched interleave. Returns
        False — with NO side effects yet — only if scheduling fails, so the caller
        can fall back to serial cleanly; once execution starts it never falls back
        (which would double-drive the robot)."""
        indexed = [(c.id, c.plan) for c in commands if c.plan and c.confirmed]
        try:
            order = interleave(indexed, world)  # pure; the only pre-side-effect failure point
        except Exception as exc:
            print(f"[gpsr] interleave scheduling failed ({exc}); falling back to serial")
            return False
        ctx.say(prompts.INTERLEAVE_ANNOUNCE)
        scheduled = {cid for cid, _ in indexed}
        for c in commands:
            if c.id in scheduled:
                c.status = CmdStatus.IN_PROGRESS
            elif c.plan and not c.confirmed:
                # declined -> stays PLANNED, not run (or-guard keeps a verify-gate reason)
                c.result_note = c.result_note or "skipped: plan not approved"
            else:
                c.status = CmdStatus.FAILED  # had no usable plan
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
