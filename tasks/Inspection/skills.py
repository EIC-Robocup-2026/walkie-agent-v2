"""Deterministic handlers for the Inspection task (RoboCup@Home robot inspection).

The referee inspects the robot end-to-end: it must notice the entry door open and
drive through, stop for a person who steps in front (safety), reach the inspection
points, declare its external devices, prove it is loud enough, then move to the exit
on a signal. This is all fixed, position-known control flow — so it runs as a plain
deterministic scaffold with LIGHTWEIGHT custom model calls (``ctx.extract`` /
``ctx.model.invoke``) where a little language understanding helps (classifying the
"go to the exit" signal, answering the referee's questions). The heavyweight agent
stack is only pulled in when ``INSPECTION_AGENT_MODE=1`` (see :func:`wait_for_exit_signal`).

All ``tasks.skills`` imports are LAZY (inside the functions) so this module stays
offline-importable and pulls the perception stack only when run on the robot.
"""

from __future__ import annotations

import os
import time

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from tasks.base import TaskContext

from . import prompts


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Points of interest (read straight from config.toml, not the surveyed world.toml)
# ---------------------------------------------------------------------------
def _parse_pose_env(name: str):
    """Parse a ``"x,y,heading_rad"`` env var to a pose tuple, or None if unset/bad."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    from tasks.skills.geometry import parse_pose

    try:
        return parse_pose(raw)
    except ValueError:
        print(f"[inspection] bad {name}={raw!r}; ignoring")
        return None


def _parse_points_env(name: str) -> list:
    """Parse a ``';'``-separated list of ``"x,y,heading_rad"`` poses; skips bad ones."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    from tasks.skills.geometry import parse_pose

    out = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(parse_pose(chunk))
        except ValueError:
            print(f"[inspection] bad point {chunk!r} in {name}; skipping")
    return out


# ---------------------------------------------------------------------------
# Head control — level the head so the depth door/safety reads see the doorway
# and a standing torso, not the floor (memory: door-check-head-leveling-gap).
# ---------------------------------------------------------------------------
def _set_auto_tilt(ctx: TaskContext, enabled: bool) -> None:
    try:
        ctx.walkie.robot.head.set_auto_tilt(bool(enabled))
    except Exception as exc:  # noqa: BLE001 — off-robot stub may lack robot.head
        print(f"[inspection] set_auto_tilt({enabled}) failed ({exc})")


def _level_head(ctx: TaskContext) -> None:
    """Pin the head at the configured (level-ish) angle for a depth/pose read.

    Disables head auto-tilt first so it stays put while the robot is stationary.
    ``INSPECTION_HEAD_TILT_RAD`` is negative = up, positive = down (SDK convention);
    default 0.0 (level). The tilt offset is uncalibrated on the new robot box, so
    tune this knob on-robot if the door/safety read false-triggers.
    """
    from tasks.skills import tilt_head

    _set_auto_tilt(ctx, False)
    angle = float(os.getenv("INSPECTION_HEAD_TILT_RAD", "0.0"))
    tilt_head(ctx, angle, settle=float(os.getenv("INSPECTION_HEAD_SETTLE_SEC", "0.5")))


# ---------------------------------------------------------------------------
# Step 1 — notice the entry door open and drive through
# ---------------------------------------------------------------------------
def enter_through_door(ctx: TaskContext) -> bool:
    """Wait at the entry door, notice it open (depth self-watch), drive inside.

    ``request_open_door`` asks once then polls the depth camera and proceeds on its
    own the instant the doorway reads clear — the "robot notices the open door"
    behaviour the referee is checking. The head is levelled first so the depth read
    sees the doorway, not the floor. Then it drives to the surveyed entry pose
    (``go_to_through_door`` with ``ask_even_if_open=False``, since we already waited
    for it to open), or creeps forward blindly when no entry pose is configured.
    """
    from tasks.skills import go_to_through_door, move_base_relative, request_open_door

    ctx.say(prompts.READY_ANNOUNCE)
    # _level_head(ctx)
    try:
        request_open_door(ctx, prompt=prompts.ENTRY_DOOR_PROMPT)
    except Exception as exc:  # noqa: BLE001 — enter anyway if the door check fails
        print(f"[inspection] door wait failed ({exc}); entering anyway")
    ctx.say(prompts.ENTERED_ANNOUNCE)
    _set_auto_tilt(ctx, True)  # hand the head back to nav

    entry = _parse_pose_env("INSPECTION_ENTRY_POINT")
    if entry is not None:
        return go_to_through_door(ctx, *entry, ask_even_if_open=False)
    fwd = float(os.getenv("INSPECTION_ENTRY_FORWARD_M", "1.0"))
    print(f"[inspection] no INSPECTION_ENTRY_POINT; creeping {fwd:.2f} m forward")
    return move_base_relative(ctx, fwd)


# ---------------------------------------------------------------------------
# Step 2 — stop for a person stepping in front (safety demonstration)
# ---------------------------------------------------------------------------
def _depth_close(ctx: TaskContext) -> bool:
    """Is a near surface filling the centre of the frame? (something is in front).

    The safety-appropriate primitive: ``door_open_from_depth`` inverted — it returns
    True (open/clear) when the centre box is far/see-through and False when a near
    surface fills it, which is exactly "something is close in front of me". Depth
    unavailable (mock ctx / no camera) → False, so we never fabricate a phantom stop.
    """
    from tasks.skills import door_open_from_depth

    snap = ctx.snapshot()
    depth = getattr(snap, "depth", None) if snap is not None else None
    if depth is None:
        return False
    near_m = float(os.getenv("INSPECTION_SAFETY_NEAR_M", "1.2"))
    center = float(os.getenv("INSPECTION_SAFETY_CENTER_FRAC", "0.4"))
    min_valid = float(os.getenv("INSPECTION_SAFETY_MIN_VALID_FRAC", "0.5"))
    return door_open_from_depth(
        depth, clear_m=near_m, center_frac=center, min_valid_frac=min_valid
    ) is False


def _looks_like_person(ctx: TaskContext) -> bool:
    """Is a person standing centred and near in view? (pose estimation).

    Confirms the obstacle is a human (so the announce says "I see you" not "something
    is in front"). A person box counts when its centre-x sits within the central band
    and it covers at least ``INSPECTION_SAFETY_MIN_AREA_FRAC`` of the frame (near).
    """
    from tasks.skills import person_bboxes

    img = ctx.capture()
    if img is None:
        return False
    boxes = person_bboxes(ctx, img)
    if not boxes:
        return False
    w, h = img.width, img.height
    band = float(os.getenv("INSPECTION_SAFETY_CENTER_FRAC", "0.4"))
    area_min = float(os.getenv("INSPECTION_SAFETY_MIN_AREA_FRAC", "0.05"))
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        centred = abs(cx - w / 2) <= band * w / 2
        near = ((x2 - x1) * (y2 - y1)) / max(1.0, w * h) >= area_min
        if centred and near:
            return True
    return False


def demonstrate_safety_stop(ctx: TaskContext) -> bool:
    """Announce, then stop and hold while a person/obstacle is in front; go once clear.

    A visible/audible demonstration of the robot's safety behaviour layered on top of
    Nav2's own costmap avoidance (which already stops the base for obstacles during
    the drive to the inspection points). It never hangs: if nobody steps in front
    within ``INSPECTION_SAFETY_APPEAR_SEC`` it continues, and a hard
    ``INSPECTION_SAFETY_MAX_WAIT_SEC`` cap proceeds regardless. Returns whether a
    person/obstacle was actually seen and cleared.
    """
    poll = float(os.getenv("INSPECTION_SAFETY_POLL_SEC", "0.4"))
    appear = float(os.getenv("INSPECTION_SAFETY_APPEAR_SEC", "20"))
    max_wait = float(os.getenv("INSPECTION_SAFETY_MAX_WAIT_SEC", "90"))
    need_clear = max(1, int(os.getenv("INSPECTION_SAFETY_CLEAR_READS", "3")))

    ctx.say(prompts.SAFETY_ANNOUNCE)
    _level_head(ctx)
    start = time.monotonic()
    seen = False
    clear_streak = 0
    result = False
    try:
        while True:
            now = time.monotonic()
            if now - start >= max_wait:
                ctx.say(prompts.SAFETY_CLEAR if seen else prompts.SAFETY_NOONE)
                result = seen
                break
            try:
                is_person = _looks_like_person(ctx)
                blocking = _depth_close(ctx) or is_person
            except Exception as exc:  # noqa: BLE001 — a read glitch must not stall
                print(f"[inspection] safety read failed ({exc})")
                is_person, blocking = False, False
            if blocking:
                if not seen:
                    ctx.say(
                        prompts.SAFETY_PERSON_DETECTED if is_person
                        else prompts.SAFETY_OBSTACLE_DETECTED
                    )
                    seen = True
                clear_streak = 0
            elif seen:
                clear_streak += 1
                if clear_streak >= need_clear:
                    ctx.say(prompts.SAFETY_CLEAR)
                    result = True
                    break
            elif now - start >= appear:
                ctx.say(prompts.SAFETY_NOONE)
                result = True
                break
            time.sleep(poll)
    finally:
        _set_auto_tilt(ctx, True)  # hand the head back to nav
    return result


# ---------------------------------------------------------------------------
# Step 3 — visit the inspection points
# ---------------------------------------------------------------------------
def visit_inspection_points(ctx: TaskContext) -> None:
    """Drive to each configured inspection point in order, announcing arrival."""
    points = _parse_points_env("INSPECTION_POINTS")
    if not points:
        print("[inspection] no INSPECTION_POINTS configured; skipping")
        return
    for i, pose in enumerate(points, 1):
        ctx.goto(*pose)
        ctx.say(prompts.ARRIVED_AT_POINT.format(n=i))


# ---------------------------------------------------------------------------
# Step 4 — declare external devices
# ---------------------------------------------------------------------------
def _devices_list() -> list[str]:
    return [d.strip() for d in os.getenv("INSPECTION_EXTERNAL_DEVICES", "").split(";") if d.strip()]


def _devices_phrase() -> str:
    items = _devices_list()
    return ", ".join(items) if items else "none"


def declare_devices(ctx: TaskContext) -> None:
    """Tell the referee about the external devices in use (rulebook requirement)."""
    items = _devices_list()
    if not items:
        ctx.say(prompts.DEVICES_NONE)
        return
    ctx.say(prompts.DEVICES_INTRO)
    for device in items:
        ctx.say(device)
    ctx.say(prompts.DEVICES_OUTRO)


# ---------------------------------------------------------------------------
# Step 5 — loudness test
# ---------------------------------------------------------------------------
def loudness_test(ctx: TaskContext) -> None:
    """Speak a clear test phrase; optionally confirm audibility and repeat once."""
    phrase = os.getenv(
        "INSPECTION_LOUDNESS_PHRASE",
        "Testing my speaker volume. Can everyone hear me clearly? One, two, three.",
    )
    ctx.say(phrase)
    if not _truthy("INSPECTION_LOUDNESS_CONFIRM", "1"):
        return
    ctx.say(prompts.LOUDNESS_ASK_CONFIRM)
    reply = ctx.listen(timeout=float(os.getenv("INSPECTION_LOUDNESS_LISTEN_SEC", "12"))).lower()
    negative = any(w in reply for w in ("no", "not", "can't", "cannot", "barely", "hardly", "louder"))
    if negative:
        ctx.say(prompts.LOUDNESS_REPEAT)
        ctx.say(phrase)
    else:
        ctx.say(prompts.LOUDNESS_OK)


# ---------------------------------------------------------------------------
# Step 6 — wait for the "go to the exit" signal (custom model calls / agent mode)
# ---------------------------------------------------------------------------
class _ExitSignal(BaseModel):
    go_to_exit: bool = Field(
        description="True if the referee is telling the robot the inspection is over "
        "or to move to the exit / leave now; False for a question or unrelated remark."
    )


def is_exit_signal(ctx: TaskContext, text: str) -> bool:
    """Does the referee's utterance mean "move to the exit now"?

    Keyword match first (fast, deterministic — the guaranteed path when the referee
    types or says an obvious word), then a lightweight ``ctx.extract`` classification
    for less-obvious phrasings. A failed/None extraction defaults to False (treat as a
    question, not an accidental early exit).
    """
    low = (text or "").lower()
    words = [w.strip().lower() for w in os.getenv("INSPECTION_EXIT_SIGNAL_WORDS", "").split(";") if w.strip()]
    if any(w in low for w in words):
        return True
    parsed = ctx.extract(_ExitSignal, prompts.EXIT_CLASSIFY_INSTRUCTIONS, text)
    return bool(parsed and parsed.go_to_exit)


def answer_referee(ctx: TaskContext, question: str, devices: str | None = None) -> str:
    """One direct model call producing a short spoken answer to the referee.

    This is the "custom model call" the inspection prefers over the full agent: a
    single ``ctx.model.invoke`` with a system prompt describing Walkie + its declared
    devices. Never raises — degrades to a polite re-ask on any failure.
    """
    if devices is None:
        devices = _devices_phrase()
    try:
        reply = ctx.model.invoke([
            SystemMessage(content=prompts.QNA_SYSTEM.format(devices=devices)),
            HumanMessage(content=question),
        ])
        text = str(reply.content).strip()
        if text:
            return text
    except Exception as exc:  # noqa: BLE001
        print(f"[inspection] answer_referee failed ({exc})")
    return "I am sorry, I did not catch that. Could you please repeat it?"


def _respond_to_referee(ctx: TaskContext, text: str, *, brain, devices: str) -> None:
    """Reply to one referee utterance — via the full agent in agent mode, else a call."""
    if brain is not None:
        turn = prompts.AGENT_TURN.format(utterance=text, devices=devices)
        try:
            brain.walkie_agent.invoke(
                {"messages": [HumanMessage(content=turn)]},
                config={"configurable": {"thread_id": "inspection"}},
            )
            return
        except Exception as exc:  # noqa: BLE001 — fall back to a direct call
            print(f"[inspection] agent turn failed ({exc}); answering directly")
    ctx.say(answer_referee(ctx, text, devices))


def wait_for_exit_signal(ctx: TaskContext) -> None:
    """Listen until the referee signals "go to the exit", answering questions meanwhile.

    The finale must never stall, so there are three ways out, all deterministic:
      * an exit keyword / classified exit signal (spoken or typed);
      * a bare Enter while typing (``DISABLE_LISTENING``) — the always-works signal;
      * an idle timeout (heard nothing understood for ``INSPECTION_EXIT_IDLE_SEC``) or
        the ``INSPECTION_MAX_QNA_ROUNDS`` cap — proceed to the exit anyway.

    Between signals, each utterance is answered: by a direct model call, or by the
    full Walkie agent when ``INSPECTION_AGENT_MODE=1`` (the scaffold still owns this
    loop and just routes each non-signal utterance to it).
    """
    ctx.say(prompts.EXIT_WAIT_ANNOUNCE)
    while not ctx.walkie.robot.button.is_pressed:
        pass
    return
    brain = ctx.data.get("brain") if _truthy("INSPECTION_AGENT_MODE") else None
    devices = _devices_phrase()
    listen_timeout = float(os.getenv("INSPECTION_EXIT_LISTEN_TIMEOUT_SEC", "20"))
    idle_sec = float(os.getenv("INSPECTION_EXIT_IDLE_SEC", "90"))
    max_rounds = max(1, int(os.getenv("INSPECTION_MAX_QNA_ROUNDS", "20")))

    idle_deadline = time.monotonic() + idle_sec
    for _ in range(max_rounds):
        text = ctx.listen(timeout=listen_timeout)
        # Typed can't-fail path: a bare Enter under DISABLE_LISTENING = "go to exit".
        if ctx.disable_listening and text == "":
            return
        if not text:
            if time.monotonic() >= idle_deadline:
                ctx.say(prompts.EXIT_NO_SIGNAL)
                return
            continue
        idle_deadline = time.monotonic() + idle_sec
        if is_exit_signal(ctx, text):
            return
        _respond_to_referee(ctx, text, brain=brain, devices=devices)


# ---------------------------------------------------------------------------
# Step 7 — move to the exit (then the referee presses the stop button)
# ---------------------------------------------------------------------------
def go_to_exit(ctx: TaskContext) -> bool:
    """Announce, drive to the exit pose, and tell the referee it may be stopped."""
    ctx.say(prompts.EXIT_ACK)
    exit_pose = _parse_pose_env("INSPECTION_EXIT_POINT")
    reached = False
    if exit_pose is not None:
        reached = ctx.goto(*exit_pose)
    else:
        print("[inspection] no INSPECTION_EXIT_POINT configured; staying put")
    ctx.say(prompts.EXIT_ARRIVED)
    return reached
