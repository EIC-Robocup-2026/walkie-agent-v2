"""Ask a human to open the arena door, then proceed — a reusable challenge-entry
primitive.

Most RoboCup@Home runs start with the robot OUTSIDE the arena; it may enter only
once a (usually human-operated) door is opened. That's the same across every
challenge, so it lives in the shared skills package rather than in one task:

    from tasks.skills import request_open_door

The skill is prompt-driven. It decides the door is open by — in order — a
**depth clear-path check** (the built-in :func:`is_door_open`, or a custom
``is_open()`` a challenge plugs in), a spoken confirmation heard on the mic, or,
after the budget runs out, proceeding anyway (the robot has to get in). The
detector degrades to ``None`` (→ ask) when there's no camera, so the spoken flow
still works on a GPU-less box / mock context. ``request_open_door`` is pure
control flow over ``ctx.say`` / ``ctx.listen`` / ``ctx.snapshot``; the
``door_open_from_depth`` decision is pure numpy and unit-tested directly.
"""

from __future__ import annotations

import math
import os
import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:  # type-only — keeps this skill importable on a GPU-less box
    import numpy as np
    from tasks.base import TaskContext

DEFAULT_PROMPT = (
    "The door appears to be closed. Could someone please open it for me so I can "
    "enter? Let me know when it is open."
)
# Any of these heard in the reply is taken as "the door is open". Lowercase;
# matched as substrings of the (lowercased) reply.
DEFAULT_CONFIRM_WORDS: tuple[str, ...] = (
    "open", "opened", "done", "ready", "yes", "okay", "go ahead", "clear", "come in",
)
DEFAULT_THANKS = "Thank you. I am coming in now."
# Said when the doorway *reads* open but the robot still can't drive through — a
# partly-open door: the centre is clear (so the depth check says "open") yet the gap
# is too narrow to fit. Depth can't see that it widened, so nav success is the only
# real signal; we ask, wait, and retry.
PARTIAL_OPEN_PROMPT = (
    "The door is only partly open and I cannot get through. "
    "Could you please open it all the way for me?"
)


# ---------------------------------------------------------------------------
# Door-state detection (the built-in `is_open` sensor)
# ---------------------------------------------------------------------------
def door_open_from_depth(
    depth: "np.ndarray",
    *,
    clear_m: float = 1.2,
    center_frac: float = 0.4,
    min_valid_frac: float = 0.5,
) -> bool:
    """Decide door open/closed from one aligned depth frame (metres, NaN/0 invalid).

    Looks at the central ``center_frac`` box of the frame — the doorway straight
    ahead — and calls the door **CLOSED** when a near surface fills it: at least
    ``min_valid_frac`` of those pixels are valid AND their median depth is under
    ``clear_m``. Otherwise **OPEN** — a clear path reads as far or see-through
    (large or invalid depth). Edges (floor/ceiling/door-frame) are excluded by
    cropping to the centre. Pure (numpy only) → unit-tested; tune the thresholds
    on-robot (``WALKIE_DOOR_*``). A degenerate frame returns True (don't claim a
    false "closed").
    """
    import numpy as np

    arr = np.asarray(depth, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return True
    h, w = arr.shape
    ch, cw = max(1, int(h * center_frac)), max(1, int(w * center_frac))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    region = arr[y0:y0 + ch, x0:x0 + cw]
    valid = region[np.isfinite(region) & (region > 0.0)]
    if valid.size == 0:
        return True  # no near returns in the doorway -> clear / open
    if valid.size / region.size >= min_valid_frac and float(np.median(valid)) < clear_m:
        return False  # a near surface fills the doorway -> closed
    return True


def is_door_open(ctx: "TaskContext") -> "bool | None":
    """Perception door-state check for :func:`request_open_door`'s ``is_open`` hook.

    Snapshots the depth camera and runs :func:`door_open_from_depth`. Returns
    ``True`` (open) / ``False`` (closed) / ``None`` when it can't tell (no snapshot
    or no depth — e.g. on a GPU-less box / mock ctx), so the caller falls back to a
    spoken confirmation. Reads the ``WALKIE_DOOR_*`` thresholds. Never raises.
    """
    try:
        snap = ctx.snapshot()
    except Exception as exc:  # no camera / mock ctx without snapshot()
        print(f"[skills.door] snapshot unavailable ({exc})")
        return None
    depth = getattr(snap, "depth", None) if snap is not None else None
    if depth is None:
        return None
    return door_open_from_depth(
        depth,
        clear_m=float(os.getenv("WALKIE_DOOR_CLEAR_M", "1.2")),
        center_frac=float(os.getenv("WALKIE_DOOR_CENTER_FRAC", "0.4")),
        min_valid_frac=float(os.getenv("WALKIE_DOOR_MIN_VALID_FRAC", "0.5")),
    )


def request_open_door(
    ctx: "TaskContext",
    *,
    poll_interval: Optional[float] = None,
    reprompt_every: Optional[float] = None,
    max_wait: Optional[float] = None,
    confirm_reads: Optional[int] = None,
    attempts: int = 3,
    listen_timeout: float = 15.0,
    prompt: str = DEFAULT_PROMPT,
    confirm_words: Sequence[str] = DEFAULT_CONFIRM_WORDS,
    thanks: str = DEFAULT_THANKS,
    is_open: Optional[Callable[[], bool]] = None,
) -> bool:
    """Ask a human to open the door, watch for it to open, then go in. Entry step.

    When the depth camera can see the doorway (the normal on-robot case) this asks
    **once** and then *polls the depth sensor itself* — the robot notices the door
    opening on its own and walks in, no spoken confirmation needed. It proceeds the
    instant the doorway reads clear for ``confirm_reads`` consecutive polls (a small
    debounce against a door swinging through frame), re-asks every ``reprompt_every``
    seconds, and after ``max_wait`` seconds (``<= 0`` = wait indefinitely) proceeds
    anyway so it never gets stuck outside.

    When the camera **can't tell** (no depth / mock ctx / GPU-less box) it falls back
    to the original spoken flow: ask, then listen for a ``confirm_words`` reply, up to
    ``attempts`` rounds.

    Returns True if the door was taken to be open (depth saw it clear, or a spoken
    confirmation), False if it gave up waiting and proceeded anyway — either way the
    caller should continue into the arena.

    Args:
        ctx: the task context (uses ``say`` / ``listen`` / ``snapshot``).
        poll_interval: seconds between depth checks while waiting (env
            ``WALKIE_DOOR_POLL_SEC``, default 0.5).
        reprompt_every: re-ask this often while waiting (env
            ``WALKIE_DOOR_REPROMPT_SEC``, default 15).
        max_wait: give up and proceed after this many seconds; ``<= 0`` waits
            forever (env ``WALKIE_DOOR_WAIT_SEC``, default 120).
        confirm_reads: consecutive open reads required before going in — debounce
            (env ``WALKIE_DOOR_CONFIRM_READS``, default 2).
        attempts: spoken-fallback only — ask + listen this many times.
        listen_timeout: spoken-fallback only — seconds to listen each attempt.
        prompt: the spoken request.
        confirm_words: spoken-fallback only — any in the reply counts as "open".
        thanks: spoken once the door is taken to be open.
        is_open: optional check overriding the default depth detector. When passed,
            the spoken-fallback flow is used (re-checking ``is_open()`` after each
            listen) rather than depth polling — useful for tests / a lidar probe.
    """
    poll_interval = float(os.getenv("WALKIE_DOOR_POLL_SEC", "0.5")) if poll_interval is None else poll_interval
    reprompt_every = float(os.getenv("WALKIE_DOOR_REPROMPT_SEC", "15")) if reprompt_every is None else reprompt_every
    max_wait = float(os.getenv("WALKIE_DOOR_WAIT_SEC", "120")) if max_wait is None else max_wait
    confirm_reads = int(os.getenv("WALKIE_DOOR_CONFIRM_READS", "2")) if confirm_reads is None else confirm_reads

    # Default the sensor to the built-in depth detector; it returns None (-> not
    # open -> ask) when there's no camera, so the spoken flow still works offline.
    check = is_open if is_open is not None else (lambda: is_door_open(ctx) is True)

    def opened() -> bool:
        try:
            return bool(check())
        except Exception as exc:  # pragma: no cover - robot-side sensor failure
            print(f"[skills.door] open-check failed ({exc})")
            return False

    if opened():
        return True  # already open — just go in

    # Prefer self-watching depth polling when the camera can actually tell open from
    # closed (tri-state read: not None). A custom is_open or a blind camera falls
    # through to the spoken ask-and-listen flow below.
    if is_open is None and is_door_open(ctx) is not None:
        return _wait_for_open(
            ctx,
            opened,
            poll_interval=poll_interval,
            reprompt_every=reprompt_every,
            max_wait=max_wait,
            confirm_reads=confirm_reads,
            prompt=prompt,
            thanks=thanks,
        )

    words = tuple(w.lower() for w in confirm_words)
    for i in range(max(1, attempts)):
        ctx.say(prompt if i == 0 else f"I am still waiting for the door. {prompt}")
        reply = (ctx.listen(timeout=listen_timeout) or "").lower()
        if opened() or any(w in reply for w in words):
            ctx.say(thanks)
            return True
    ctx.say("I will assume the door is open now. Thank you.")
    return False


def _wait_for_open(
    ctx: "TaskContext",
    opened: Callable[[], bool],
    *,
    poll_interval: float,
    reprompt_every: float,
    max_wait: float,
    confirm_reads: int,
    prompt: str,
    thanks: str,
) -> bool:
    """Ask once, then poll ``opened()`` until the door reads clear and go in.

    Proceeds the moment the door is seen open for ``confirm_reads`` consecutive
    polls (debounces a door swinging through frame). Re-asks every ``reprompt_every``
    seconds. After ``max_wait`` seconds (``<= 0`` = no limit) it proceeds anyway so
    the robot never gets stuck outside. Returns True on a detected open, False on the
    timeout fallback.
    """
    ctx.say(prompt)
    start = time.monotonic()
    last_prompt = start
    streak = 0
    need = max(1, confirm_reads)
    while True:
        if opened():
            streak += 1
            if streak >= need:
                ctx.say(thanks)
                return True
        else:
            streak = 0
        now = time.monotonic()
        if max_wait > 0 and (now - start) >= max_wait:
            ctx.say("I will assume the door is open now. Thank you.")
            return False
        if (now - last_prompt) >= reprompt_every:
            ctx.say("I am still waiting for the door to open.")
            last_prompt = now
        if poll_interval > 0:
            time.sleep(poll_interval)


def _dist_to(ctx: "TaskContext", x: float, y: float) -> "float | None":
    """Robot's planar distance (m) to a map-frame point, or None if pose is unknown.

    Used to tell "still stuck at the doorway" from "now driving through it". Returns
    None when ``ctx`` has no ``current_pose`` (e.g. a test/mock ctx), which the caller
    treats as "no progress" — so it degrades to asking on every retry.
    """
    pose = getattr(ctx, "current_pose", None)
    if not callable(pose):
        return None
    try:
        p = pose()
        return math.hypot(float(p["x"]) - x, float(p["y"]) - y)
    except Exception:
        return None


def go_to_through_door(
    ctx: "TaskContext",
    x: float,
    y: float,
    heading_rad: float,
    *,
    door_attempts: int = 2,
    ask_even_if_open: bool = False,
    retry_pause: Optional[float] = None,
    progress_eps: Optional[float] = None,
    **request_kwargs,
) -> bool:
    """Navigate to a map pose, asking a human to open a door in the way.

    The purpose of this whole skill: when the robot **can't reach a destination
    because a door is in the way**, it should ask for help rather than give up. So
    this is a drop-in for ``ctx.goto`` that, on a nav failure, requests the door be
    opened and retries — up to ``door_attempts`` rounds. Reusable by any challenge
    whose route crosses a (human-operated) door.

    Two failure shapes, told apart by the depth door-state read:

    * **Closed / can't tell** — run the full :func:`request_open_door` (depth
      self-watch, or the spoken fallback), then retry. A human opening the way is the
      useful action when the doorway reads blocked or unknown.
    * **Reads open but nav still failed** — by default this is taken as "nav failed
      for some other reason" and asking is skipped. But a **partly-open door** looks
      exactly like this: the central doorway reads clear (so the depth check says
      "open") yet the gap is too narrow for the robot to fit. Pass
      ``ask_even_if_open=True`` (the arena-entry setting) to treat such a block as a
      partly-open door: ask for it to be opened wider (:data:`PARTIAL_OPEN_PROMPT`),
      pause ``retry_pause`` seconds for the human, and retry — depth can't tell us it
      widened, so nav success is the only real signal.

      The "ask wider" only fires while the robot is **still stuck at the doorway**.
      Once it has advanced toward the goal by more than ``progress_eps`` metres (it's
      now driving *through* the opened door), a further nav failure is no longer a
      door problem — it just retries quietly instead of re-asking, so the operator
      isn't pestered again as the robot passes through.

    Returns True if the destination was reached, False otherwise. ``request_kwargs``
    pass through to :func:`request_open_door` (prompt, max_wait, …). ``retry_pause``
    defaults to env ``WALKIE_DOOR_RETRY_PAUSE_SEC`` (3 s); ``progress_eps`` to env
    ``WALKIE_DOOR_PROGRESS_EPS_M`` (0.3 m).
    """
    retry_pause = float(os.getenv("WALKIE_DOOR_RETRY_PAUSE_SEC", "3")) if retry_pause is None else retry_pause
    progress_eps = float(os.getenv("WALKIE_DOOR_PROGRESS_EPS_M", "0.3")) if progress_eps is None else progress_eps
    if ctx.goto(x, y, heading_rad):
        return True
    prev_dist = _dist_to(ctx, x, y)
    for _ in range(max(1, door_attempts)):
        door = is_door_open(ctx)  # True (open) / False (closed) / None (can't tell)
        if door is True and not ask_even_if_open:
            break  # positively open and not door-gated — nav failed for another reason

        cur_dist = _dist_to(ctx, x, y)
        moved_through = (
            prev_dist is not None and cur_dist is not None and (prev_dist - cur_dist) > progress_eps
        )
        prev_dist = cur_dist

        if door is not True:
            # Closed / can't tell: run the full ask-and-wait, then retry.
            print("[skills.door] cannot reach the goal — asking for the door to be opened")
            request_open_door(ctx, **request_kwargs)
        elif not moved_through:
            # Still stuck at a doorway that reads open -> partly-open door. Ask for it
            # wider, give the human a moment, then retry nav.
            print("[skills.door] nav blocked while the door reads open — assuming a "
                  "partly-open door; asking for it to be opened wider")
            ctx.say(PARTIAL_OPEN_PROMPT)
            if retry_pause > 0:
                time.sleep(retry_pause)
        else:
            # Door reads open AND the robot advanced -> it's driving through; the
            # failure isn't the door (costmap still clearing, a recovery, …). Just
            # retry without pestering the operator again.
            print("[skills.door] nav failed but the robot advanced through the doorway "
                  "— retrying without re-asking")

        if ctx.goto(x, y, heading_rad):
            return True
    return False
