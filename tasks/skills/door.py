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

import os
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
    attempts: int = 3,
    listen_timeout: float = 15.0,
    prompt: str = DEFAULT_PROMPT,
    confirm_words: Sequence[str] = DEFAULT_CONFIRM_WORDS,
    thanks: str = DEFAULT_THANKS,
    is_open: Optional[Callable[[], bool]] = None,
) -> bool:
    """Ask a human to open the door and wait until it's open. Reusable entry step.

    Returns True if the door was confirmed open (by ``is_open()`` or a spoken
    confirmation), False if it gave up waiting and proceeded anyway — either way
    the caller should continue into the arena.

    Args:
        ctx: the task context (uses ``say`` / ``listen`` only).
        attempts: how many times to ask + listen before proceeding regardless.
        listen_timeout: seconds to listen for a confirmation each attempt.
        prompt: the spoken request.
        confirm_words: any of these in the reply counts as "it's open".
        thanks: spoken once the door is taken to be open.
        is_open: optional check overriding the default. When omitted, the built-in
            depth clear-path detector (:func:`is_door_open`) is used — so callers
            get automatic open/closed detection for free, and it degrades to the
            spoken flow when there's no camera. Pass a callable to override (e.g. a
            lidar test), or ``lambda: False`` to force the ask-only behaviour.
    """
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
    words = tuple(w.lower() for w in confirm_words)
    for i in range(max(1, attempts)):
        ctx.say(prompt if i == 0 else f"I am still waiting for the door. {prompt}")
        reply = (ctx.listen(timeout=listen_timeout) or "").lower()
        if opened() or any(w in reply for w in words):
            ctx.say(thanks)
            return True
    ctx.say("I will assume the door is open now. Thank you.")
    return False


def go_to_through_door(
    ctx: "TaskContext",
    x: float,
    y: float,
    heading_rad: float,
    *,
    door_attempts: int = 2,
    **request_kwargs,
) -> bool:
    """Navigate to a map pose, asking a human to open a **closed door** in the way.

    The purpose of this whole skill: when the robot **can't reach a destination
    because a door is shut**, it should ask for help rather than give up. So this is
    a drop-in for ``ctx.goto`` that, on a nav failure, requests the door be opened
    and retries — up to ``door_attempts`` rounds. It only skips asking when the
    depth check positively reports the door **open** (then the nav failed for some
    other reason and asking wouldn't help); a "closed" or "can't tell" reading still
    asks, since a human opening the way is the useful action when blocked. Reusable
    by any challenge whose route crosses a (human-operated) door.

    Returns True if the destination was reached, False otherwise. ``request_kwargs``
    pass through to :func:`request_open_door` (prompt, listen_timeout, …).
    """
    if ctx.goto(x, y, heading_rad):
        return True
    for _ in range(max(1, door_attempts)):
        if is_door_open(ctx) is True:
            break  # the door is open — nav failed for another reason; don't ask
        print("[skills.door] cannot reach the goal — asking for the door to be opened")
        request_open_door(ctx, **request_kwargs)
        if ctx.goto(x, y, heading_rad):
            return True
    return False
