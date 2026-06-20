"""Ask a human to open the arena door, then proceed — a reusable challenge-entry
primitive.

Most RoboCup@Home runs start with the robot OUTSIDE the arena; it may enter only
once a (usually human-operated) door is opened. That's the same across every
challenge, so it lives in the shared skills package rather than in one task:

    from tasks.skills import request_open_door

The skill is prompt-driven and sensor-optional. It speaks a polite request and
waits, deciding the door is open by — in order — an optional ``is_open()`` check
(e.g. a clear-path / lidar test a challenge can plug in), a spoken confirmation
heard on the mic, or, after the budget runs out, proceeding anyway (the robot has
to get in). It's pure control flow over ``ctx.say`` / ``ctx.listen``, so it's
offline-testable with a mock context and pulls in no hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

if TYPE_CHECKING:  # type-only — keeps this skill importable on a GPU-less box
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
        is_open: optional check (e.g. a clear-path / lidar test); when it returns
            True the wait ends immediately, no spoken confirmation needed. A
            challenge with a real door sensor plugs it in here.
    """

    def opened() -> bool:
        if is_open is None:
            return False
        try:
            return bool(is_open())
        except Exception as exc:  # pragma: no cover - robot-side sensor failure
            print(f"[skills.door] is_open check failed ({exc})")
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
