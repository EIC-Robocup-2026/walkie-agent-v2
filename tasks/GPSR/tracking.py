"""Shared "are we there / is the person still with me" helpers for follow + guide.

`follow` ("follow me to X") and `guide` ("lead the person to X") are the same gap
at heart: a person and the robot move together toward a place, and the skill needs
to know when they have ARRIVED (so follow stops there instead of timing out) and
whether the person is STILL ALONG (so guide doesn't arrive alone). Rather than
solve it twice, both skills share this module.

The decision logic — arrival distance, companion presence — is pure / mockable and
offline-tested. `ArrivalStopper` is the thin threaded glue that turns "arrived"
into the stop signal `tasks.skills.follow_person` already polls (`.triggered`,
mirroring `CommandListener`); like the follow loop itself it is verified on the
robot.
"""

from __future__ import annotations

import math
import os
import threading

from tasks.base import TaskContext

Point = tuple[float, float]


def within(a: Point, b: Point, radius: float) -> bool:
    """True if planar point *a* is within *radius* metres of *b*. Pure."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= radius


def robot_xy(ctx: TaskContext) -> Point:
    """The robot's current map-frame (x, y). Zeros if odometry has no fix."""
    p = ctx.current_pose()
    return (p["x"], p["y"])


def arrived(ctx: TaskContext, target: Point, radius: float) -> bool:
    """True once the robot is within *radius* of *target*."""
    return within(robot_xy(ctx), target, radius)


def companion_present(ctx: TaskContext) -> bool:
    """True if at least one person is currently visible in the forward frame.

    The building block for ``guide``'s **mid-route re-acquire** (the open follow-
    back TODO): while leading, the robot periodically pauses and looks back at the
    trailing follower. It is deliberately NOT used at a guide *arrival* — there the
    robot faces the destination with its back to the person, so the forward frame
    can't see them and this would false-negative on a compliant follower. False on
    no detection or camera failure.
    """
    snap = ctx.snapshot()
    if snap is None:
        return False
    try:
        return bool(ctx.walkieAI.image.estimate_poses(snap.img))
    except Exception as exc:  # pragma: no cover - robot-side failure path
        print(f"[gpsr.tracking] companion check failed ({exc})")
        return False


def _arrival_radius() -> float:
    return float(os.getenv("GPSR_ARRIVAL_RADIUS_M", "1.0"))


def _poll_sec() -> float:
    return float(os.getenv("GPSR_ARRIVAL_POLL_SEC", "0.5"))


class ArrivalStopper:
    """A ``follow_person`` *stopper*: a context manager whose ``.triggered`` Event
    is set once the robot reaches *target* (within ``GPSR_ARRIVAL_RADIUS_M``).

    Mirrors ``tasks.skills.CommandListener``'s protocol (context manager +
    ``.triggered``) so ``follow_person`` ends the follow on arrival — returning
    ``'stopped'`` — instead of running to ``HRI_FOLLOW_TIMEOUT_SEC``. A daemon
    poll thread reads ``ctx.current_pose()``; ``__exit__`` stops and joins it.
    """

    def __init__(self, ctx: TaskContext, target: Point, *, radius: float | None = None,
                 poll_sec: float | None = None) -> None:
        self.ctx = ctx
        self.target = target
        self.radius = _arrival_radius() if radius is None else radius
        self.poll_sec = _poll_sec() if poll_sec is None else poll_sec
        self.triggered = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ArrivalStopper":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                if arrived(self.ctx, self.target, self.radius):
                    self.triggered.set()
                    return
            except Exception as exc:  # pragma: no cover - robot-side failure path
                print(f"[gpsr.tracking] arrival poll failed ({exc})")
            self._stop.wait(self.poll_sec)  # interruptible sleep
