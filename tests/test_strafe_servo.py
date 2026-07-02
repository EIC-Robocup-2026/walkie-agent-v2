"""Unit tests for tasks.skills.navigation.strafe_servo.

No robot — a fake ctx records every cmd_vel publish and (optionally) integrates
the last commanded body-frame strafe velocity into its odom pose, so the servo's
closed loop actually closes. ``time.sleep`` is no-oped so the loop runs at full
speed; gains come from the ``WALKIE_CREEP_*`` defaults baked into the code.
The real-gain behaviour (nav obstacle scaling, odom latency) is on-robot only.
"""

import math

import pytest

import tasks.skills.navigation as navigation
from tasks.skills.navigation import strafe_servo

HZ = 15.0
DT = 1.0 / HZ


class FakeNav:
    def __init__(self, ctx):
        self._ctx = ctx
        self.commands = []  # every (vx, vy, wz) published

    def set_velocity(self, vx, vy, wz):
        self.commands.append((vx, vy, wz))
        self._ctx._last_vy = vy
        return True


class FakeStatus:
    def __init__(self, ctx):
        self._ctx = ctx

    def get_position(self):
        ctx = self._ctx
        if ctx.no_odom:
            return None
        if ctx.integrate:  # advance by the last commanded strafe, body -> map
            h = ctx.pose["heading"]
            ctx.pose["x"] += -ctx._last_vy * math.sin(h) * DT
            ctx.pose["y"] += ctx._last_vy * math.cos(h) * DT
        pose = dict(ctx.pose)
        if ctx.drift_heading is not None and ctx.calls > 0:
            pose["heading"] = ctx.drift_heading  # h0 stays the first-call heading
        ctx.calls += 1
        return pose


class FakeHead:
    def __init__(self):
        self.auto_tilt = None

    def set_auto_tilt(self, on):
        self.auto_tilt = on


class FakeServoCtx:
    """Fake TaskContext: odom integrates the last commanded strafe velocity."""

    def __init__(self, *, integrate=True, no_odom=False, drift_heading=None):
        self.integrate = integrate
        self.no_odom = no_odom
        self.drift_heading = drift_heading
        self.pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.calls = 0
        self._last_vy = 0.0

        class _Walkie:
            pass

        self.walkie = _Walkie()
        self.walkie.nav = FakeNav(self)
        self.walkie.status = FakeStatus(self)

        class _Robot:
            pass

        self.walkie.robot = _Robot()
        self.walkie.robot.head = FakeHead()


@pytest.fixture(autouse=True)
def fast_loop(monkeypatch):
    monkeypatch.setattr(navigation.time, "sleep", lambda s: None)


def test_converges_and_stops_at_zero():
    # Base must move 0.2 m left; the fake integrates commanded vy, so the live
    # error shrinks tick by tick until it is within tolerance.
    ctx = FakeServoCtx()
    target_left = 0.2
    err = lambda: target_left - ctx.pose["y"]  # noqa: E731
    assert strafe_servo(ctx, err, tol_m=0.015, hz=HZ) is True
    assert abs(err()) <= 0.015
    assert ctx.walkie.nav.commands[-1] == (0.0, 0.0, 0.0)  # guaranteed stop
    assert ctx.walkie.robot.head.auto_tilt is True  # restored after the drive
    # every command is a pure strafe toward the target (no forward component)
    assert all(vx == 0.0 for vx, _, _ in ctx.walkie.nav.commands)
    assert all(vy >= 0.0 for _, vy, _ in ctx.walkie.nav.commands)


def test_blocked_base_stalls_out():
    # Frozen odom while commanding = nav scaled us to ~zero (obstacle) or odom
    # died: the servo must give up after stall_sec, not ride the timeout.
    ctx = FakeServoCtx(integrate=False)
    assert strafe_servo(ctx, lambda: 0.3, tol_m=0.015, hz=HZ,
                        stall_sec=0.5, timeout_sec=30.0) is False
    moving = [c for c in ctx.walkie.nav.commands if c[1] != 0.0]
    assert moving  # it did try to strafe
    assert len(moving) <= int(0.5 * HZ) + 2  # bailed on the stall guard, fast
    assert ctx.walkie.nav.commands[-1] == (0.0, 0.0, 0.0)


def test_error_fn_exception_still_stops():
    ctx = FakeServoCtx()

    def boom():
        raise RuntimeError("perception died")

    assert strafe_servo(ctx, boom, hz=HZ) is False  # never raises
    assert ctx.walkie.nav.commands[-1] == (0.0, 0.0, 0.0)
    assert ctx.walkie.robot.head.auto_tilt is True


def test_error_none_holds_still_then_gives_up():
    # None = "can't compute this tick": command zero (hold), count toward the
    # stall guard, and bail once it persists — never drive blind.
    ctx = FakeServoCtx()
    assert strafe_servo(ctx, lambda: None, hz=HZ, stall_sec=0.5) is False
    assert all(c == (0.0, 0.0, 0.0) for c in ctx.walkie.nav.commands)


def test_no_odom_at_start_refuses_to_drive():
    ctx = FakeServoCtx(no_odom=True)
    assert strafe_servo(ctx, lambda: 0.3, hz=HZ) is False
    assert ctx.walkie.nav.commands == []  # nothing commanded at all
    assert ctx.walkie.robot.head.auto_tilt is True


def test_heading_hold_counters_drift():
    # Heading drifts +0.1 rad off the start heading -> the yaw P-term must push
    # back (negative wz), scaled by the default kp_yaw=1.2.
    ctx = FakeServoCtx(integrate=False, drift_heading=0.1)
    strafe_servo(ctx, lambda: 0.3, hz=HZ, stall_sec=0.3)
    moving = [c for c in ctx.walkie.nav.commands if c[1] != 0.0]
    assert moving[-1][2] == pytest.approx(-0.12, abs=1e-6)


def test_heading_hold_yaw_is_clamped():
    # A huge drift must saturate at the WALKIE_CREEP_MAX_YAW_RPS cap (0.4).
    ctx = FakeServoCtx(integrate=False, drift_heading=2.0)
    strafe_servo(ctx, lambda: 0.3, hz=HZ, stall_sec=0.3)
    moving = [c for c in ctx.walkie.nav.commands if c[1] != 0.0]
    assert moving[-1][2] == pytest.approx(-0.4, abs=1e-6)
