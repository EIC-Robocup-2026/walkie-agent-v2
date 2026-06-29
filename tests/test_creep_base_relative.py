"""Closed-loop unit tests for creep_base_relative (the Nav2-free base creep).

No robot: a fake nav records the ``set_velocity(vx, vy, wz)`` commands and a fake odom
either advances along the requested direction (so the loop converges and stops) or stays
frozen (so the no-progress guard must trip). These cover the parts the grasp geometry
tests can't — the stop condition, the guaranteed finally-stop, and the stall backstop that
protects against dead/frozen odom driving the base into the table. Hardware reachability
(does cmd_vel actually move the base?) is verified separately by
manual_tests/test_base_creep.py.

The base is driven via the SDK's ``nav.set_velocity`` (which builds the correct
geometry_msgs/msg/TwistStamped); a raw plain-Twist publish is malformed for this base.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tasks.skills.navigation import creep_base_relative


class FakeNav:
    """Records every set_velocity(vx, vy, wz) the creep issues, as (vx, vy, wz) tuples."""

    def __init__(self):
        self.cmds: list[tuple[float, float, float]] = []

    def set_velocity(self, vx, vy, wz):
        self.cmds.append((float(vx), float(vy), float(wz)))
        return True

    def stop(self):
        self.cmds.append((0.0, 0.0, 0.0))
        return True


class MovingStatus:
    """Odom that advances `step` m per poll along (ux, uy) up to `target` (heading 0)."""

    def __init__(self, ux, uy, step, target):
        self.ux, self.uy, self.step, self.target = ux, uy, step, target
        self.travelled = 0.0

    def get_position(self):
        pos = {"x": self.ux * self.travelled, "y": self.uy * self.travelled, "heading": 0.0}
        self.travelled = min(self.target, self.travelled + self.step)
        return pos


class FrozenStatus:
    """Odom that never moves — simulates a stuck base or dead odom feed."""

    def get_position(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


def _ctx(nav, status):
    # creep_base_relative disables head auto-tilt before the blind cmd_vel dock, so the
    # fake walkie must expose robot.head.set_auto_tilt (no-op here).
    head = SimpleNamespace(set_auto_tilt=lambda *_a, **_k: None)
    robot = SimpleNamespace(head=head)
    return SimpleNamespace(walkie=SimpleNamespace(nav=nav, status=status, robot=robot))


@pytest.fixture(autouse=True)
def fast_creep(monkeypatch):
    # Tiny period + short stall window so the loop runs in milliseconds.
    monkeypatch.setenv("WALKIE_CREEP_HZ", "1000")
    monkeypatch.setenv("WALKIE_CREEP_STALL_SEC", "0.02")


def test_creep_reaches_target_then_stops():
    nav = FakeNav()
    ctx = _ctx(nav, MovingStatus(ux=1.0, uy=0.0, step=0.02, target=0.20))
    assert creep_base_relative(ctx, 0.20, 0.0) is True
    # Drove forward at some point...
    assert any(vx > 0 for (vx, vy, wz) in nav.cmds)
    # ...and the LAST command is a full stop (the finally guard).
    assert nav.cmds[-1] == (0.0, 0.0, 0.0)


def test_creep_strafe_right_is_negative_y():
    nav = FakeNav()
    ctx = _ctx(nav, MovingStatus(ux=0.0, uy=-1.0, step=0.02, target=0.15))
    assert creep_base_relative(ctx, 0.0, -0.15) is True
    moving = [c for c in nav.cmds if c != (0.0, 0.0, 0.0)]
    assert moving and all(vy < 0 for (vx, vy, wz) in moving)  # +left convention: right = -y
    assert all(abs(vx) < 1e-9 for (vx, vy, wz) in moving)     # pure strafe, no forward


def test_creep_no_op_when_already_there():
    nav = FakeNav()
    ctx = _ctx(nav, MovingStatus(ux=1.0, uy=0.0, step=0.0, target=0.0))
    assert creep_base_relative(ctx, 0.0, 0.0) is True
    assert nav.cmds == []  # below 1mm: never commands the base


def test_creep_stall_guard_trips_on_frozen_odom():
    nav = FakeNav()
    ctx = _ctx(nav, FrozenStatus())
    # Odom never advances -> returns False (target not reached) and still stops.
    assert creep_base_relative(ctx, 0.30, 0.0) is False
    assert nav.cmds[-1] == (0.0, 0.0, 0.0)
    # Bailed via the stall guard, not by riding the full deadline at speed.
    assert len(nav.cmds) < 200


def test_creep_returns_false_without_odom_fix():
    nav = FakeNav()
    ctx = _ctx(nav, SimpleNamespace(get_position=lambda: None))
    assert creep_base_relative(ctx, 0.20, 0.0) is False
    assert nav.cmds == []  # never started: no fix, no motion
