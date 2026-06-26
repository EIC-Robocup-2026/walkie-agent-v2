"""Closed-loop unit tests for creep_base_relative (the Nav2-free base creep).

No robot: a fake transport records the published Twists and a fake odom either
advances along the requested direction (so the loop converges and stops) or stays
frozen (so the no-progress guard must trip). These cover the parts the grasp
geometry tests can't — the stop condition, the guaranteed finally-stop, and the
stall backstop that protects against dead/frozen odom driving the base into the
table. Hardware reachability (does cmd_vel actually move the base?) is verified
separately by manual_tests/test_base_creep.py.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tasks.skills.navigation import creep_base_relative


class FakeTransport:
    def __init__(self):
        self.msgs = []

    def publish(self, topic, msg_type, msg):
        self.msgs.append(msg)


class FakeNav:
    def __init__(self, transport):
        self._transport = transport
        self.cmd_vel_topic = "cmd_vel"


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
    return SimpleNamespace(walkie=SimpleNamespace(nav=nav, status=status))


@pytest.fixture(autouse=True)
def fast_creep(monkeypatch):
    # Tiny period + short stall window so the loop runs in milliseconds.
    monkeypatch.setenv("WALKIE_CREEP_HZ", "1000")
    monkeypatch.setenv("WALKIE_CREEP_STALL_SEC", "0.02")


def test_creep_reaches_target_then_stops():
    transport = FakeTransport()
    ctx = _ctx(FakeNav(transport), MovingStatus(ux=1.0, uy=0.0, step=0.02, target=0.20))
    assert creep_base_relative(ctx, 0.20, 0.0) is True
    # Drove forward at some point...
    assert any(m["linear"]["x"] > 0 for m in transport.msgs)
    # ...and the LAST command is a full stop (the finally guard).
    assert transport.msgs[-1]["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_creep_strafe_right_is_negative_y():
    transport = FakeTransport()
    ctx = _ctx(FakeNav(transport), MovingStatus(ux=0.0, uy=-1.0, step=0.02, target=0.15))
    assert creep_base_relative(ctx, 0.0, -0.15) is True
    moving = [m for m in transport.msgs if m["linear"] != {"x": 0.0, "y": 0.0, "z": 0.0}]
    assert moving and all(m["linear"]["y"] < 0 for m in moving)  # +left convention: right = -y
    assert all(abs(m["linear"]["x"]) < 1e-9 for m in moving)     # pure strafe, no forward


def test_creep_no_op_when_already_there():
    transport = FakeTransport()
    ctx = _ctx(FakeNav(transport), MovingStatus(ux=1.0, uy=0.0, step=0.0, target=0.0))
    assert creep_base_relative(ctx, 0.0, 0.0) is True
    assert transport.msgs == []  # below 1mm: never commands the base


def test_creep_stall_guard_trips_on_frozen_odom():
    transport = FakeTransport()
    ctx = _ctx(FakeNav(transport), FrozenStatus())
    # Odom never advances -> returns False (target not reached) and still stops.
    assert creep_base_relative(ctx, 0.30, 0.0) is False
    assert transport.msgs[-1]["linear"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    # Bailed via the stall guard, not by riding the full deadline at speed.
    assert len(transport.msgs) < 200


def test_creep_returns_false_without_odom_fix():
    transport = FakeTransport()
    ctx = _ctx(FakeNav(transport), SimpleNamespace(get_position=lambda: None))
    assert creep_base_relative(ctx, 0.20, 0.0) is False
    assert transport.msgs == []  # never started: no fix, no motion
