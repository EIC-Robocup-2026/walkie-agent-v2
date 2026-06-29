"""Stall guard in approach_customer (tasks/Restaurant/skills.py).

When the robot steps toward a waving customer but can't make headway (path blocked
by chairs / a narrow aisle / no reachable free spot), it should STOP and take the
order from where it stands — not burn every step and then abandon the customer.
These lock that behaviour: per-step base displacement < RESTAURANT_APPROACH_MIN_
PROGRESS_M counts as a stall; RESTAURANT_APPROACH_MAX_STALLS consecutive stalls end
the approach with a True ("take the order here") result.

The heavy helpers (nav drive, re-detect, facing) are patched out; only the pure
step/stall control flow is exercised, driven by a scripted per-step displacement.
"""

from __future__ import annotations

import math

import pytest

from tasks.Restaurant import skills

CUSTOMER = (10.0, 0.0)  # far to the +x; robot starts at the origin


class _Status:
    def __init__(self, pose: dict):
        self._pose = pose

    def get_position(self) -> dict:
        return dict(self._pose)  # _robot_pose only reads x/y/heading


class _Walkie:
    def __init__(self, pose: dict):
        self.status = _Status(pose)


class FakeCtx:
    """Minimal TaskContext stand-in: a mutable base pose the fake drive moves."""

    def __init__(self, start=(0.0, 0.0)):
        self.pose = {"x": start[0], "y": start[1], "heading": 0.0}
        self.walkie = _Walkie(self.pose)  # status reads the SAME dict we mutate
        self.data: dict = {}
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


@pytest.fixture
def patched(monkeypatch):
    """Patch approach_to_standoff (scripted displacement), find_person_near (no
    re-acquire), and face_person (record calls). Returns a builder."""

    def build(moves: list[float], **env):
        env.setdefault("RESTAURANT_STANDOFF_M", "0.8")
        env.setdefault("RESTAURANT_APPROACH_MIN_PROGRESS_M", "0.2")
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))

        calls = {"n": 0}

        def fake_standoff(ctx, world_xy, *, standoff_m=None):
            i = calls["n"]
            calls["n"] += 1
            ctx.pose["x"] += moves[i] if i < len(moves) else 0.0  # toward +x customer
            return True

        faces: list[tuple[float, float]] = []
        monkeypatch.setattr(skills, "approach_to_standoff", fake_standoff)
        monkeypatch.setattr(skills, "find_person_near", lambda *a, **k: None)
        monkeypatch.setattr(skills, "face_person", lambda ctx, xy, **k: faces.append(xy) or True)
        return calls, faces

    return build


def test_stall_stops_and_takes_order_from_here(patched):
    """One stalled step (max_stalls=1) -> stop and take the order (True), face customer."""
    calls, faces = patched([0.0], RESTAURANT_APPROACH_MAX_STALLS="1")
    ctx = FakeCtx()
    assert skills.approach_customer(ctx, CUSTOMER) is True
    assert calls["n"] == 1          # stopped after the first (stalled) step
    assert faces == [CUSTOMER]      # faced the customer before returning to take the order


def test_single_stall_tolerated_when_max_is_two(patched):
    """max_stalls=2: a lone stall doesn't stop; the next step makes headway and reaches."""
    # step1: no movement (stall 1/2); step2: jump to within the stand-off -> close enough.
    calls, faces = patched([0.0, 9.0], RESTAURANT_APPROACH_MAX_STALLS="2")
    ctx = FakeCtx()
    assert skills.approach_customer(ctx, CUSTOMER) is True
    assert faces == []              # never hit the stall-stop branch
    assert calls["n"] == 3          # step1 + step2 drives, then the final close-enough park
    assert math.isclose(ctx.pose["x"], 9.0)


def test_normal_progress_reaches_standoff_without_stalling(patched):
    """Steady headway each step -> reaches the stand-off normally, no stall stop."""
    calls, faces = patched([8.0, 1.5], RESTAURANT_APPROACH_MAX_STALLS="1")
    ctx = FakeCtx()
    assert skills.approach_customer(ctx, CUSTOMER) is True
    assert faces == []
    assert calls["n"] == 3          # two stepping drives + final park, no early stop


def test_two_consecutive_stalls_stop_when_max_is_two(patched):
    """max_stalls=2: two stalls in a row end the approach with a take-order-here True."""
    calls, faces = patched([0.0, 0.0], RESTAURANT_APPROACH_MAX_STALLS="2")
    ctx = FakeCtx()
    assert skills.approach_customer(ctx, CUSTOMER) is True
    assert calls["n"] == 2          # stopped on the second consecutive stall
    assert faces == [CUSTOMER]


def test_announces_when_starting_to_approach(patched):
    """The robot speaks the 'I see you' line as it starts heading to the customer."""
    from tasks.Restaurant import prompts

    patched([0.0], RESTAURANT_APPROACH_MAX_STALLS="1")
    ctx = FakeCtx()
    skills.approach_customer(ctx, CUSTOMER)
    assert prompts.FOUND_CUSTOMER in ctx.said
