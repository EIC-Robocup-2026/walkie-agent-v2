"""Offline tests for the shared follow/guide tracking helpers (tracking.py).

The decision logic (arrival distance, companion presence) is pure / mockable; the
ArrivalStopper thread is exercised with a fake ctx and short poll so the test
asserts on its `.triggered` Event (no sleep-timing assertions).
"""

from __future__ import annotations

from tasks.GPSR.tracking import (
    ArrivalStopper,
    arrived,
    companion_present,
    robot_xy,
    within,
)


class _FakeImage:
    def __init__(self, people):
        self.people = people

    def estimate_poses(self, img):
        return list(self.people)


class _FakeAI:
    def __init__(self, people):
        self.image = _FakeImage(people)


class _FakeSnap:
    img = "frame"


class _Ctx:
    def __init__(self, *, pose=(0.0, 0.0), people=(), snap=True):
        self._pose = pose
        self.walkieAI = _FakeAI(list(people))
        self._snap = _FakeSnap() if snap else None

    def current_pose(self):
        return {"x": self._pose[0], "y": self._pose[1], "heading": 0.0}

    def snapshot(self):
        return self._snap


def test_within_is_a_radius_check():
    assert within((0.0, 0.0), (0.5, 0.5), 1.0)
    assert within((1.0, 2.0), (1.0, 2.0), 0.0)  # exactly on the point
    assert not within((0.0, 0.0), (2.0, 0.0), 1.0)


def test_robot_xy_and_arrived():
    ctx = _Ctx(pose=(3.0, 4.0))
    assert robot_xy(ctx) == (3.0, 4.0)
    assert arrived(ctx, (3.5, 4.0), 1.0)
    assert not arrived(ctx, (10.0, 10.0), 1.0)


def test_companion_present():
    assert companion_present(_Ctx(people=["p"]))
    assert not companion_present(_Ctx(people=[]))
    assert not companion_present(_Ctx(snap=False))  # no camera frame


def test_arrival_stopper_triggers_once_at_target():
    ctx = _Ctx(pose=(1.0, 1.0))  # already within radius of the target
    with ArrivalStopper(ctx, (1.0, 1.0), radius=0.5, poll_sec=0.01) as st:
        assert st.triggered.wait(2.0)  # fires promptly since we are at the target


def test_arrival_stopper_stays_silent_when_far():
    ctx = _Ctx(pose=(10.0, 10.0))  # far from the target
    with ArrivalStopper(ctx, (0.0, 0.0), radius=0.5, poll_sec=0.01) as st:
        assert not st.triggered.wait(0.1)  # never within radius -> never set
