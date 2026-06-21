"""GPSR navigation routes through the closed-door skill when enabled.

go_to_named should use tasks.skills.go_to_through_door (ask-for-a-closed-door +
retry) when GPSR_NAV_DOOR_RETRY is on, and fall back to a plain ctx.goto when off.
The door flow itself is tested in test_skills_door.py; here we only assert the
routing + the gate.
"""

from __future__ import annotations

from tasks.GPSR.skills import go_to_named


class _World:
    def __init__(self, barrier=False):
        self._barrier = barrier

    def location_pose(self, name):
        return (1.0, 2.0, 0.0)

    def is_barrier(self, name):
        return self._barrier


def _ctx(calls):
    return type("_C", (), {"goto": lambda self, *a: (calls.append(("plain", a)), True)[1]})()


def test_routes_through_door_skill_when_enabled(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "1")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p, **kw: (calls.append(("door", p, kw)), True)[1])
    assert go_to_named(_ctx(calls), "kitchen", _World(), {}) is True
    # went via the door-aware nav; a non-barrier place doesn't force the open-ask
    assert calls == [("door", (1.0, 2.0, 0.0), {"ask_even_if_open": False})]


def test_barrier_place_asks_even_when_door_reads_open(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "1")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p, **kw: (calls.append(kw), True)[1])
    # a place flagged barrier=true (partition/screen) -> ask on the block even if depth says open
    assert go_to_named(_ctx(calls), "living_room", _World(barrier=True), {}) is True
    assert calls == [{"ask_even_if_open": True}]


def test_plain_goto_when_disabled(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "0")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p, **kw: (calls.append(("door", p, kw)), True)[1])
    assert go_to_named(_ctx(calls), "kitchen", _World(), {}) is True
    assert calls == [("plain", (1.0, 2.0, 0.0))]  # door skill not used


def test_already_here_skips_navigation(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "1")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p, **kw: (calls.append(("door", p, kw)), True)[1])
    # state says we're already at "kitchen" -> no drive at all (door skill untouched)
    assert go_to_named(_ctx(calls), "kitchen", _World(), {"at": "kitchen"}) is True
    assert calls == []
