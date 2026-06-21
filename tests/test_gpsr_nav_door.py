"""GPSR navigation routes through the closed-door skill when enabled.

go_to_named should use tasks.skills.go_to_through_door (ask-for-a-closed-door +
retry) when GPSR_NAV_DOOR_RETRY is on, and fall back to a plain ctx.goto when off.
The door flow itself is tested in test_skills_door.py; here we only assert the
routing + the gate.
"""

from __future__ import annotations

from tasks.GPSR.skills import go_to_named


class _World:
    def location_pose(self, name):
        return (1.0, 2.0, 0.0)


def _ctx(calls):
    return type("_C", (), {"goto": lambda self, *a: (calls.append(("plain", a)), True)[1]})()


def test_routes_through_door_skill_when_enabled(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "1")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p: (calls.append(("door", p)), True)[1])
    assert go_to_named(_ctx(calls), "kitchen", _World(), {}) is True
    assert calls == [("door", (1.0, 2.0, 0.0))]  # went via the door-aware nav


def test_plain_goto_when_disabled(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "0")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p: (calls.append(("door", p)), True)[1])
    assert go_to_named(_ctx(calls), "kitchen", _World(), {}) is True
    assert calls == [("plain", (1.0, 2.0, 0.0))]  # door skill not used


def test_already_here_skips_navigation(monkeypatch):
    monkeypatch.setenv("GPSR_NAV_DOOR_RETRY", "1")
    calls: list = []
    monkeypatch.setattr("tasks.GPSR.skills.go_to_through_door",
                        lambda ctx, *p: (calls.append(("door", p)), True)[1])
    # state says we're already at "kitchen" -> no drive at all (door skill untouched)
    assert go_to_named(_ctx(calls), "kitchen", _World(), {"at": "kitchen"}) is True
    assert calls == []
