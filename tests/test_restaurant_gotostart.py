"""GoToStart is decoupled from the shared pre-map (rulebook 5.5).

Restaurant must NOT consult the LocationBook: a ``kitchen_bar`` waypoint in the
shared ``world.toml`` (e.g. defined for GPSR) must never make the robot drive.
The only way to drive is an explicit ``RESTAURANT_KITCHEN_BAR_POSE`` env pose; the
default (``"current"``/unset) anchors on the robot's current pose and stays put.

These lock the decoupling: if anyone re-wires GoToStart back through the map, the
first test fails loudly (the patched ``get_location_book`` raises on any call).
"""

from __future__ import annotations

from tasks.base import StepResult
from tasks.Restaurant.subtasks import GoToStart


class _FakeStatus:
    def __init__(self, fix):
        self._fix = fix

    def get_position(self):
        return self._fix


class _FakeWalkie:
    def __init__(self, fix):
        self.status = _FakeStatus(fix)


class FakeCtx:
    """Minimal TaskContext stand-in: just the surface GoToStart touches."""

    def __init__(self, fix):
        self.data: dict = {}
        self.walkie = _FakeWalkie(fix)
        self.goto_calls: list[tuple[float, float, float]] = []

    def goto(self, x, y, h):
        self.goto_calls.append((x, y, h))
        return True


_FIX = {"x": 4.0, "y": 5.0, "heading": 0.25}


def _explode():
    raise AssertionError("GoToStart must not consult the shared LocationBook")


def test_default_anchors_on_current_pose_and_never_touches_the_map(monkeypatch):
    """Unset env -> anchor on the current pose, don't drive, never read the map."""
    monkeypatch.delenv("RESTAURANT_KITCHEN_BAR_POSE", raising=False)
    # If GoToStart consulted the shared book at all, this would blow up.
    monkeypatch.setattr("walkie_world.map.locations.get_location_book", _explode)

    ctx = FakeCtx(_FIX)
    assert GoToStart().run(ctx) is StepResult.DONE
    assert ctx.goto_calls == []                       # stayed put
    assert ctx.data["bar_anchor"] == {"x": 4.0, "y": 5.0, "heading": 0.25}


def test_kitchen_bar_waypoint_in_the_map_is_ignored(monkeypatch):
    """A kitchen_bar entry in the shared map must NOT make Restaurant drive."""
    monkeypatch.delenv("RESTAURANT_KITCHEN_BAR_POSE", raising=False)

    class _BookWithBar:
        def has(self, name):
            return name == "kitchen_bar"

        def pose(self, name):
            return (9.0, 9.0, 9.0)

    monkeypatch.setattr(
        "walkie_world.map.locations.get_location_book", lambda: _BookWithBar()
    )

    ctx = FakeCtx(_FIX)
    assert GoToStart().run(ctx) is StepResult.DONE
    assert ctx.goto_calls == []                       # ignored the map; stayed put
    assert ctx.data["bar_anchor"] == {"x": 4.0, "y": 5.0, "heading": 0.25}


def test_current_sentinel_anchors_in_place(monkeypatch):
    monkeypatch.setenv("RESTAURANT_KITCHEN_BAR_POSE", "current")
    ctx = FakeCtx(_FIX)
    assert GoToStart().run(ctx) is StepResult.DONE
    assert ctx.goto_calls == []


def test_explicit_env_pose_drives_then_anchors_on_reached_pose(monkeypatch):
    monkeypatch.setenv("RESTAURANT_KITCHEN_BAR_POSE", "1.5,2.0,1.57")
    ctx = FakeCtx(_FIX)
    assert GoToStart().run(ctx) is StepResult.DONE
    assert ctx.goto_calls == [(1.5, 2.0, 1.57)]       # drove to the env pose
    # Anchors on the pose actually reached (a genuine fix is available).
    assert ctx.data["bar_anchor"] == {"x": 4.0, "y": 5.0, "heading": 0.25}


def test_no_odometry_fix_retries(monkeypatch):
    monkeypatch.delenv("RESTAURANT_KITCHEN_BAR_POSE", raising=False)
    ctx = FakeCtx(None)                               # no genuine odometry fix
    assert GoToStart().run(ctx) is StepResult.RETRY
    assert "bar_anchor" not in ctx.data
