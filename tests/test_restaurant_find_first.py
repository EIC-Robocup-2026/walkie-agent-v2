"""Approach-on-first-sighting: find_first_caller + the _next_caller routing seam.

find_first_caller sweeps the dining area and returns the FIRST usable waving customer,
stopping the sweep the instant one appears (so the robot heads over immediately instead
of finishing the whole arc). These drive it with synthetic per-offset detections (no
robot / AI server): is_calling + the depth lift are stubbed, the fake ctx scripts what
each sweep offset "sees".
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tasks.Restaurant import skills
from tasks.Restaurant.skills import Caller


class P:
    """A fake detected person carrying the world point the lift should return."""

    def __init__(self, wxy, conf=0.7):
        self.bbox = (0.0, 0.0, 10.0, 10.0)  # cxcywh; _cxcywh_to_xyxy handles it
        self.confidence = conf
        self._wxy = wxy


class FakeCtx:
    """Scripts one person-list per sweep offset; records rotations + tilt toggles."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.idx = 0
        self.rotations: list[float] = []
        self.tilt: list[bool] = []        # set_auto_tilt(bool) calls
        self.head_tilt: list[float] = []  # head.tilt(rad) calls
        self.data: dict = {}
        self.walkie = SimpleNamespace(
            status=SimpleNamespace(get_position=lambda: {"x": 0.0, "y": 0.0, "heading": 0.0}),
            robot=SimpleNamespace(head=SimpleNamespace(
                set_auto_tilt=self.tilt.append,
                tilt=self.head_tilt.append,  # the sweep levels the head each offset
            )),
        )
        self.walkieAI = SimpleNamespace(image=SimpleNamespace(estimate_poses=self._estimate))

    def _estimate(self, _img):
        frame = self.frames[self.idx] if self.idx < len(self.frames) else []
        self.idx += 1
        return frame

    def rotate_to(self, heading):
        self.rotations.append(heading)

    def snapshot(self):
        return SimpleNamespace(img=object())


@pytest.fixture(autouse=True)
def _stub_detection(monkeypatch):
    """All fake persons are 'calling'; the lift returns each person's scripted point."""
    monkeypatch.setenv("RESTAURANT_SCAN_SETTLE_SEC", "0")  # no real dwell
    monkeypatch.setattr(skills, "is_calling", lambda p: True)
    monkeypatch.setattr(skills, "_person_world_xy", lambda _snap, p, **_k: p._wxy)


def test_returns_first_waver_and_stops_sweeping(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "90")   # angles {-45,-15,15,45}
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "30")
    # CENTER-OUT visit order is [-15, 15, -45, 45]: the waver appears at the 2nd offset
    # visited (+15°); the wider -45/45 must never be looked at.
    ctx = FakeCtx([[], [P((3.0, 0.0))], [P((9.0, 0.0))], [P((1.0, 0.0))]])

    out = skills.find_first_caller(ctx)

    assert isinstance(out, Caller) and out.world_xy == (3.0, 0.0)
    assert ctx.idx == 2                                   # stopped after the 2nd offset
    assert len(ctx.rotations) == 2                        # didn't sweep the rest...
    assert math.isclose(ctx.rotations[-1], math.radians(15))   # ...and stayed facing them
    # the sweep checks centre first, then steps out — never starts by swinging to an edge
    assert math.isclose(ctx.rotations[0], math.radians(-15))
    # head aimed to the person-look tilt (never left pointing down) before each capture
    look = skills._person_look_tilt()
    assert ctx.head_tilt == [look, look]


def test_skips_blocked_spots(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "30")   # offsets [-15, 15]
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "30")
    monkeypatch.setenv("RESTAURANT_HANDLED_RADIUS_M", "0.6")
    # First offset's waver sits on an already-handled spot -> skip; take the next.
    ctx = FakeCtx([[P((5.0, 0.0))], [P((2.0, 0.0))]])

    out = skills.find_first_caller(ctx, blocked=[(5.0, 0.0)], radius_m=0.6)

    assert out.world_xy == (2.0, 0.0)


def test_none_when_no_waver_and_recenters(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "0")    # single offset [0]
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "30")
    ctx = FakeCtx([[]])

    out = skills.find_first_caller(ctx)

    assert out is None
    assert ctx.rotations[-1] == 0.0          # left base back at the dining centre
    assert ctx.tilt[-1] is True              # head auto-tilt re-enabled in finally


def test_skips_waver_without_depth(monkeypatch):
    monkeypatch.setenv("RESTAURANT_SCAN_ARC_DEG", "0")
    monkeypatch.setenv("RESTAURANT_SCAN_STEP_DEG", "30")
    ctx = FakeCtx([[P(None)]])               # waving, but the lift can't place them

    assert skills.find_first_caller(ctx) is None


def test_next_caller_routes_on_the_flag(monkeypatch):
    """_next_caller routing: LIVE_SCAN=1 -> live_scan; else APPROACH_FIRST picks the
    old discrete path (find_first_caller vs scan+nearest)."""
    from tasks.Restaurant import subtasks

    hits = {"live": 0, "first": 0, "scan": 0}
    live_sentinel = object()
    first_sentinel = object()
    monkeypatch.setattr(subtasks, "live_scan_for_caller",
                        lambda ctx, b, r: hits.__setitem__("live", hits["live"] + 1) or live_sentinel)
    monkeypatch.setattr(subtasks, "find_first_caller",
                        lambda ctx, b, r: hits.__setitem__("first", hits["first"] + 1) or first_sentinel)
    monkeypatch.setattr(subtasks, "scan_for_callers",
                        lambda ctx: hits.__setitem__("scan", hits["scan"] + 1) or [])
    monkeypatch.setattr(subtasks, "nearest_caller", lambda ctx, callers: None)
    monkeypatch.setattr(subtasks, "exclude_handled", lambda callers, b, r: callers)

    # Live path is the default.
    monkeypatch.setenv("RESTAURANT_LIVE_SCAN", "1")
    assert subtasks._next_caller(object(), [], 0.6) is live_sentinel
    assert hits == {"live": 1, "first": 0, "scan": 0}

    # Old path, approach-on-first.
    monkeypatch.setenv("RESTAURANT_LIVE_SCAN", "0")
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "1")
    assert subtasks._next_caller(object(), [], 0.6) is first_sentinel
    assert hits == {"live": 1, "first": 1, "scan": 0}

    # Old path, full-sweep + nearest.
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "0")
    subtasks._next_caller(object(), [], 0.6)
    assert hits == {"live": 1, "first": 1, "scan": 1}
