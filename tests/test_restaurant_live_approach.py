"""Offline smoke tests for live_approach_caller's control flow (no robot/network).

live_approach_caller is ~200 lines of stateful loop — non-blocking drive, velocity-stall
timer, lost-in-view timer, interleaved appearance capture, arrival geometry. These tests
stub the perception/nav SURFACE (a fake walkie + the heavy module-level skills it calls)
and assert the loop's control flow: that a customer in view is reached and reported with a
caption + refined point, and that a customer that's never seen is given up (return to bar).
They don't validate the geometry — that's the pure-helper suite — only that the branches
wire up and don't raise.
"""

from __future__ import annotations

import pytest

from tasks.Restaurant import skills
from tasks.Restaurant.skills import Caller, live_approach_caller


class _Nav:
    """Fake nav: is_navigating walks a scripted list of bools (then False)."""

    def __init__(self, nav_states):
        self._states = list(nav_states)
        self.go_to_calls = []
        self.cancelled = 0
        self.stopped = 0

    @property
    def is_navigating(self):
        return self._states.pop(0) if self._states else False

    def go_to(self, **kw):
        self.go_to_calls.append(kw)
        return "IN_PROGRESS"

    def cancel(self):
        self.cancelled += 1
        return True

    def stop(self):
        self.stopped += 1
        return True


class _Status:
    def __init__(self, linear):
        self._linear = linear

    def get_velocity(self):
        return {"linear": self._linear, "angular": 0.0}

    def get_position(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


class _Head:
    def set_auto_tilt(self, *a, **k):
        return True

    def tilt(self, *a, **k):
        return None


class _Walkie:
    def __init__(self, nav, status):
        self.nav = nav
        self.status = status
        self.robot = type("R", (), {"head": _Head()})()


class FakeCtx:
    def __init__(self, walkie):
        self.walkie = walkie
        self.said: list[str] = []
        self.data: dict = {}

    def say(self, text):
        self.said.append(text)

    def snapshot(self):
        return object()

    def current_pose(self):
        return {"x": 0.0, "y": 0.0, "heading": 0.0}


def _stub_perception(monkeypatch, *, seen: bool, caption="a person in a red shirt"):
    """Stub the heavy module-level skills live_approach_caller calls."""
    found = (Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9), object(), (0, 0, 10, 10))
    monkeypatch.setattr(skills, "_detect_caller_near",
                        lambda ctx, target, bearing, **kw: found if seen else (None, object(), None))
    monkeypatch.setattr(skills, "describe_customer", lambda ctx, snap, box: caption)
    monkeypatch.setattr(skills, "find_person_near",
                        lambda ctx, xy, **kw: Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9))
    monkeypatch.setattr(skills, "face_person", lambda ctx, xy, **kw: True)
    monkeypatch.setattr(skills, "capture_appearance", lambda ctx, xy: None)


def test_live_approach_reaches_a_visible_customer(monkeypatch):
    """In view + nav settles within tolerance -> (True, caption, refined_xy)."""
    _stub_perception(monkeypatch, seen=True)
    nav = _Nav([False])  # nav already settled -> arrival geometry check passes (1m <= 0.8+0.4)
    ctx = FakeCtx(_Walkie(nav, _Status(linear=0.2)))  # moving -> no stall
    caller = Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9)

    reached, caption, final_xy = live_approach_caller(ctx, caller)

    assert reached is True
    assert caption == "a person in a red shirt"
    assert final_xy == (1.0, 0.0)
    assert nav.go_to_calls  # issued the non-blocking drive
    assert nav.go_to_calls[0]["align_method"] == "face_target"
    assert any("coming over" in s for s in ctx.said)  # FOUND_CUSTOMER spoken


def test_live_approach_gives_up_and_returns_to_bar_when_never_seen(monkeypatch):
    """Customer out of view past LOST_SEC, re-scan finds nobody -> (False, None, None)."""
    _stub_perception(monkeypatch, seen=False)
    monkeypatch.setattr(skills, "_lost_recover", lambda ctx, caller: None)
    rtb = []
    monkeypatch.setattr(skills, "return_to_bar", lambda ctx, **kw: rtb.append(kw) or True)
    monkeypatch.setenv("RESTAURANT_LOST_SEC", "0")  # trip the lost timer immediately
    nav = _Nav([True, True, True])  # still driving
    ctx = FakeCtx(_Walkie(nav, _Status(linear=0.2)))
    caller = Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9)

    reached, caption, final_xy = live_approach_caller(ctx, caller)

    assert reached is False
    assert caption is None and final_xy is None
    assert rtb  # went back to the bar
    assert nav.cancelled >= 1
    assert any("lost sight" in s for s in ctx.said)


def test_live_approach_stall_in_view_serves_from_here(monkeypatch):
    """Base barely moving past LOW_SPEED_SEC but customer in view -> reachable (True)."""
    _stub_perception(monkeypatch, seen=True)
    monkeypatch.setenv("RESTAURANT_APPROACH_LOW_SPEED_SEC", "0")  # trip the stall timer at once
    nav = _Nav([True, True, True, True])  # still "navigating" (never settles on its own)
    ctx = FakeCtx(_Walkie(nav, _Status(linear=0.0)))  # stalled
    caller = Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9)

    reached, caption, final_xy = live_approach_caller(ctx, caller)

    assert reached is True            # stalled but in view -> serve from here
    assert nav.cancelled >= 1         # cancelled the stalled goal
    assert final_xy == (1.0, 0.0)


def test_live_scan_rotates_via_go_to_and_returns_first_waver(monkeypatch):
    """The sweep rotates with ctx.rotate_to (go_to), detects a waver, announces, returns it.

    Guards against the regression where the loop published to cmd_vel (which doesn't move
    this base) and issued NO rotation. We assert rotate_to was actually called."""
    from types import SimpleNamespace

    monkeypatch.setattr(skills, "is_calling", lambda p, **k: True)
    monkeypatch.setattr(skills, "_torso_center_u", lambda p, **k: 320.0)
    monkeypatch.setattr(skills, "_bearing_from_pixel_u", lambda u, snap, **k: 0.1)
    monkeypatch.setattr(skills, "_person_world_xy", lambda snap, p, **k: (2.0, 0.0))
    monkeypatch.setattr(skills, "_aim_head_for_people", lambda ctx: None)
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "1")
    monkeypatch.setenv("RESTAURANT_SCAN_SETTLE_SEC", "0")

    person = SimpleNamespace(bbox=(320.0, 240.0, 40.0, 80.0), confidence=0.9, keypoints=[])
    rotates: list[float] = []

    class _ScanCtx:
        def __init__(self):
            self.data: dict = {}
            self.said: list[str] = []
            self.walkie = SimpleNamespace(
                status=SimpleNamespace(get_position=lambda: {"x": 0.0, "y": 0.0, "heading": 0.0}),
                robot=SimpleNamespace(head=SimpleNamespace(
                    set_auto_tilt=lambda *a, **k: True, tilt=lambda *a, **k: None)),
            )
            self.walkieAI = SimpleNamespace(
                image=SimpleNamespace(estimate_poses=lambda img: [person]))

        def rotate_to(self, heading, **kw):
            rotates.append(heading)
            return True

        def snapshot(self):
            return SimpleNamespace(img=None)

        def say(self, text):
            self.said.append(text)

    ctx = _ScanCtx()
    caller = skills.live_scan_for_caller(ctx, [], 0.6)

    assert caller is not None
    assert caller.world_xy == (2.0, 0.0)
    assert abs(caller.bearing - 0.1) < 1e-9
    assert rotates, "scan must rotate via go_to (rotate_to)"
    assert any("waving" in s for s in ctx.said)  # announced the count


def _stub_scan_detection(monkeypatch):
    """Make any 'person' read as a single waver at a fixed bearing/point."""
    from types import SimpleNamespace
    monkeypatch.setattr(skills, "is_calling", lambda p, **k: True)
    monkeypatch.setattr(skills, "_torso_center_u", lambda p, **k: 320.0)
    monkeypatch.setattr(skills, "_bearing_from_pixel_u", lambda u, snap, **k: 0.0)
    monkeypatch.setattr(skills, "_person_world_xy", lambda snap, p, **k: (2.0, 0.0))
    monkeypatch.setattr(skills, "_aim_head_for_people", lambda ctx: None)
    monkeypatch.setenv("RESTAURANT_APPROACH_FIRST", "1")
    monkeypatch.setenv("RESTAURANT_SCAN_SETTLE_SEC", "0")
    return SimpleNamespace(bbox=(320.0, 240.0, 40.0, 80.0), confidence=0.9, keypoints=[])


def test_live_scan_cmdvel_spins_via_set_velocity(monkeypatch):
    """cmdvel mode spins the base via nav.set_velocity (non-zero yaw) and zero-stops after."""
    from types import SimpleNamespace
    person = _stub_scan_detection(monkeypatch)
    monkeypatch.setenv("RESTAURANT_SCAN_ROTATE_MODE", "cmdvel")
    monkeypatch.setenv("RESTAURANT_SCAN_DETECT_SEC", "0")  # detect on the first iteration

    cmds: list[tuple[float, float, float]] = []
    nav = SimpleNamespace(
        set_velocity=lambda vx, vy, wz: cmds.append((vx, vy, wz)) or True,
        stop=lambda: True,
    )

    class _Ctx:
        def __init__(self):
            self.data: dict = {}
            self.said: list[str] = []
            self.walkie = SimpleNamespace(
                nav=nav,
                status=SimpleNamespace(get_position=lambda: {"x": 0.0, "y": 0.0, "heading": 0.0}),
                robot=SimpleNamespace(head=SimpleNamespace(
                    set_auto_tilt=lambda *a, **k: True, tilt=lambda *a, **k: None)),
            )
            self.walkieAI = SimpleNamespace(image=SimpleNamespace(estimate_poses=lambda img: [person]))

        def rotate_to(self, heading, **kw):
            return True

        def snapshot(self):
            return SimpleNamespace(img=None)

        def say(self, text):
            self.said.append(text)

    caller = skills.live_scan_for_caller(_Ctx(), [], 0.6)
    assert caller is not None and caller.world_xy == (2.0, 0.0)
    assert any(abs(wz) > 0 for (vx, vy, wz) in cmds), "cmdvel must spin via set_velocity"
    assert cmds[-1] == (0.0, 0.0, 0.0)  # guaranteed zero-stop in finally


def test_live_scan_cmdvel_falls_back_to_gotostep_without_channel(monkeypatch):
    """cmdvel mode degrades to go_to-step rotation when the cmd_vel channel won't open."""
    from types import SimpleNamespace
    person = _stub_scan_detection(monkeypatch)
    monkeypatch.setenv("RESTAURANT_SCAN_ROTATE_MODE", "cmdvel")

    rotates: list[float] = []

    class _Ctx:
        def __init__(self):
            self.data: dict = {}
            self.said: list[str] = []
            # walkie has NO `nav` attribute -> cmd_vel channel open raises -> fallback.
            self.walkie = SimpleNamespace(
                status=SimpleNamespace(get_position=lambda: {"x": 0.0, "y": 0.0, "heading": 0.0}),
                robot=SimpleNamespace(head=SimpleNamespace(
                    set_auto_tilt=lambda *a, **k: True, tilt=lambda *a, **k: None)),
            )
            self.walkieAI = SimpleNamespace(image=SimpleNamespace(estimate_poses=lambda img: [person]))

        def rotate_to(self, heading, **kw):
            rotates.append(heading)
            return True

        def snapshot(self):
            return SimpleNamespace(img=None)

        def say(self, text):
            self.said.append(text)

    caller = skills.live_scan_for_caller(_Ctx(), [], 0.6)
    assert caller is not None
    assert rotates, "cmdvel must fall back to go_to-step rotation when cmd_vel is unavailable"


def test_live_approach_no_odom_fix_fails_fast(monkeypatch):
    """No odometry fix -> (False, None, None) without issuing a drive."""
    _stub_perception(monkeypatch, seen=True)

    class _NoFixStatus(_Status):
        def get_position(self):
            return None

    nav = _Nav([False])
    ctx = FakeCtx(_Walkie(nav, _NoFixStatus(linear=0.0)))
    caller = Caller((1.0, 0.0), 0.0, (0, 0, 10, 10), 0.9)

    reached, caption, final_xy = live_approach_caller(ctx, caller)

    assert reached is False and caption is None and final_xy is None
    assert nav.go_to_calls == []
