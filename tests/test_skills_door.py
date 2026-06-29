"""Offline tests for the reusable arena-entry door skill (tasks/skills/door.py).

Pure control flow over a mock ctx.say / ctx.listen — no robot, no mic.
"""

from __future__ import annotations

import numpy as np
import pytest

from tasks.skills import (
    door_open_from_depth,
    go_to_through_door,
    is_door_open,
    mapped_door_near,
    request_open_door,
)
from tasks.skills.locations import _reset_cache


@pytest.fixture(autouse=True)
def _fresh_location_book():
    """Drop the process-wide LocationBook cache around every test.

    The map gate reads ``get_location_book()`` (cached). Resetting before AND after
    each test keeps the doors a map-gate test installs from leaking into the depth-only
    tests (which must see the real, door-less sibling world.toml → gate None → legacy).
    """
    _reset_cache()
    yield
    _reset_cache()


class _Ctx:
    """Records spoken lines; serves canned mic replies (then ''), counts listens."""

    def __init__(self, replies=()):
        self._replies = list(replies)
        self.said: list[str] = []
        self.listens = 0

    def say(self, text):
        self.said.append(text)

    def listen(self, timeout=30.0):
        self.listens += 1
        return self._replies.pop(0) if self._replies else ""


def _depth(value, shape=(48, 64)):
    return np.full(shape, value, dtype=float)


class _CamCtx(_Ctx):
    """_Ctx plus a snapshot() returning a fixed-depth frame (None = no snapshot)."""

    def __init__(self, depth_value, **kw):
        super().__init__(**kw)
        self._depth_value = depth_value

    def snapshot(self):
        if self._depth_value is None:
            return None
        return type("_S", (), {"depth": _depth(self._depth_value)})()


class _SeqCamCtx(_Ctx):
    """_Ctx whose snapshot() walks a list of depth values (last value repeats).

    A scalar is treated as a one-element list (a fixed depth). Lets a test simulate
    a door that reads closed for a few polls and then opens.
    """

    def __init__(self, depth_values, **kw):
        super().__init__(**kw)
        self._depths = list(depth_values) if isinstance(depth_values, (list, tuple)) else [depth_values]

    def snapshot(self):
        value = self._depths.pop(0) if len(self._depths) > 1 else self._depths[0]
        if value is None:
            return None
        return type("_S", (), {"depth": _depth(value)})()


def test_confirms_on_spoken_reply():
    ctx = _Ctx(replies=["the door is open now"])
    assert request_open_door(ctx, attempts=3) is True
    assert any("coming in" in s.lower() for s in ctx.said)   # thanked + proceeding
    assert ctx.listens == 1                                  # stopped after the confirm


def test_proceeds_after_attempts_without_confirmation():
    ctx = _Ctx(replies=[])  # nobody ever confirms
    assert request_open_door(ctx, attempts=2) is False       # proceeded unconfirmed
    assert ctx.listens == 2                                  # asked + listened twice
    assert any("assume the door is open" in s.lower() for s in ctx.said)


def test_is_open_check_short_circuits_without_listening():
    ctx = _Ctx()
    assert request_open_door(ctx, is_open=lambda: True) is True
    assert ctx.listens == 0                                  # already open -> never listened
    assert ctx.said == []                                    # and didn't even ask


def test_is_open_becomes_true_after_a_prompt():
    seq = iter([False, True])  # closed at first check, open after one prompt
    ctx = _Ctx(replies=[""])
    assert request_open_door(ctx, attempts=3, is_open=lambda: next(seq)) is True
    assert ctx.listens == 1


def test_non_confirming_chatter_does_not_open():
    # An unrelated reply must NOT be taken as a confirmation.
    ctx = _Ctx(replies=["hello robot", "what are you doing"])
    assert request_open_door(ctx, attempts=2) is False


# --- depth door-state detection (door_open_from_depth, pure) ----------------

def test_door_closed_when_a_near_surface_fills_the_centre():
    assert door_open_from_depth(_depth(0.7), clear_m=1.2) is False


def test_door_open_when_the_path_is_far():
    assert door_open_from_depth(_depth(2.5), clear_m=1.2) is True


def test_door_open_when_the_centre_is_see_through_nan():
    assert door_open_from_depth(_depth(np.nan), clear_m=1.2) is True


def test_only_the_central_doorway_matters():
    # near floor/ceiling bands but a far centre -> OPEN (we look through, not at the frame)
    d = _depth(3.0)
    d[:8, :] = 0.3
    d[-8:, :] = 0.3
    assert door_open_from_depth(d, clear_m=1.2, center_frac=0.4) is True
    # a near surface across the central crop -> CLOSED
    d2 = _depth(3.0)
    d2[10:38, 16:48] = 0.6
    assert door_open_from_depth(d2, clear_m=1.2, center_frac=0.4) is False


def test_low_valid_fraction_reads_open():
    # mostly see-through centre with a stray near pixel -> not enough to call "closed"
    d = _depth(np.nan)
    d[24, 32] = 0.5
    assert door_open_from_depth(d, clear_m=1.2, min_valid_frac=0.5) is True


def test_degenerate_frame_is_not_called_closed():
    assert door_open_from_depth(np.zeros((0, 0)), clear_m=1.2) is True


# --- is_door_open (ctx wrapper) + request_open_door default detector ---------

def test_is_door_open_reads_the_depth_frame():
    assert is_door_open(_CamCtx(0.7)) is False   # near surface -> closed
    assert is_door_open(_CamCtx(2.5)) is True     # clear path -> open


def test_is_door_open_none_when_it_cannot_tell():
    assert is_door_open(_Ctx()) is None           # no snapshot() (mock/no camera)
    assert is_door_open(_CamCtx(None)) is None     # snapshot returned None


def test_request_open_door_detects_open_without_asking():
    ctx = _CamCtx(2.5)  # camera sees a clear path -> open
    assert request_open_door(ctx) is True
    assert ctx.listens == 0 and ctx.said == []     # never had to ask


def test_request_open_door_polls_until_the_camera_sees_open():
    # Camera says closed, then opens a few frames later -> walk in, no mic needed.
    ctx = _SeqCamCtx([0.6, 0.6, 0.6, 2.5])
    assert request_open_door(ctx, poll_interval=0.0, confirm_reads=1) is True
    assert ctx.listens == 0                                       # depth-driven, never asked the mic
    assert any("door appears to be closed" in s.lower() for s in ctx.said)  # asked once
    assert any("coming in" in s.lower() for s in ctx.said)                  # thanked + proceeding


def test_request_open_door_proceeds_after_timeout_if_never_opens():
    # Door never reads open -> proceed anyway after the wait budget (never stuck).
    ctx = _CamCtx(0.6)
    assert request_open_door(ctx, poll_interval=0.0, max_wait=0.02) is False
    assert ctx.listens == 0
    assert any("assume the door is open" in s.lower() for s in ctx.said)


# --- go_to_through_door: the actual purpose (can't reach -> ask for the door) -

class _NavCtx(_SeqCamCtx):
    """_SeqCamCtx plus a goto() that returns a canned sequence of results."""

    def __init__(self, depth_value, goto_results, **kw):
        super().__init__(depth_value, **kw)
        self._goto = list(goto_results)
        self.gotos = 0

    def goto(self, x, y, heading_rad):
        self.gotos += 1
        return self._goto.pop(0) if self._goto else False


class _PoseNavCtx(_NavCtx):
    """_NavCtx plus current_pose() walking a list of (x, y) (last value repeats).

    Lets a test model the robot advancing toward the goal between nav retries.
    """

    def __init__(self, depth_value, goto_results, poses, **kw):
        super().__init__(depth_value, goto_results, **kw)
        self._poses = list(poses)

    def current_pose(self):
        x, y = self._poses.pop(0) if len(self._poses) > 1 else self._poses[0]
        return {"x": x, "y": y, "heading": 0.0}


def test_reaches_goal_directly_without_touching_the_door():
    ctx = _NavCtx(2.5, goto_results=[True])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is True
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0  # never asked


def test_closed_door_asks_then_retries_and_reaches():
    # nav fails, depth says closed -> ask a human, watch the door open, then nav succeeds
    ctx = _NavCtx([0.6, 0.6, 0.6, 2.5], goto_results=[False, True])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0, poll_interval=0.0, confirm_reads=1) is True
    assert ctx.gotos == 2 and ctx.listens == 0          # depth-driven entry, no mic
    assert any("coming in" in s.lower() for s in ctx.said)


def test_nav_fails_but_door_is_open_does_not_ask():
    # nav failed for some OTHER reason (door positively open) -> don't pester a human
    ctx = _NavCtx(2.5, goto_results=[False])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is False
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0


def test_partly_open_door_asks_to_widen_then_reaches():
    # Doorway READS open (centre clear) but the robot can't fit through the narrow gap
    # -> with ask_even_if_open (the entry setting) ask for it wider, retry, succeed.
    ctx = _NavCtx(2.5, goto_results=[False, True])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0, ask_even_if_open=True, retry_pause=0.0) is True
    assert ctx.gotos == 2 and ctx.listens == 0                       # nav success is the signal
    assert any("all the way" in s.lower() for s in ctx.said)         # asked to open it wider


def test_partly_open_door_gives_up_after_attempts():
    # Gap never widens / nav never fits -> ask a bounded number of times, then fail.
    ctx = _NavCtx(2.5, goto_results=[False, False, False])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0,
                              ask_even_if_open=True, retry_pause=0.0, door_attempts=2) is False
    assert ctx.gotos == 3                                            # initial + 2 retries
    assert sum("all the way" in s.lower() for s in ctx.said) == 2    # one ask per retry round


def test_partly_open_door_stops_asking_once_driving_through():
    # Person opens the door; the robot starts driving through. The transit goto still
    # FAILs once (costmap clearing), but the robot has clearly advanced toward the goal
    # -> it must NOT pester the operator again, just retry quietly. The transit waypoint
    # stays > WALKIE_DOOR_AT_GOAL_M (0.5 m) from the goal so the at-goal guard doesn't
    # short-circuit it (that path is covered by the test below).
    ctx = _PoseNavCtx(
        2.5,                                           # depth reads open the whole time
        goto_results=[False, False, True],             # stuck, transit-fail, then arrive
        poses=[(1.0, 0.5), (1.0, 0.5), (1.0, 1.0)],    # far, far (stuck), advanced 1.0 m short
    )
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0,
                              ask_even_if_open=True, retry_pause=0.0, door_attempts=3) is True
    assert ctx.gotos == 3
    assert sum("all the way" in s.lower() for s in ctx.said) == 1    # asked ONCE, not on transit


def test_nav_fails_at_goal_is_treated_as_reached_not_a_door():
    # A placement pose surveyed right at the furniture (a cabinet/table): nav FAILs at
    # the goal and the near surface fills the depth box (reads "closed"). The robot is
    # already there, so it must count as reached, NOT ask for a door that isn't there.
    ctx = _PoseNavCtx(0.6, goto_results=[False], poses=[(1.0, 1.8)])  # 0.2 m short, depth "closed"
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0, ask_even_if_open=True) is True
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0     # never asked


def test_cannot_tell_still_asks_when_blocked():
    # no camera -> is_door_open None; blocked nav still asks (the helpful action)
    ctx = _NavCtx(None, goto_results=[False, True], replies=["done"])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is True
    assert ctx.gotos == 2 and ctx.listens == 1


# --- map-gated door asking (a [doors] table makes the ask location-precise) --

def _map_with_door(tmp_path, monkeypatch, x, y, *, radius=None):
    """Write a world.toml whose only content is one door at (x, y) and point the
    shared LocationBook at it. Returns nothing — the gate reads the cached book."""
    toml = f"[doors.entrance]\npose = [{x}, {y}, 0.0]\n"
    if radius is not None:
        toml += f"radius = {radius}\n"
    p = tmp_path / "world.toml"
    p.write_text(toml)
    monkeypatch.setenv("WALKIE_MAP_FILE", str(p))
    _reset_cache()  # the autouse fixture cleared it; drop the no-map book loaded since


def test_mapped_door_near_is_tristate(tmp_path, monkeypatch):
    # No doors mapped -> None (caller keeps legacy depth behaviour).
    assert mapped_door_near(_PoseNavCtx(2.5, [True], poses=[(0.0, 0.0)])) is None
    # Door at (1, 2): robot on it -> True, robot far -> False.
    _map_with_door(tmp_path, monkeypatch, 1.0, 2.0, radius=1.5)
    assert mapped_door_near(_PoseNavCtx(2.5, [True], poses=[(1.0, 2.0)])) is True
    assert mapped_door_near(_PoseNavCtx(2.5, [True], poses=[(10.0, 10.0)])) is False


def test_mapped_door_near_none_when_pose_unknown(tmp_path, monkeypatch):
    # Doors mapped, but a mock ctx with no current_pose() can't be gated -> None.
    _map_with_door(tmp_path, monkeypatch, 1.0, 2.0)
    assert mapped_door_near(_NavCtx(2.5, goto_results=[True])) is None


def test_mapped_door_gate_off_returns_none(tmp_path, monkeypatch):
    _map_with_door(tmp_path, monkeypatch, 1.0, 2.0)
    monkeypatch.setenv("WALKIE_DOOR_MAP_GATE", "0")
    assert mapped_door_near(_PoseNavCtx(2.5, [True], poses=[(1.0, 2.0)])) is None


def test_asks_at_a_mapped_door_even_when_depth_reads_open(tmp_path, monkeypatch):
    # Depth reads OPEN, but the robot is sitting on a mapped door and nav is blocked
    # -> treat as a partly-open door (we KNOW one is here): ask to widen, retry, reach.
    _map_with_door(tmp_path, monkeypatch, 1.0, 1.0)
    ctx = _PoseNavCtx(2.5, goto_results=[False, True], poses=[(1.0, 1.0)])
    assert go_to_through_door(ctx, 1.0, 5.0, 0.0, retry_pause=0.0) is True
    assert ctx.gotos == 2 and ctx.listens == 0
    assert any("all the way" in s.lower() for s in ctx.said)  # asked because mapped door


def test_no_ask_when_block_is_away_from_every_mapped_door(tmp_path, monkeypatch):
    # Depth reads CLOSED (legacy would ask) but no mapped door is near -> not a door,
    # so the robot doesn't pester anyone: precisely the false-ask the map kills.
    _map_with_door(tmp_path, monkeypatch, 9.0, 9.0)
    ctx = _PoseNavCtx(0.6, goto_results=[False], poses=[(1.0, 1.0)])
    assert go_to_through_door(ctx, 1.0, 5.0, 0.0) is False
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0


def test_gate_off_restores_depth_only_behaviour(tmp_path, monkeypatch):
    # A door is mapped far away, but with the gate OFF it's ignored: the legacy
    # ask_even_if_open partly-open path runs as before, regardless of the map.
    _map_with_door(tmp_path, monkeypatch, 9.0, 9.0)
    monkeypatch.setenv("WALKIE_DOOR_MAP_GATE", "0")
    ctx = _PoseNavCtx(2.5, goto_results=[False, True], poses=[(1.0, 1.0)])
    assert go_to_through_door(ctx, 1.0, 5.0, 0.0, ask_even_if_open=True, retry_pause=0.0) is True
    assert any("all the way" in s.lower() for s in ctx.said)
