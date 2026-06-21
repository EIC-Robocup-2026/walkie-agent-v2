"""Offline tests for the reusable arena-entry door skill (tasks/skills/door.py).

Pure control flow over a mock ctx.say / ctx.listen — no robot, no mic.
"""

from __future__ import annotations

import numpy as np

from tasks.skills import (
    door_open_from_depth,
    go_to_through_door,
    is_door_open,
    request_open_door,
)


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


def test_request_open_door_asks_when_camera_sees_closed():
    ctx = _CamCtx(0.6, replies=["it's open now"])  # camera says closed -> ask + confirm
    assert request_open_door(ctx, attempts=2) is True
    assert ctx.listens == 1


# --- go_to_through_door: the actual purpose (can't reach -> ask for the door) -

class _NavCtx(_CamCtx):
    """_CamCtx plus a goto() that returns a canned sequence of results."""

    def __init__(self, depth_value, goto_results, **kw):
        super().__init__(depth_value, **kw)
        self._goto = list(goto_results)
        self.gotos = 0

    def goto(self, x, y, heading_rad):
        self.gotos += 1
        return self._goto.pop(0) if self._goto else False


def test_reaches_goal_directly_without_touching_the_door():
    ctx = _NavCtx(2.5, goto_results=[True])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is True
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0  # never asked


def test_closed_door_asks_then_retries_and_reaches():
    # nav fails, depth says closed -> ask a human, then nav succeeds
    ctx = _NavCtx(0.6, goto_results=[False, True], replies=["it's open now"])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is True
    assert ctx.gotos == 2 and ctx.listens == 1
    assert any("coming in" in s.lower() for s in ctx.said)


def test_nav_fails_but_door_is_open_does_not_ask():
    # nav failed for some OTHER reason (door positively open) -> don't pester a human
    ctx = _NavCtx(2.5, goto_results=[False])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is False
    assert ctx.gotos == 1 and ctx.said == [] and ctx.listens == 0


def test_cannot_tell_still_asks_when_blocked():
    # no camera -> is_door_open None; blocked nav still asks (the helpful action)
    ctx = _NavCtx(None, goto_results=[False, True], replies=["done"])
    assert go_to_through_door(ctx, 1.0, 2.0, 0.0) is True
    assert ctx.gotos == 2 and ctx.listens == 1
