"""Offline tests for the reusable arena-entry door skill (tasks/skills/door.py).

Pure control flow over a mock ctx.say / ctx.listen — no robot, no mic.
"""

from __future__ import annotations

from tasks.skills import request_open_door


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
