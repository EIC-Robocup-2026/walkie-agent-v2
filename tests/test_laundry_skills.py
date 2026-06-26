"""Laundry manipulation gating (no robot/LLM/network).

``pick_garment`` is wired to the shared grasp system but gated behind
LAUNDRY_ARM_CALIBRATED (mirrors PnP / Restaurant). Gate off (default): announce
only, never touch the arm. Gate on: delegate to ``tasks.skills.pick_object`` with
the garment's class name as the grasp prompt. fold/stack stay genuine stubs.
"""

from __future__ import annotations

from tasks.Laundry import skills
from tasks.Laundry.skills import Garment


class FakeCtx:
    def __init__(self):
        self.data = {}
        self.said: list[str] = []

    def say(self, text):
        self.said.append(text)


def _garment(name="t-shirt"):
    return Garment(bbox_xyxy=(0.0, 0.0, 1.0, 1.0), class_name=name, confidence=0.9,
                   world_xy=(1.0, 0.0))


def test_pick_garment_gated_off_announces_and_skips_arm(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(skills, "_pick_object",
                        lambda ctx, prompts: called.__setitem__("n", called["n"] + 1) or True)
    monkeypatch.setenv("LAUNDRY_ARM_CALIBRATED", "0")
    ctx = FakeCtx()

    assert skills.pick_garment(ctx, _garment()) is False
    assert called["n"] == 0                       # arm gated: never grasped
    assert any("arm is not enabled" in s for s in ctx.said)


def test_pick_garment_gated_on_delegates_with_class_prompt(monkeypatch):
    called = {"n": 0, "prompts": None}

    def _fake_pick(ctx, prompts):
        called["n"] += 1
        called["prompts"] = prompts
        return True

    monkeypatch.setattr(skills, "_pick_object", _fake_pick)
    monkeypatch.setenv("LAUNDRY_ARM_CALIBRATED", "1")
    ctx = FakeCtx()

    assert skills.pick_garment(ctx, _garment("t-shirt")) is True
    assert called["n"] == 1
    assert called["prompts"] == ["t-shirt"]       # class name passed through as the prompt


def test_fold_and_stack_remain_stubs(monkeypatch):
    """The deformable steps are not implemented — they must still degrade to False."""
    monkeypatch.setenv("LAUNDRY_ARM_CALIBRATED", "1")  # even with the arm on
    ctx = FakeCtx()

    assert skills.fold_garment(ctx, _garment()) is False
    assert skills.stack_garment(ctx) is False
