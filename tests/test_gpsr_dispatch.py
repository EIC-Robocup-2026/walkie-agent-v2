"""Offline unit tests for the pure dispatch policy (plan.prefer_tier1 /
plan.summarize_status). The skill/agent wiring (dispatch.py) is robot-side and
verified there; the routing + status logic is pure and tested here.
"""

from __future__ import annotations

from tasks.GPSR.plan import CmdStatus, PlanStep, Primitive, prefer_tier1, summarize_status


def _step(primitive, grounded=True):
    return PlanStep(primitive, {}, "raw", [] if grounded else [("x", "y")])


def test_prefer_tier1_grounded_nonmanip():
    assert prefer_tier1(_step(Primitive.NAVIGATE), manip_enabled=False)


def test_prefer_tier1_ungrounded_goes_tier2():
    assert not prefer_tier1(_step(Primitive.NAVIGATE, grounded=False), manip_enabled=True)


def test_prefer_tier1_manip_gated_off():
    assert not prefer_tier1(_step(Primitive.PICK), manip_enabled=False)


def test_prefer_tier1_manip_enabled():
    assert prefer_tier1(_step(Primitive.PICK), manip_enabled=True)


def test_summarize_status():
    assert summarize_status([]) is CmdStatus.FAILED
    assert summarize_status([True, True]) is CmdStatus.DONE
    assert summarize_status([True, False]) is CmdStatus.PARTIAL
    assert summarize_status([False, False]) is CmdStatus.FAILED
