"""Unit tests for the human (HRI) sub-agent's pure pose heuristics.

These cover the offline-testable logic only — `describe_person` /
`count_people` themselves hit the camera + walkie-ai-server, so they're
exercised on the robot, not here.
"""

from types import SimpleNamespace as NS

import pytest

from agents.human_agent.tools import (
    _arm_raised,
    _kp_map,
    _posture,
    _summarize_person,
    make_human_tools,
)


def _kp(index, x, y, conf=0.9):
    return NS(index=index, x=x, y=y, confidence=conf)


def _pose(*keypoints):
    return NS(keypoints=list(keypoints))


# --- arm raised -----------------------------------------------------------


def test_arm_raised_when_wrist_above_shoulder():
    # image y grows downward, so "above" means a smaller y
    pose = _pose(_kp(5, 0, 100), _kp(9, 0, 50))  # left shoulder, left wrist
    assert _arm_raised(_kp_map(pose)) is True


def test_arm_not_raised_when_wrist_below_shoulder():
    pose = _pose(_kp(6, 0, 100), _kp(10, 0, 150))  # right shoulder, right wrist
    assert _arm_raised(_kp_map(pose)) is False


def test_arm_raised_ignores_low_confidence_keypoints():
    pose = _pose(_kp(5, 0, 100, conf=0.1), _kp(9, 0, 50, conf=0.1))
    assert _arm_raised(_kp_map(pose)) is False


# --- posture --------------------------------------------------------------


def test_posture_standing_when_knees_well_below_hips():
    pose = _pose(
        _kp(5, 0, 0), _kp(6, 0, 0),      # shoulders
        _kp(11, 0, 100), _kp(12, 0, 100),  # hips
        _kp(13, 0, 200), _kp(14, 0, 200),  # knees (leg_drop = 1.0)
    )
    assert _posture(_kp_map(pose)) == "standing"


def test_posture_sitting_when_knees_near_hip_height():
    pose = _pose(
        _kp(5, 0, 0), _kp(6, 0, 0),
        _kp(11, 0, 100), _kp(12, 0, 100),
        _kp(13, 0, 110), _kp(14, 0, 110),  # leg_drop = 0.1
    )
    assert _posture(_kp_map(pose)) == "sitting"


def test_posture_unknown_without_legs():
    pose = _pose(_kp(5, 0, 0), _kp(11, 0, 100))
    assert _posture(_kp_map(pose)) == "unknown"


def test_posture_unknown_on_degenerate_torso():
    # hips above shoulders → torso <= 0, can't reason
    pose = _pose(
        _kp(5, 0, 100), _kp(6, 0, 100),
        _kp(11, 0, 0), _kp(12, 0, 0),
        _kp(13, 0, 200), _kp(14, 0, 200),
    )
    assert _posture(_kp_map(pose)) == "unknown"


# --- summary + tool surface ----------------------------------------------


def test_summarize_person_combines_flags():
    pose = _pose(
        _kp(5, 0, 100), _kp(9, 0, 50),   # left arm raised
        _kp(6, 0, 100),
        _kp(11, 0, 200), _kp(12, 0, 200),
        _kp(13, 0, 250), _kp(14, 0, 250),  # leg_drop = 0.5 -> standing
    )
    assert _summarize_person(pose) == {"arm_raised": True, "posture": "standing"}


def test_make_human_tools_exposes_expected_surface():
    tools = make_human_tools(NS(camera=None), NS(), agent_name="human")
    assert [t.name for t in tools] == [
        "describe_person",
        "count_people",
        "enroll_person",
        "recognize_person",
        "list_known_people",
        "find_empty_seat",
        "locate_person",
        "speak",
    ]
