"""Unit tests for the single-frame gesture / posture heuristics (HRI C7).

Pure-function coverage over synthetic COCO keypoints — no camera, no server.
The pointing/lying additions live in ``perception/gestures.py``; the older
arm-raised/posture behaviour is re-tested through it (the human-agent tools
re-export the same functions, covered in ``test_human_tools.py``).
"""

from types import SimpleNamespace as NS

from perception import gestures as g


def _kp(index, x, y, conf=0.9):
    return NS(index=index, x=x, y=y, confidence=conf)


def _pose(*keypoints):
    return NS(keypoints=list(keypoints))


# --- pointing -------------------------------------------------------------


def test_pointing_right_when_wrist_extends_past_shoulders():
    # shoulders span x=0..40 (width 40); right wrist reaches far to the right.
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.RIGHT_WRIST, 140, 105),  # dx=+100 > 1.2*40, nearly level
    )
    assert g.pointing(g.kp_map(pose)) == "right"


def test_pointing_left_when_wrist_extends_to_the_left():
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.LEFT_WRIST, -100, 98),  # dx=-100 from left shoulder
    )
    assert g.pointing(g.kp_map(pose)) == "left"


def test_no_pointing_for_a_raised_hand():
    # wrist straight up above the shoulder: vertical, not a sideways point.
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.LEFT_WRIST, 2, 10),  # dx tiny, dy large
    )
    assert g.pointing(g.kp_map(pose)) is None


def test_no_pointing_when_arm_only_slightly_extended():
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.RIGHT_WRIST, 70, 100),  # dx=30 < 1.2*40
    )
    assert g.pointing(g.kp_map(pose)) is None


def test_pointing_none_without_both_shoulders():
    pose = _pose(_kp(g.LEFT_SHOULDER, 0, 100), _kp(g.LEFT_WRIST, -200, 100))
    assert g.pointing(g.kp_map(pose)) is None


def test_pointing_ignores_low_confidence_wrist():
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.RIGHT_WRIST, 200, 100, conf=0.1),
    )
    assert g.pointing(g.kp_map(pose)) is None


# --- posture: lying -------------------------------------------------------


def test_posture_lying_when_torso_is_horizontal():
    # shoulders and hips side-by-side (big dx, small dy) -> lying.
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 50), _kp(g.RIGHT_SHOULDER, 0, 60),
        _kp(g.LEFT_HIP, 200, 50), _kp(g.RIGHT_HIP, 200, 60),
    )
    assert g.posture(g.kp_map(pose)) == "lying"


def test_posture_standing_not_mistaken_for_lying():
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 0), _kp(g.RIGHT_SHOULDER, 10, 0),
        _kp(g.LEFT_HIP, 0, 100), _kp(g.RIGHT_HIP, 10, 100),
        _kp(g.LEFT_KNEE, 0, 200), _kp(g.RIGHT_KNEE, 10, 200),
    )
    assert g.posture(g.kp_map(pose)) == "standing"


def test_posture_lying_takes_priority_over_missing_legs():
    # horizontal torso, no knees at all -> still 'lying', not 'unknown'.
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 50), _kp(g.RIGHT_SHOULDER, 0, 50),
        _kp(g.LEFT_HIP, 300, 55), _kp(g.RIGHT_HIP, 300, 55),
    )
    assert g.posture(g.kp_map(pose)) == "lying"


# --- describe_gestures + phrase -------------------------------------------


def test_describe_gestures_combines_waving_pointing_posture():
    pose = _pose(
        _kp(g.LEFT_SHOULDER, 0, 100), _kp(g.RIGHT_SHOULDER, 40, 100),
        _kp(g.LEFT_WRIST, 0, 40),       # hand up -> waving
        _kp(g.RIGHT_WRIST, 150, 102),   # extended right -> pointing right
        _kp(g.LEFT_HIP, 0, 200), _kp(g.RIGHT_HIP, 40, 200),
        _kp(g.LEFT_KNEE, 0, 300), _kp(g.RIGHT_KNEE, 40, 300),  # standing
    )
    out = g.describe_gestures(pose)
    assert out == {"waving": True, "pointing": "right", "posture": "standing"}


def test_gesture_phrase_reads_naturally():
    phrase = g.gesture_phrase({"waving": True, "pointing": "left", "posture": "sitting"})
    assert "waving" in phrase and "left" in phrase and "sitting" in phrase


def test_gesture_phrase_when_nothing_detected():
    phrase = g.gesture_phrase({"waving": False, "pointing": None, "posture": "unknown"})
    assert phrase == "no clear gesture"
