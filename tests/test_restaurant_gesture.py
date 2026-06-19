"""Pure unit tests for the Restaurant wave/calling-gesture classifier.

No robot, no AI server — synthetic COCO keypoints fed to is_calling_gesture.
"""

from client import PersonPose, PoseKeypoint
from tasks.skills import is_calling_gesture

# COCO indices used by the classifier.
_NAMES = {5: "left_shoulder", 6: "right_shoulder", 9: "left_wrist", 10: "right_wrist"}


def _kp(index, y, conf=0.9):
    return PoseKeypoint(x=100.0, y=y, confidence=conf, name=_NAMES.get(index, str(index)), index=index)


def _person(keypoints, bbox=(320, 240, 80, 200)):
    return PersonPose(bbox=bbox, confidence=0.9, keypoints=keypoints)


def test_left_arm_raised_is_calling():
    # Left wrist (y=50) above left shoulder (y=150) -> raised.
    p = _person([_kp(5, 150), _kp(9, 50), _kp(6, 150), _kp(10, 150)])
    assert is_calling_gesture(p) is True


def test_right_arm_raised_is_calling():
    p = _person([_kp(5, 150), _kp(9, 150), _kp(6, 150), _kp(10, 40)])
    assert is_calling_gesture(p) is True


def test_arms_down_is_not_calling():
    # Both wrists below their shoulders -> not calling.
    p = _person([_kp(5, 150), _kp(9, 260), _kp(6, 150), _kp(10, 255)])
    assert is_calling_gesture(p) is False


def test_low_confidence_wrist_ignored():
    # Wrist is geometrically raised but below the confidence threshold.
    p = _person([_kp(5, 150), _kp(9, 50, conf=0.1), _kp(6, 150), _kp(10, 150)])
    assert is_calling_gesture(p) is False


def test_missing_keypoints_is_not_calling():
    p = _person([_kp(5, 150), _kp(6, 150)])  # no wrists
    assert is_calling_gesture(p) is False
