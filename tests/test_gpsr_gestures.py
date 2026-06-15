"""Offline unit tests for the GPSR keypoint gesture heuristics (gestures.py).

Synthetic COCO poses — no robot, no detector. Image y grows downward, so a
raised hand has a SMALLER y than the shoulder.
"""

from __future__ import annotations

from client.pose_estimation import PersonPose, PoseKeypoint
from tasks.GPSR import gestures


def _person(**pts: tuple[float, float]) -> PersonPose:
    """Build a PersonPose from name -> (x, y) keypoints (confidence 0.9)."""
    kps = [
        PoseKeypoint(x=x, y=y, confidence=0.9, name=name, index=i)
        for i, (name, (x, y)) in enumerate(pts.items())
    ]
    xs = [x for x, _ in pts.values()] or [0]
    ys = [y for _, y in pts.values()] or [0]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    w, h = (max(xs) - min(xs)) or 50, (max(ys) - min(ys)) or 50
    return PersonPose(bbox=(int(cx), int(cy), int(w), int(h)), confidence=0.9, keypoints=kps)


# Upright, arms hanging, legs extended down.
STANDING = _person(
    left_shoulder=(140, 100), right_shoulder=(160, 100),
    left_hip=(140, 200), right_hip=(160, 200),
    left_knee=(140, 300), right_knee=(160, 300),
    left_ankle=(140, 400), right_ankle=(160, 400),
    left_wrist=(140, 200), right_wrist=(160, 200),
)

# Thighs ~horizontal: knees and ankles tucked up near the hips.
SITTING = _person(
    left_shoulder=(140, 100), right_shoulder=(160, 100),
    left_hip=(140, 200), right_hip=(160, 200),
    left_knee=(140, 225), right_knee=(160, 225),
    left_ankle=(140, 250), right_ankle=(160, 250),
)

# Left hand raised above the left shoulder.
RAISING_LEFT = _person(
    left_shoulder=(140, 100), right_shoulder=(160, 100),
    left_hip=(140, 200), right_hip=(160, 200),
    left_wrist=(140, 40), right_wrist=(160, 200),
    left_ankle=(140, 400), right_ankle=(160, 400),
)

# Right arm extended horizontally to image-right, at shoulder height.
POINTING_RIGHT = _person(
    left_shoulder=(140, 100), right_shoulder=(160, 100),
    left_hip=(140, 200), right_hip=(160, 200),
    right_wrist=(250, 108), left_wrist=(140, 200),
)

# Body horizontal: everything at similar y, spread across x.
LYING = _person(
    left_shoulder=(100, 100), right_shoulder=(110, 102),
    left_hip=(200, 100), right_hip=(210, 102),
    left_ankle=(300, 100), right_ankle=(310, 102),
)


def test_standing():
    assert gestures.is_standing(STANDING)
    assert not gestures.is_sitting(STANDING)
    assert not gestures.is_waving(STANDING)
    assert not gestures.is_lying_down(STANDING)


def test_sitting():
    assert gestures.is_sitting(SITTING)
    assert not gestures.is_standing(SITTING)


def test_raising_left_arm():
    assert gestures.matches_gesture(RAISING_LEFT, "raising_left_arm")
    assert not gestures.matches_gesture(RAISING_LEFT, "raising_right_arm")
    assert gestures.is_waving(RAISING_LEFT)  # a raised hand reads as waving


def test_pointing_right():
    assert gestures.matches_gesture(POINTING_RIGHT, "pointing_right")
    assert not gestures.matches_gesture(POINTING_RIGHT, "pointing_left")


def test_lying_down():
    assert gestures.is_lying_down(LYING)
    assert not gestures.is_standing(LYING)


def test_classify_and_match_dispatch():
    assert "standing" in gestures.classify_gestures(STANDING)
    assert gestures.matches_gesture(STANDING, "standing")
    assert not gestures.matches_gesture(STANDING, "waving")
    # Unknown gesture id never matches.
    assert not gestures.matches_gesture(STANDING, "breakdancing")


def test_low_confidence_keypoints_ignored():
    # Same as RAISING_LEFT but the raised wrist is below threshold -> not raised.
    p = _person(
        left_shoulder=(140, 100), right_shoulder=(160, 100),
        left_hip=(140, 200), right_hip=(160, 200),
    )
    p.keypoints.append(PoseKeypoint(x=140, y=40, confidence=0.1, name="left_wrist", index=9))
    assert not gestures.matches_gesture(p, "raising_left_arm")
