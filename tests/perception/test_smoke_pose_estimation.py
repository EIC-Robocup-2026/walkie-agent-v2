"""Smoke: PoseEstimationClient returns PersonPose with 17 COCO keypoints.

The perception loop records people poses alongside objects so the design
doc's "is anyone raising a hand?" query can be served without an extra
inference round-trip. We need the keypoint indices to be the COCO ordering
(0=nose, 5=left_shoulder, 10=right_wrist, ...).
"""

from __future__ import annotations

from unittest.mock import patch

from client.pose_estimation import PersonPose, PoseEstimationClient

COCO_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def _fake_person_payload() -> dict:
    return {
        "bbox": [320, 240, 100, 200],
        "confidence": 0.88,
        "keypoints": [
            {"index": i, "name": name, "x": 100 + i, "y": 100 + i, "confidence": 0.5}
            for i, name in enumerate(COCO_NAMES)
        ],
    }


def test_estimate_parses_person_pose(tiny_pil_image, fake_success_response):
    client = PoseEstimationClient(base_url="http://stub")
    with patch.object(
        client._session,
        "post",
        return_value=fake_success_response([_fake_person_payload()]),
    ):
        people = client.estimate(tiny_pil_image)

    assert len(people) == 1
    person = people[0]
    assert isinstance(person, PersonPose)
    assert person.bbox == (320, 240, 100, 200)
    assert person.confidence == 0.88
    assert len(person.keypoints) == 17
    assert [kp.index for kp in person.keypoints] == list(range(17))
    assert person.keypoints[5].name == "left_shoulder"
    assert person.keypoints[10].name == "right_wrist"
