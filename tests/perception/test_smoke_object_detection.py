"""Smoke: ObjectDetectionClient returns DetectedObject in the expected shape.

The perception loop will call ``walkieAI.object_detection.detect(image)`` and
feed the resulting bboxes into the SDK's ``bboxes_to_positions``. We only care
here that the HTTP wrapper:

  - POSTs to ``/object-detection/detect`` with a multipart ``image`` field
  - parses ``{"success": true, "data": [...]}`` into a list of DetectedObject
  - preserves ``bbox`` as a 4-tuple, ``class_name`` and ``confidence`` as set
"""

from __future__ import annotations

from unittest.mock import patch

from client.object_detection import DetectedObject, ObjectDetectionClient


def test_detect_parses_yolo_envelope(tiny_pil_image, fake_success_response):
    server_payload = [
        {
            "bbox": [10, 20, 60, 80],
            "area_ratio": 0.05,
            "class_id": 56,
            "class_name": "chair",
            "confidence": 0.91,
            "mask_b64": None,
        },
        {
            "bbox": [200, 100, 260, 180],
            "area_ratio": 0.04,
            "class_id": 39,
            "class_name": "bottle",
            "confidence": 0.72,
            "mask_b64": None,
        },
    ]

    client = ObjectDetectionClient(base_url="http://stub")
    with patch.object(client._session, "post", return_value=fake_success_response(server_payload)) as post:
        detections = client.detect(tiny_pil_image)

    post.assert_called_once()
    url, = post.call_args.args
    assert url == "http://stub/object-detection/detect"
    assert "image" in post.call_args.kwargs["files"]

    assert len(detections) == 2
    chair, bottle = detections
    assert isinstance(chair, DetectedObject)
    assert chair.bbox == (10, 20, 60, 80)
    assert chair.class_name == "chair"
    assert 0.0 <= chair.confidence <= 1.0
    assert bottle.class_name == "bottle"


def test_detect_raises_on_server_failure(tiny_pil_image, fake_error_response):
    from client.base import WalkieAPIError

    client = ObjectDetectionClient(base_url="http://stub")
    with patch.object(client._session, "post", return_value=fake_error_response("no GPU")):
        try:
            client.detect(tiny_pil_image)
        except WalkieAPIError as e:
            assert "no GPU" in str(e)
        else:
            raise AssertionError("expected WalkieAPIError")
