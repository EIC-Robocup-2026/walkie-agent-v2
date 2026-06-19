"""Unit tests for ImageClient.detect()'s opt-in downscale.

No server: a stub overrides ``_post_files`` to capture the bytes that would be
sent and to return canned detections (in the unified ``/image/process``
envelope) in the *downscaled* coordinate frame. The tests assert the client
(a) downscales the sent image when ``max_size`` is set, (b) scales the returned
bbox/mask back to the ORIGINAL input resolution, and (c) is byte-for-byte
identity when ``max_size`` is None (the default).
"""

import base64
import io

import numpy as np
from PIL import Image

from client.image import ImageClient


def _mask_b64(w: int, h: int) -> str:
    """A w×h binary mask (top-left quadrant set) as the server encodes it:
    an 8-bit grayscale PNG scaled by 255, base64'd."""
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[: h // 2, : w // 2] = 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _StubClient(ImageClient):
    """ImageClient whose HTTP POST is replaced by a canned response.

    Records the image bytes + form it was handed so a test can inspect what the
    real client would have sent over the wire. ``detections`` are returned in the
    unified ``{"detection": [...]}`` envelope that ``process`` parses.
    """

    def __init__(self, detections):
        super().__init__(base_url="http://stub")
        self._detections = detections
        self.sent_size: tuple[int, int] | None = None  # (w, h) of the sent JPEG
        self.sent_form: dict | None = None

    def _post_files(self, path, files, data):  # type: ignore[override]
        image_bytes = files["image"][1]
        self.sent_size = Image.open(io.BytesIO(image_bytes)).size
        self.sent_form = dict(data)
        return {"detection": self._detections}


def test_downscale_scales_bbox_and_mask_back_to_input_resolution():
    # 1280x720 input, max_size=640 -> scale 0.5. The server "sees" 640x360 and
    # returns geometry in that frame; the client must restore input coords.
    data = [{
        "bbox": [10, 20, 100, 200],
        "area_ratio": 0.1,
        "class_name": "chair",
        "confidence": 0.9,
        "mask_b64": _mask_b64(640, 360),
    }]
    client = _StubClient(data)
    img = Image.new("RGB", (1280, 720), (128, 128, 128))

    out = client.detect(img, max_size=640, return_mask=True)

    assert client.sent_size == (640, 360)  # image was downscaled before sending
    assert len(out) == 1
    det = out[0]
    assert det.bbox == (20, 40, 200, 400)  # bbox upscaled ×2 back to input coords
    assert det.mask is not None
    assert det.mask.shape == (720, 1280)  # mask resized back to (h, w)
    assert set(np.unique(det.mask)).issubset({0, 1})  # still binary
    assert det.area_ratio == 0.1  # ratio is scale-invariant — untouched


def test_no_resize_is_identity_when_max_size_none():
    data = [{"bbox": [10, 20, 100, 200], "area_ratio": 0.1, "class_name": "chair"}]
    client = _StubClient(data)
    img = Image.new("RGB", (1280, 720), (128, 128, 128))

    out = client.detect(img)  # max_size defaults to None

    assert client.sent_size == (1280, 720)  # full resolution sent
    assert out[0].bbox == (10, 20, 100, 200)  # unchanged


def test_no_resize_when_image_smaller_than_max_size():
    # max_size larger than the longest edge must NOT upscale.
    data = [{"bbox": [1, 2, 3, 4], "area_ratio": 0.01, "class_name": "stool"}]
    client = _StubClient(data)
    img = Image.new("RGB", (320, 240), (0, 0, 0))

    out = client.detect(img, max_size=640)

    assert client.sent_size == (320, 240)
    assert out[0].bbox == (1, 2, 3, 4)


def test_downscale_numpy_path():
    # BGR ndarray (H, W, C): 720x1280 -> max_size 640 -> 360x640.
    data = [{"bbox": [10, 20, 100, 200], "area_ratio": 0.1, "class_name": "chair"}]
    client = _StubClient(data)
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)

    out = client.detect(arr, max_size=640)

    assert client.sent_size == (640, 360)
    assert out[0].bbox == (20, 40, 200, 400)


def test_spec_carries_detection_task():
    import json

    client = _StubClient([])
    img = Image.new("RGB", (320, 240), (0, 0, 0))

    client.detect(img, prompts=["chair"], return_mask=True)
    spec = json.loads(client.sent_form["spec"])
    assert spec["detection"]["return_mask"] is True
    assert spec["detection"]["prompts"] == ["chair"]
