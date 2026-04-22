"""Top-level WalkieClient — single entry point for the entire HTTP API."""

from __future__ import annotations

from .image_caption import ImageCaptionClient
from .object_detection import ObjectDetectionClient
from .pose_estimation import PoseEstimationClient
from .stt import STTClient
from .tts import TTSClient


class WalkieAIClient:
    """Unified client for the walkie-agent-v2 HTTP API.

    Composes all service sub-clients under one object.  Each sub-client
    shares the same ``base_url`` and ``timeout``, but uses its own
    ``requests.Session`` so they can be used concurrently.

    Example::

        from client import WalkieAIClient

        walkie = WalkieAIClient(base_url="http://localhost:5000")

        # Speech-to-Text
        text = walkie.stt.transcribe(audio_bytes)

        # Text-to-Speech
        audio = walkie.tts.synthesize("Hello, I am Walkie.")
        for chunk in walkie.tts.synthesize_stream("Streaming speech."):
            speaker.write(chunk)

        # Object Detection
        detections = walkie.object_detection.detect(pil_image)
        for obj in detections:
            print(obj.class_name, obj.confidence)

        # Pose Estimation
        poses = walkie.pose_estimation.estimate(pil_image)
        for person in poses:
            print(len(person.keypoints), "keypoints detected")

        # Image Captioning
        caption = walkie.image_caption.caption(pil_image)
        captions = walkie.image_caption.caption_batch([img_a, img_b])

    Args:
        base_url: Root URL of the running walkie-agent-v2 server.
        timeout: HTTP request timeout in seconds (applied to all sub-clients).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: int = 60,
    ) -> None:
        self.stt = STTClient(base_url=base_url, timeout=timeout)
        self.tts = TTSClient(base_url=base_url, timeout=timeout)
        self.object_detection = ObjectDetectionClient(base_url=base_url, timeout=timeout)
        self.pose_estimation = PoseEstimationClient(base_url=base_url, timeout=timeout)
        self.image_caption = ImageCaptionClient(base_url=base_url, timeout=timeout)
