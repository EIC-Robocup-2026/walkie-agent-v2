"""Top-level WalkieClient — single entry point for the entire HTTP API."""

from __future__ import annotations
import os

from .appearance import AppearanceClient
from .face_recognition import FaceRecognitionClient
from .image_caption import ImageCaptionClient
from .image_embed import ImageEmbedClient
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
        base_url: str|None = None,
        timeout: int = 60,
    ) -> None:
        if base_url is None:
            base_url = os.getenv("WALKIE_BASE_URL") or "http://localhost:5000"
        self.stt = STTClient(base_url=base_url, timeout=timeout)
        self.tts = TTSClient(base_url=base_url, timeout=timeout)
        self.object_detection = ObjectDetectionClient(base_url=base_url, timeout=timeout)
        self.pose_estimation = PoseEstimationClient(base_url=base_url, timeout=timeout)
        self.image_caption = ImageCaptionClient(base_url=base_url, timeout=timeout)
        self.image_embed = ImageEmbedClient(base_url=base_url, timeout=timeout)
        self.face_recognition = FaceRecognitionClient(base_url=base_url, timeout=timeout)
        self.appearance = AppearanceClient(base_url=base_url, timeout=timeout)
