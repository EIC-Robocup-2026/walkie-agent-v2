"""Top-level WalkieClient — single entry point for the entire HTTP API."""

from __future__ import annotations
import os

from .image import ImageClient
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

        # Vision — one image upload, any combination of tasks
        res = walkie.image.process(pil_image, detection=True, caption=True, pose=True)
        for obj in res.detection:
            print(obj.class_name, obj.confidence)
        print(res.caption)
        for person in res.pose:
            print(len(person.keypoints), "keypoints detected")

        # Single-task helpers
        detections = walkie.image.detect(pil_image)
        poses      = walkie.image.estimate_poses(pil_image)
        caption    = walkie.image.caption(pil_image)

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
            base_url = os.getenv("WALKIE_AI_BASE_URL") or "http://localhost:5000"
        self.stt = STTClient(base_url=base_url, timeout=timeout)
        self.tts = TTSClient(base_url=base_url, timeout=timeout)
        self.image = ImageClient(base_url=base_url, timeout=timeout)
