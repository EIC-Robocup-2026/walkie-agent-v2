"""walkie-agent-v2 HTTP client SDK.

Quick start::

    from client import WalkieAIClient, WalkieAPIError

    walkie = WalkieAIClient(base_url="http://localhost:5000")

    text    = walkie.stt.transcribe(audio_bytes)
    audio   = walkie.tts.synthesize("Hello!")
    objects = walkie.object_detection.detect(pil_image)
    poses   = walkie.pose_estimation.estimate(pil_image)
    caption = walkie.image_caption.caption(pil_image)
"""

from .base import WalkieAPIError
from .image_caption import ImageCaptionClient
from .object_detection import ObjectDetectionClient
from .pose_estimation import PoseEstimationClient
from .stt import STTClient
from .tts import TTSClient
from .walkie_client import WalkieAIClient

__all__ = [
    "WalkieAIClient",
    "WalkieAPIError",
    "STTClient",
    "TTSClient",
    "ObjectDetectionClient",
    "PoseEstimationClient",
    "ImageCaptionClient",
]
