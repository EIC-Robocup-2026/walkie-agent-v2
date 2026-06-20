"""walkie-agent-v2 HTTP client SDK.

Quick start::

    from client import WalkieAIClient, WalkieAPIError

    walkie = WalkieAIClient(base_url="http://localhost:5000")

    text  = walkie.stt.transcribe(audio_bytes)
    audio = walkie.tts.synthesize("Hello!")

    # Vision — one image upload, any combination of tasks
    res = walkie.image.process(pil_image, detection=True, caption=True, pose=True)
    objects = walkie.image.detect(pil_image)
    poses   = walkie.image.estimate_poses(pil_image)
    caption = walkie.image.caption(pil_image)
"""

from .base import WalkieAPIError
from .grasp import GraspClient, GraspPose
from .image import (
    DetectedObject,
    FaceEmbedding,
    ImageClient,
    ImageResult,
    PersonPose,
    PoseKeypoint,
)
from .stt import STTClient
from .tts import TTSClient
from .walkie_client import WalkieAIClient

__all__ = [
    "WalkieAIClient",
    "WalkieAPIError",
    "STTClient",
    "TTSClient",
    "ImageClient",
    "ImageResult",
    "DetectedObject",
    "PersonPose",
    "PoseKeypoint",
    "FaceEmbedding",
    "GraspClient",
    "GraspPose",
]
