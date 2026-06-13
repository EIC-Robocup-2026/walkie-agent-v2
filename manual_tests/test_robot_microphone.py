"""Manual hardware poke: list audio devices, then loop record→STT and print text.

Needs walkie-ai-server up at WALKIE_AI_BASE_URL and a working microphone.
Run: uv run python -m manual_tests.test_robot_microphone
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from walkie_config import load_config
from client import WalkieAIClient
from interfaces.devices.microphone import Microphone, list_audio_devices


def main() -> None:
    load_dotenv()
    load_config()

    walkieAI = WalkieAIClient(base_url=os.getenv("WALKIE_AI_BASE_URL", "http://localhost:5000"))
    microphone = Microphone()

    print("Available audio devices:")
    for device in list_audio_devices():
        print(device)

    while True:
        print("Say something...")
        audio = microphone.record_until_silence()
        text = walkieAI.stt.transcribe(audio)
        print("You said:", text)


if __name__ == "__main__":
    main()
