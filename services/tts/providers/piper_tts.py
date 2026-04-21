"""Piper TTS provider implementation (local, offline)."""

from __future__ import annotations

from typing import Any, Iterator

from ..base import TTSProvider


class PiperProvider(TTSProvider):
    """Local Piper Text-to-Speech provider."""

    SUPPORTED_FORMATS = [
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        voice_path = config.get("voice_path")
        voice_name = config.get("voice_name", "en_US-lessac-medium")
        self.sample_rate = config.get("sample_rate", 22050)
        self.output_format = config.get("output_format", f"pcm_{self.sample_rate}")

        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise ImportError(
                "piper-tts is required for the Piper provider. "
                "Install with: pip install piper-tts"
            ) from exc

        if voice_path:
            self.voice = PiperVoice.load(voice_path)
        else:
            self.voice = PiperVoice.load(voice_name)

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        for chunk in self.voice.synthesize(text):
            yield chunk.audio_int16_bytes

    def get_supported_formats(self) -> list[str]:
        return self.SUPPORTED_FORMATS.copy()

    def supports_streaming(self) -> bool:
        return True
