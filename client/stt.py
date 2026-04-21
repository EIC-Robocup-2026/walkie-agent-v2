"""HTTP client for the STT (Speech-to-Text) API — /stt/*"""

from __future__ import annotations

from .base import WalkieBaseClient


class STTClient(WalkieBaseClient):
    """Client for the ``/stt`` blueprint.

    Mirrors the interface of :class:`services.stt.STT`.

    Example::

        client = STTClient()
        text = client.transcribe(audio_bytes)
    """

    def transcribe(self, audio_content: bytes) -> str:
        """Transcribe audio bytes to text.

        Args:
            audio_content: Raw audio data (WAV, MP3, etc.) as bytes.

        Returns:
            Transcribed text string.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        data = self._post_files(
            "/stt/transcribe",
            files={"audio": ("audio.wav", audio_content, "audio/wav")},
        )
        return data["transcription"]

    def available_providers(self) -> list[str]:
        """List all providers registered on the server."""
        return self._get("/stt/providers")
