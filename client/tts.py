"""HTTP client for the TTS (Text-to-Speech) API — /tts/*"""

from __future__ import annotations

from typing import Iterator

from .base import WalkieBaseClient


class TTSClient(WalkieBaseClient):
    """Client for the ``/tts`` blueprint.

    Mirrors the interface of :class:`services.tts.TTS`.

    Example::

        client = TTSClient()
        audio_bytes = client.synthesize("Hello, world!")

        for chunk in client.synthesize_stream("Hello, world!"):
            speaker.play(chunk)
    """

    def synthesize(self, text: str) -> bytes:
        """Synthesize *text* to audio and return the full audio as bytes.

        Args:
            text: The text to convert to speech.

        Returns:
            Audio data as bytes (format depends on server TTS provider).

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        resp = self._session.post(
            f"{self._base_url}/tts/synthesize",
            json={"text": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        if resp.headers.get("Content-Type", "").startswith("application/json"):
            # Error response came back as JSON
            self._unwrap(resp.json())
        return resp.content

    def synthesize_stream(self, text: str, chunk_size: int = 4096) -> Iterator[bytes]:
        """Synthesize *text* to audio, yielding chunks as they arrive.

        Args:
            text: The text to convert to speech.
            chunk_size: Number of bytes per chunk yielded.

        Yields:
            Raw audio bytes chunks.

        Raises:
            WalkieAPIError: If the server returns a failure response.
        """
        yield from self._post_json_stream(
            "/tts/synthesize-stream",
            payload={"text": text},
            chunk_size=chunk_size,
        )

    def available_providers(self) -> list[str]:
        """List all providers registered on the server."""
        return self._get("/tts/providers")
