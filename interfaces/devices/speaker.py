"""Speaker for audio playback with streaming support using PyAudio."""

import contextlib
import io
import os
import threading
import time
from typing import TYPE_CHECKING, Callable, Iterator

import pyaudio

if TYPE_CHECKING:
    from .microphone import Microphone


@contextlib.contextmanager
def _suppress_stderr_fd() -> Iterator[None]:
    """Silence ALSA/JACK probe noise on stderr while PortAudio initializes."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = 2
    saved = os.dup(stderr_fd)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        os.close(devnull)


def _pyaudio_output_stream(
    *,
    sample_rate: int,
    channels: int,
    frames_per_buffer: int,
    device: int | None,
):
    """Create PyAudio instance and output stream with minimal stderr noise."""
    with _suppress_stderr_fd():
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            output=True,
            frames_per_buffer=frames_per_buffer,
            output_device_index=device,
        )
    return p, stream


def list_output_devices() -> list[dict]:
    """List available audio output devices.
    
    Returns:
        List of device info dicts with id, name, channels, and sample_rate.
    """
    with _suppress_stderr_fd():
        p = pyaudio.PyAudio()
        try:
            result = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev["maxOutputChannels"] > 0:
                    result.append({
                        "id": i,
                        "name": dev["name"],
                        "channels": dev["maxOutputChannels"],
                        "sample_rate": int(dev["defaultSampleRate"]),
                        "is_default": i == p.get_default_output_device_info()["index"],
                    })
            return result
        finally:
            p.terminate()


def print_output_devices() -> None:
    """Print available audio output devices in a readable format."""
    devices = list_output_devices()
    print("Available output devices:")
    print("-" * 60)
    for dev in devices:
        default = " (default)" if dev["is_default"] else ""
        print(f"  [{dev['id']}] {dev['name']}{default}")
        print(f"      Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']}")
    print("-" * 60)


def _parse_format(format_str: str) -> tuple[str, int, int]:
    """Parse format string into (codec, sample_rate, bitrate).
    
    Args:
        format_str: Format string like "mp3_44100_128" or "pcm_16000"
        
    Returns:
        Tuple of (codec, sample_rate, bitrate or 0)
    """
    parts = format_str.split("_")
    codec = parts[0]
    sample_rate = int(parts[1]) if len(parts) > 1 else 44100
    bitrate = int(parts[2]) if len(parts) > 2 else 0
    return codec, sample_rate, bitrate


class Speaker:
    """Audio playback with support for streaming and non-streaming modes using PyAudio.
    
    Supports PCM audio format directly. MP3 requires pydub and ffmpeg for decoding.
    """

    def __init__(
        self,
        device: int | None = None,
        sample_rate: int = 24000,
        channels: int = 1,
        frames_per_buffer: int = 1024,
        mic: "Microphone | None" = None,
    ) -> None:
        """Initialize speaker.

        Args:
            device: Audio output device index. Use list_output_devices() to see options.
                   If None, uses the default output device.
            sample_rate: Default sample rate for playback.
            channels: Number of audio channels (1 for mono, 2 for stereo).
            frames_per_buffer: Buffer size for streaming playback.
            mic: Microphone to pause for the duration of playback, so the robot
                never transcribes its own voice. Optional; no-op when None.
        """
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.mic = mic
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @contextlib.contextmanager
    def _mic_muted(self) -> Iterator[None]:
        """Pause the linked mic (if any) for the duration of playback."""
        if self.mic is None:
            yield
            return
        with self.mic.paused():
            yield

    def play(
        self,
        audio_data: bytes,
        format: str = "pcm_24000",
        blocking: bool = True,
    ) -> None:
        """Play complete audio.

        Args:
            audio_data: Audio data as bytes.
            format: Audio format string (e.g., "pcm_24000", "mp3_44100_128").
            blocking: If True, blocks until playback completes. If False, plays
                in a background thread and returns immediately.
        """
        if blocking:
            self._play(audio_data, format)
            return

        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play,
            args=(audio_data, format),
            daemon=True,
        )
        self._thread.start()

    def _play(self, audio_data: bytes, format: str) -> None:
        codec, sample_rate, _ = _parse_format(format)

        if codec == "mp3":
            try:
                from pydub import AudioSegment
            except ImportError as e:
                raise RuntimeError(
                    "pydub is required for MP3 playback. "
                    "Install with: pip install pydub\n"
                    "Also ensure ffmpeg is installed on your system."
                ) from e
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
            sample_rate = audio_segment.frame_rate
            audio_data = audio_segment.raw_data

        p, stream = _pyaudio_output_stream(
            sample_rate=sample_rate,
            channels=self.channels,
            frames_per_buffer=self.frames_per_buffer,
            device=self.device,
        )

        # Chunk writes so the stop event can interrupt mid-playback.
        bytes_per_frame = 2 * self.channels  # paInt16 = 2 bytes per sample
        chunk_bytes = self.frames_per_buffer * bytes_per_frame
        try:
            with self._mic_muted():  # don't let the mic transcribe our own voice
                for offset in range(0, len(audio_data), chunk_bytes):
                    if self._stop_event.is_set():
                        break
                    stream.write(audio_data[offset:offset + chunk_bytes])
                drain_time = (self.frames_per_buffer / sample_rate) + 0.1
                time.sleep(drain_time)
        finally:
            stream.close()
            p.terminate()

    def play_stream(
        self,
        audio_stream: Iterator[bytes],
        format: str = "pcm_24000",
        stream_handler: Callable[[bytes], None] | None = None,
        blocking: bool = True,
    ) -> bytes:
        """Play audio chunks in real-time as they arrive.

        Args:
            audio_stream: Iterator yielding audio chunks.
            format: Audio format string (e.g., "pcm_24000").
            stream_handler: Optional callback for each chunk (e.g., for logging).
            blocking: If True, blocks until the stream is exhausted and returns
                the full audio. If False, plays in a background thread and
                returns b"" immediately.

        Returns:
            Complete audio data as bytes when blocking=True, else b"".
        """
        if blocking:
            return self._play_stream(audio_stream, format, stream_handler)

        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._play_stream,
            args=(audio_stream, format, stream_handler),
            daemon=True,
        )
        self._thread.start()
        return b""

    def _play_stream(
        self,
        audio_stream: Iterator[bytes],
        format: str,
        stream_handler: Callable[[bytes], None] | None,
    ) -> bytes:
        codec, sample_rate, _ = _parse_format(format)

        if codec != "pcm":
            raise ValueError(
                f"Streaming playback only supports PCM format, got '{codec}'. "
                "Use output_format like 'pcm_24000' for streaming."
            )

        if stream_handler is None:
            stream_handler = lambda x: None

        p, stream = _pyaudio_output_stream(
            sample_rate=sample_rate,
            channels=self.channels,
            frames_per_buffer=self.frames_per_buffer,
            device=self.device,
        )

        audio = b""
        try:
            with self._mic_muted():  # don't let the mic transcribe our own voice
                for chunk in audio_stream:
                    if self._stop_event.is_set():
                        break
                    # print(f"Playing chunk: {len(chunk)} bytes")
                    stream.write(chunk)
                    stream_handler(chunk)
                    audio += chunk
                drain_time = (self.frames_per_buffer / sample_rate) + 0.1
                time.sleep(drain_time)
        finally:
            stream.close()
            p.terminate()

        return audio

    def stop(self) -> None:
        """Stop any currently playing audio started with blocking=False."""
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)
        self._thread = None
