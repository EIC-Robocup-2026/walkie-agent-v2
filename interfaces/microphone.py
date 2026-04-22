"""Microphone input with Voice Activity Detection using Silero VAD."""

import time

import numpy as np
import sounddevice as sd
from scipy import signal
from silero_vad import VADIterator, load_silero_vad


def list_audio_devices(input_only: bool = True) -> list[dict]:
    """List available audio devices.
    
    Args:
        input_only: If True, only show input devices (microphones).
        
    Returns:
        List of device info dicts with id, name, channels, and sample_rate.
    """
    devices = sd.query_devices()
    result = []
    
    for i, dev in enumerate(devices):
        if input_only and dev["max_input_channels"] == 0:
            continue
        result.append({
            "id": i,
            "name": dev["name"],
            "channels": dev["max_input_channels"],
            "sample_rate": dev["default_samplerate"],
            "is_default": i == sd.default.device[0],
        })
    
    return result


def print_audio_devices(input_only: bool = True) -> None:
    """Print available audio devices in a readable format.
    
    Args:
        input_only: If True, only show input devices (microphones).
    """
    devices = list_audio_devices(input_only)
    print("Available audio devices:")
    print("-" * 60)
    for dev in devices:
        default = " (default)" if dev["is_default"] else ""
        print(f"  [{dev['id']}] {dev['name']}{default}")
        print(f"      Channels: {dev['channels']}, Sample Rate: {dev['sample_rate']}")
    print("-" * 60)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Audio samples as numpy array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio, preserving original dtype.
    """
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    # Preserve dtype but handle float types properly
    if audio.dtype in (np.float32, np.float64):
        return resampled.astype(audio.dtype)
    else:
        # Clip to avoid overflow when converting back to int
        return np.clip(resampled, np.iinfo(audio.dtype).min, np.iinfo(audio.dtype).max).astype(audio.dtype)


class Microphone:
    """Microphone recorder with Voice Activity Detection.
    
    Uses Silero VAD to detect speech and automatically start/stop recording.
    Automatically resamples audio from any sample rate to 16kHz.
    """

    VAD_SAMPLE_RATE = 16000  # Silero VAD requires 16kHz
    VAD_CHUNK_SAMPLES = 512  # Silero VAD requires exactly 512 samples at 16kHz

    def __init__(
        self,
        device: int | str | None = None,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 1000,
        speech_pad_ms: int = 1000,
    ) -> None:
        """Initialize microphone with VAD.
        
        Args:
            device: Audio device ID or name. Use list_audio_devices() to see options.
            threshold: VAD sensitivity (0.0-1.0). Higher = less sensitive.
            min_silence_duration_ms: Silence duration to end speech segment.
            speech_pad_ms: Padding around detected speech.
        """
        self.device = device
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # Get device sample rate
        if device is not None:
            dev_info = sd.query_devices(device)
            self.device_sample_rate = int(dev_info["default_samplerate"])
        else:
            self.device_sample_rate = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
        
        # Calculate chunk size at device rate that corresponds to 512 samples at 16kHz
        # We use a slightly larger chunk to ensure we always have enough after resampling
        self.chunk_size = int(self.VAD_CHUNK_SAMPLES * self.device_sample_rate / self.VAD_SAMPLE_RATE)
        
        # Load Silero VAD model
        self.model = load_silero_vad(onnx=True)
        self._reset_vad()

    def _reset_vad(self) -> None:
        """Reset VAD iterator for new recording session."""
        self.vad_iterator = VADIterator(
            self.model,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def _resample_to_vad_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to exactly 512 samples at 16kHz for VAD.
        
        Args:
            audio: Audio samples (int16 or float32).
            
        Returns:
            Audio as float32 normalized to [-1, 1], exactly 512 samples.
        """
        # Convert to float32 and normalize to [-1, 1] range
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32) / 32768.0
        
        # Resample to 16kHz
        if self.device_sample_rate != self.VAD_SAMPLE_RATE:
            resampled = _resample(audio_float, self.device_sample_rate, self.VAD_SAMPLE_RATE)
        else:
            resampled = audio_float
        
        # Ensure exactly 512 samples (truncate or pad)
        if len(resampled) > self.VAD_CHUNK_SAMPLES:
            resampled = resampled[:self.VAD_CHUNK_SAMPLES]
        elif len(resampled) < self.VAD_CHUNK_SAMPLES:
            resampled = np.pad(resampled, (0, self.VAD_CHUNK_SAMPLES - len(resampled)))
        
        return resampled.astype(np.float32)

    def record_until_silence(
        self,
        timeout: float = 30.0,
        min_duration: float = 2.0,
        wait_for_speech: bool = True,
    ) -> bytes:
        """Record audio until speech ends (silence detected).
        
        Args:
            timeout: Maximum recording duration in seconds.
            min_duration: Minimum recording duration in seconds before silence
                detection can stop recording.
            wait_for_speech: If True, only stop after speech was detected and ended.
                If False, can stop on any silence after min_duration.
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz mono).
        """
        self._reset_vad()
        audio_chunks: list[np.ndarray] = []
        speech_started = False
        speech_ended = False
        recording_start_time = None

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, speech_ended, recording_start_time
            if speech_ended:
                return
            
            # Track when recording actually starts
            if recording_start_time is None:
                recording_start_time = time.time()
            
            # Store original audio
            chunk = indata[:, 0].copy()
            audio_chunks.append(chunk)
            
            # Resample to exactly 512 samples for VAD
            vad_chunk = self._resample_to_vad_chunk(chunk)
            
            # Process with VAD
            result = self.vad_iterator(vad_chunk)
            if result:
                if "start" in result:
                    speech_started = True
                if "end" in result:
                    elapsed = time.time() - recording_start_time
                    # Only end if:
                    # 1. Minimum duration has passed AND
                    # 2. Either we don't need to wait for speech, or speech was detected
                    if elapsed >= min_duration and (not wait_for_speech or speech_started):
                        speech_ended = True

        # Record audio at device's native sample rate
        with sd.InputStream(
            device=self.device,
            samplerate=self.device_sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=self.chunk_size,
            callback=callback,
        ):
            start_time = time.time()
            while not speech_ended and (time.time() - start_time) < timeout:
                sd.sleep(10)

        # Combine chunks and resample to 16kHz for output
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            audio_16k = _resample(audio, self.device_sample_rate, self.VAD_SAMPLE_RATE)
            return audio_16k.tobytes()
        return b""

    def record_seconds(self, duration: float) -> bytes:
        """Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds.
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz mono).
        """
        samples = int(self.device_sample_rate * duration)
        audio = sd.rec(
            samples,
            samplerate=self.device_sample_rate,
            channels=1,
            dtype=np.int16,
            device=self.device,
        )
        sd.wait()
        
        # Resample to 16kHz for output
        audio_16k = _resample(audio[:, 0], self.device_sample_rate, self.VAD_SAMPLE_RATE)
        return audio_16k.tobytes()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio samples as numpy array (at 16kHz, 512 samples).
            
        Returns:
            True if speech detected.
        """
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        
        result = self.vad_iterator(audio_chunk)
        return result is not None and "start" in result
