"""Microphone input with Voice Activity Detection using Silero VAD."""

import os
import threading
import time
import wave
from contextlib import contextmanager
from typing import Iterator

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
        debug_save_dir: str | None = None,
    ) -> None:
        """Initialize microphone with VAD.

        Args:
            device: Audio device ID or name. Use list_audio_devices() to see options.
            threshold: VAD sensitivity (0.0-1.0). Higher = less sensitive.
            min_silence_duration_ms: Silence duration to end speech segment.
            speech_pad_ms: Padding around detected speech.
            debug_save_dir: When set (or the WALKIE_MIC_DEBUG_DIR env var is set),
                every record_until_silence result is written to a timestamped WAV
                here and its duration/peak amplitude printed — so you can tell a
                silent capture (mic/pause bug) from a healthy one (STT bug).
        """
        self.device = device
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.debug_save_dir = debug_save_dir or os.getenv("WALKIE_MIC_DEBUG_DIR")
        self._debug_counter = 0
        
        # Get device sample rate
        if device is not None:
            dev_info = sd.query_devices(device)
            self.device_sample_rate = int(dev_info["default_samplerate"])
        else:
            dev_info = sd.query_devices(sd.default.device[0])
            self.device_sample_rate = int(dev_info["default_samplerate"])
        # Log which input the mic actually opened — the #1 cause of "STT hears
        # nothing" is capturing from the wrong/empty default device. Set
        # WALKIE_MIC_DEVICE (index or name substring) to pin the real mic.
        print(f"[microphone] using input device {device if device is not None else 'default'}"
              f" -> '{dev_info['name']}' @ {self.device_sample_rate} Hz"
              f" (in_channels={dev_info['max_input_channels']})")
        
        # Calculate chunk size at device rate that corresponds to 512 samples at 16kHz
        # We use a slightly larger chunk to ensure we always have enough after resampling
        self.chunk_size = int(self.VAD_CHUNK_SAMPLES * self.device_sample_rate / self.VAD_SAMPLE_RATE)
        
        # Load Silero VAD model
        self.model = load_silero_vad(onnx=True)
        self._reset_vad()

        # Pause state: while paused, an in-flight record_until_silence drops every
        # captured chunk (so nothing is recorded and the VAD never triggers) and
        # stops counting the paused span against its timeout. Re-entrant via a
        # depth counter so overlapping pause()/resume() calls (e.g. nested speech)
        # don't un-pause early. The speaker holds this paused while it plays so the
        # robot never transcribes its own voice.
        self._pause_lock = threading.Lock()
        self._pause_depth = 0
        self._paused = threading.Event()

    def pause(self) -> None:
        """Suspend feeding captured audio to any active recording until resume().

        Re-entrant: nested pause() calls stack, and the mic only un-pauses once a
        matching number of resume() calls arrive. Cheap and safe to call when no
        recording is running — it just sets a flag.
        """
        print("============== Microphone paused ==============")
        with self._pause_lock:
            self._pause_depth += 1
            self._paused.set()

    def resume(self) -> None:
        """Undo one pause(); the mic resumes once every pause() has been matched."""
        print("============== Microphone resumed ==============")
        with self._pause_lock:
            if self._pause_depth > 0:
                self._pause_depth -= 1
            if self._pause_depth == 0:
                self._paused.clear()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Context manager: pause() on enter, resume() on exit (exception-safe)."""
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _reset_vad(self) -> None:
        """Reset VAD iterator for new recording session."""
        self.vad_iterator = VADIterator(
            self.model,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

    def _save_debug_wav(self, data: bytes, label: str = "rec") -> None:
        """Write 16kHz/16-bit mono `data` to the debug dir and print its stats.

        Best-effort: a save failure is logged, never raised. The printed peak
        amplitude (0–32767) is the quick tell — near-zero means the capture
        itself was silent (a mic/pause problem), a healthy peak points at STT.
        """
        if not self.debug_save_dir:
            return
        try:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            self._debug_counter += 1
            path = os.path.join(self.debug_save_dir, f"{label}_{self._debug_counter:04d}.wav")
            arr = np.frombuffer(data, dtype=np.int16)
            peak = int(np.abs(arr).max()) if arr.size else 0
            duration = arr.size / self.VAD_SAMPLE_RATE
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(self.VAD_SAMPLE_RATE)
                wf.writeframes(data)
            print(
                f"[mic] debug saved {duration:.2f}s, {arr.size} samples, "
                f"peak {peak}/32767 -> {path}"
            )
        except Exception as exc:
            print(f"[mic] debug save failed ({exc})")

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
        print("============== Microphone recording started ==============")
        self._reset_vad()
        audio_chunks: list[np.ndarray] = []
        speech_started = False
        speech_ended = False
        recording_start_time = None
        pause_started = None   # time.time() of the current pause span, or None
        paused_total = 0.0     # accumulated paused seconds (excluded from timeout)

        def callback(indata, frames, time_info, status):
            nonlocal speech_started, speech_ended, recording_start_time
            nonlocal pause_started, paused_total
            if speech_ended:
                return

            if self._paused.is_set():
                # Paused (e.g. the robot is speaking): drop this chunk so its own
                # voice is neither recorded nor seen by the VAD, and note when the
                # pause began so the loop can exclude it from the timeout.
                if pause_started is None:
                    pause_started = time.time()
                return
            if pause_started is not None:
                # Just resumed: bank the paused span and reset the VAD so any
                # trailing playback doesn't bleed into the next detection.
                paused_total += time.time() - pause_started
                pause_started = None
                self._reset_vad()
                speech_started = False

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
            while not speech_ended:
                # Don't let paused time (the robot speaking) eat the timeout — a
                # phrase spoken right after resume() should still get its full window.
                in_pause = (time.time() - pause_started) if pause_started is not None else 0.0
                if (time.time() - start_time - paused_total - in_pause) >= timeout:
                    break
                sd.sleep(10)

        # Combine chunks and resample to 16kHz for output
        if audio_chunks:
            audio = np.concatenate(audio_chunks)
            audio_16k = _resample(audio, self.device_sample_rate, self.VAD_SAMPLE_RATE)
            data = audio_16k.tobytes()
        else:
            data = b""
        self._save_debug_wav(data)
        return data

    def record_seconds(self, duration: float) -> bytes:
        """Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds.
            
        Returns:
            Audio data as bytes (16-bit PCM, 16kHz mono).
        """
        print(f"============== Microphone recording for {duration:.2f} seconds ==============")
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
        data = audio_16k.tobytes()
        self._save_debug_wav(data, label="fixed")
        return data

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


def _diagnostic() -> None:
    """Quick mic check: list devices, then record a few seconds from the chosen
    one and report the audio level — a near-zero level means the device captures
    nothing (the usual "STT hears nothing" cause).

        python -m interfaces.devices.microphone            # default device
        WALKIE_MIC_DEVICE=fifine python -m interfaces.devices.microphone
        python -m interfaces.devices.microphone 4          # device index 4
    """
    import os
    import sys

    print_audio_devices(input_only=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else (os.getenv("WALKIE_MIC_DEVICE") or "").strip()
    device: int | str | None = None
    if arg:
        device = int(arg) if arg.lstrip("-").isdigit() else arg
    secs = 4.0
    mic = Microphone(device=device)
    print(f"\nRecording {secs:.0f}s — say something now...")
    raw = mic.record_seconds(secs)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if not len(audio):
        print("  !! captured 0 samples — device opened but returned nothing")
        return
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"  samples={len(audio)} rms={rms:.4f} peak={peak:.4f}")
    if rms < 1e-3:
        print("  !! essentially silent — wrong device, muted, or unplugged. "
              "Try another WALKIE_MIC_DEVICE (index or name above).")
    else:
        print("  OK — audio captured. If STT still fails, the issue is downstream "
              "(VAD threshold / walkie-ai-server STT).")


if __name__ == "__main__":
    _diagnostic()
