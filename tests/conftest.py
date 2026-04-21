"""Shared pytest fixtures for walkie-agent-v2 API tests."""

import io
import struct

import pytest
from PIL import Image


BASE_URL = "http://localhost:5000"


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default=BASE_URL,
        help="Base URL of the running walkie-agent-v2 server",
    )


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A minimal 100x100 RGB PNG image."""
    img = Image.new("RGB", (100, 100), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_audio_bytes() -> bytes:
    """A minimal silent WAV file: 1 second, 16 kHz, mono, 16-bit PCM."""
    sample_rate = 16_000
    num_channels = 1
    bits_per_sample = 16
    num_samples = sample_rate  # 1 second of silence
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))                                          # chunk size
    buf.write(struct.pack("<H", 1))                                           # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
    buf.write(struct.pack("<H", num_channels * bits_per_sample // 8))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    buf.seek(0)
    return buf.getvalue()
