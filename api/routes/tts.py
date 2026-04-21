"""TTS (Text-to-Speech) blueprint — ElevenLabs provider, loaded at startup."""

from flask import Blueprint, Response, request, stream_with_context

from api.utils import error, success
from services.tts import TTS

bp = Blueprint("tts", __name__, url_prefix="/tts")

# _tts = TTS(provider="elevenlabs", voice_id="fDeOZu1sNd7qahm2fV4k")
_tts = TTS(provider="piper", voice_path="voices/en_US-lessac-medium.onnx")


def _infer_content_type(formats: list[str]) -> str:
    fmt_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "pcm": "audio/pcm",
    }
    for fmt in formats:
        mime = fmt_map.get(fmt.lower().split("_")[0])
        if mime:
            return mime
    return "application/octet-stream"


_audio_content_type = _infer_content_type(_tts.get_supported_formats())


@bp.get("/providers")
def list_providers():
    return success(TTS.available_providers())


@bp.post("/synthesize")
def synthesize():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    try:
        audio_bytes = _tts.synthesize(text)
    except Exception as exc:
        return error(str(exc), 500)

    return Response(audio_bytes, mimetype=_audio_content_type)


@bp.post("/synthesize-stream")
def synthesize_stream():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return error("Missing 'text' field")

    try:
        def generate():
            for chunk in _tts.synthesize_stream(text):
                yield chunk
    except Exception as exc:
        return error(str(exc), 500)

    return Response(stream_with_context(generate()), mimetype=_audio_content_type)
