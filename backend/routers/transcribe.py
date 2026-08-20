"""
Transcribe Router — Voice Search Endpoint

POST /api/v1/transcribe

Accepts multipart/form-data audio uploads, transcribes them using your
custom STT model, and returns the text. The temporary audio file is
deleted immediately after processing (in a finally block).
"""

import os
import tempfile
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import get_settings
from models.responses import TranscriptionResponse
from services.transcription import transcribe_audio

router = APIRouter(prefix="/api/v1", tags=["Voice Search"])
logger = logging.getLogger("sarathi.transcribe")


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(..., description="Audio file to transcribe")):
    """
    Transcribe an audio file to text using your custom STT model.

    Accepts common audio formats: WAV, MP3, OGG, WebM, MP4.
    The audio file is saved temporarily and deleted after processing.
    """
    settings = get_settings()

    # ── Validate file type ───────────────────────────────────────
    content_type = audio.content_type or ""
    if content_type not in settings.ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {content_type}. "
            f"Allowed: {', '.join(settings.ALLOWED_AUDIO_TYPES)}",
        )

    # ── Validate file size ───────────────────────────────────────
    content = await audio.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_AUDIO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({size_mb:.1f}MB). Maximum: {settings.MAX_AUDIO_SIZE_MB}MB.",
        )

    # ── Transcribe with cleanup ──────────────────────────────────
    tmp_path = None
    try:
        # Write to a named temp file (STT endpoint needs a file path)
        suffix = _extension_from_content_type(content_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await transcribe_audio(tmp_path)

        return TranscriptionResponse(
            text=result["text"],
            language=result.get("language"),
            duration_seconds=result.get("duration_seconds"),
        )

    except RuntimeError as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))

    finally:
        # Always delete the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.debug("Temp audio file deleted: %s", tmp_path)


def _extension_from_content_type(ct: str) -> str:
    """Map content type to file extension."""
    return {
        "audio/wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/mp4": ".m4a",
    }.get(ct, ".wav")
