"""
Transcription Service — Pluggable STT Interface

Provides a clean interface for speech-to-text transcription. This module
calls YOUR custom STT model endpoint. If no endpoint is configured, it
returns a clear error guiding you to set one up.

Your STT endpoint should accept a POST with multipart/form-data (key: "audio")
and return JSON: {"text": "...", "language": "...", "duration": 12.5}
"""

import logging
import httpx

from config import get_settings

logger = logging.getLogger("sarathi.transcription")


async def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using your custom STT model.

    Args:
        audio_path: Path to the temporary audio file.

    Returns:
        dict with keys: text, language, duration_seconds

    Raises:
        RuntimeError if no STT endpoint is configured or the call fails.
    """
    settings = get_settings()

    if not settings.STT_ENDPOINT:
        raise RuntimeError(
            "No STT_ENDPOINT configured. Set the STT_ENDPOINT environment variable "
            "to point at your custom speech-to-text model's HTTP endpoint."
        )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(audio_path, "rb") as f:
                resp = await client.post(
                    settings.STT_ENDPOINT,
                    files={"audio": ("audio", f)},
                )
                resp.raise_for_status()
                data = resp.json()

        result = {
            "text": data.get("text", "").strip(),
            "language": data.get("language"),
            "duration_seconds": data.get("duration"),
        }

        if not result["text"]:
            raise RuntimeError("STT model returned empty transcription.")

        logger.info(
            "Transcription complete: %d chars, language=%s",
            len(result["text"]),
            result["language"],
        )
        return result

    except httpx.HTTPError as exc:
        logger.error("STT endpoint call failed: %s", exc)
        raise RuntimeError(f"Transcription failed: {exc}") from exc
