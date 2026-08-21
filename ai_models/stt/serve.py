"""
STT Model Server — Speech-to-Text HTTP Endpoint

Serves the fine-tuned Whisper model as a FastAPI endpoint that the Sarathi
backend calls for voice transcription. This is the endpoint you set as
STT_ENDPOINT in the backend's .env file.

The server accepts audio file uploads and returns transcribed text with
language detection and duration.

Usage:
    python serve.py
    python serve.py --model ./output/sarathi-stt --port 9002

    # Or with the base Whisper model (no fine-tuning needed)
    python serve.py --model openai/whisper-small

Then set in backend/.env:
    STT_ENDPOINT=http://localhost:9002/transcribe
"""

import os
import time
import tempfile
import logging
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.stt.serve")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL = SCRIPT_DIR / "output" / "sarathi-stt"

# Module-level model references
_model = None
_processor = None


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Whisper model at startup."""
    global _model, _processor
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    model_path = app.state.model_path

    logger.info("Loading Whisper model from %s", model_path)
    _processor = WhisperProcessor.from_pretrained(str(model_path))
    _model = WhisperForConditionalGeneration.from_pretrained(str(model_path))

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    logger.info("Model loaded on %s", device)

    yield

    logger.info("STT server shutting down.")


def create_app(model_path: Path) -> FastAPI:
    app = FastAPI(
        title="Sarathi STT Server",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_path = model_path

    @app.post("/transcribe", response_model=TranscribeResponse)
    async def transcribe(audio: UploadFile = File(...)):
        """
        Transcribe an audio file to text.

        This endpoint matches the interface expected by the Sarathi backend's
        transcription service (services/transcription.py).
        """
        import torch
        import librosa

        # Validate file type
        allowed = {"audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4", "audio/x-wav"}
        if audio.content_type and audio.content_type not in allowed:
            raise HTTPException(415, f"Unsupported audio type: {audio.content_type}")

        # Save to temp file
        tmp_path = None
        try:
            content = await audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            start_time = time.time()

            # Load and resample audio to 16kHz
            audio_array, sr = librosa.load(tmp_path, sr=16000)
            duration = len(audio_array) / sr

            # Process
            input_features = _processor.feature_extractor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features

            device = next(_model.parameters()).device
            input_features = input_features.to(device)

            # Generate transcription
            with torch.no_grad():
                predicted_ids = _model.generate(
                    input_features,
                    max_length=448,
                    num_beams=5,
                    language="hi",  # Prioritize Hindi, but model will auto-detect
                    task="transcribe",
                )

            text = _processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()

            elapsed = time.time() - start_time
            logger.info(
                "Transcribed %.1fs audio in %.1fs: '%s...'",
                duration, elapsed, text[:60],
            )

            # Simple language detection based on script
            language = detect_language(text)

            return TranscribeResponse(
                text=text,
                language=language,
                duration=round(duration, 2),
            )

        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            raise HTTPException(500, f"Transcription failed: {str(exc)}")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
        }

    return app


def detect_language(text: str) -> str:
    """Simple heuristic language detection based on Unicode script."""
    devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return "unknown"

    hindi_ratio = devanagari_count / total_alpha
    if hindi_ratio > 0.5:
        return "hi"
    elif hindi_ratio > 0.1:
        return "hi-en"  # Code-mixed
    else:
        return "en"


def main():
    parser = argparse.ArgumentParser(description="Serve Sarathi STT model")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="Path to trained model or HuggingFace model name",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9002)
    args = parser.parse_args()

    # Check if model exists (local path) or is a HF model name
    model_path = args.model
    if not model_path.exists():
        model_str = str(model_path)
        if "/" in model_str and not model_str.startswith("/"):
            # Looks like a HuggingFace model name (e.g., "openai/whisper-small")
            logger.info("Using HuggingFace model: %s", model_str)
            model_path = Path(model_str)
        else:
            logger.error(
                "Model not found at %s.\n"
                "Options:\n"
                "  1. Train first: python train.py\n"
                "  2. Use base model: python serve.py --model openai/whisper-small",
                model_path,
            )
            return

    app = create_app(model_path)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
