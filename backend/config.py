"""
Sarathi — Application Configuration

All settings are loaded from environment variables or a .env file.
No paid API keys required — models are self-hosted via Ollama.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration — every tunable knob lives here."""

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "Sarathi"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:3000"])

    # ── Ollama (model-agnostic LLM server) ───────────────────────────
    # Point this at any OpenAI-compatible API serving your custom model.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "sarathi-legal"  # Your custom model name
    OLLAMA_EMBED_MODEL: str = "sarathi-embed"  # Your custom embedding model
    OLLAMA_TIMEOUT: int = 120  # seconds — generous for large models
    LLM_TEMPERATURE: float = 0.1  # Low temp for deterministic legal output
    LLM_MAX_TOKENS: int = 2048
    LLM_MAX_RETRIES: int = 3

    # ── MongoDB ──────────────────────────────────────────────────────
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "sarathi"

    # ── ChromaDB ─────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION: str = "indian_legal_corpus"
    RAG_TOP_K: int = 5

    # ── Transcription ────────────────────────────────────────────────
    # Your custom STT model endpoint (or leave empty to use the stub).
    STT_ENDPOINT: str = ""
    MAX_AUDIO_SIZE_MB: int = 25
    ALLOWED_AUDIO_TYPES: list[str] = Field(
        default=["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/mp4"]
    )

    # ── Security ─────────────────────────────────────────────────────
    INJECTION_THREAT_THRESHOLD: float = 0.6
    PII_REDACTION_ENABLED: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance, cached for the process lifetime."""
    return Settings()
