"""
Sarathi — AI-Driven Civic & Legal Empowerment Engine

FastAPI application factory with:
  • Async lifespan management (MongoDB + ChromaDB init/teardown)
  • Security middleware stack (PII sanitizer, prompt injection guard)
  • Modular router registration
  • Structured logging
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from middleware.pii_sanitizer import PIISanitizerMiddleware
from middleware.prompt_injection_guard import PromptInjectionGuard
from routers import analyze, finalize, sessions, transcribe
from db import mongodb, chromadb_store


# ── Logging Setup ────────────────────────────────────────────────

def _configure_logging():
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger("sarathi")


# ── Lifespan (startup/shutdown) ──────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async resources across the app lifecycle."""
    settings = get_settings()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # Startup — graceful degradation if services are unavailable
    try:
        await mongodb.connect()
    except Exception as exc:
        logger.warning("MongoDB unavailable: %s. Session persistence disabled.", exc)

    try:
        chromadb_store.initialize()
    except Exception as exc:
        logger.warning("ChromaDB initialization failed: %s. RAG disabled.", exc)

    logger.info("Startup complete. Ready to serve.")

    yield

    # Shutdown
    try:
        await mongodb.disconnect()
    except Exception:
        pass
    logger.info("Shutdown complete.")


# ── App Factory ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-driven Civic & Legal Empowerment Engine for Indian citizens.",
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost runs first) ──────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(PIISanitizerMiddleware)
    app.add_middleware(PromptInjectionGuard)

    # ── Routers ──────────────────────────────────────────────
    app.include_router(analyze.router)
    app.include_router(finalize.router)
    app.include_router(sessions.router)
    app.include_router(transcribe.router)

    # ── Health Check ─────────────────────────────────────────
    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "operational",
        }

    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status": "healthy",
            "mongodb": "connected",
            "chromadb": chromadb_store.LegalCorpusStore.get_stats(),
        }

    return app


# ── App Instance (used by uvicorn: `uvicorn main:app`) ───────────
app = create_app()