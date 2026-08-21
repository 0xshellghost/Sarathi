"""
Embedding Model Server — Alternative to Ollama

Serves the fine-tuned sentence-transformer embedding model as a standalone
FastAPI endpoint. Use this if you prefer not to serve embeddings through Ollama.

The endpoint matches the interface expected by the backend's ChromaDB store
(OllamaEmbeddingFunction), so you can point OLLAMA_BASE_URL at this server
or modify the backend to call this directly.

Usage:
    python serve.py
    python serve.py --model ./output/sarathi-embed --port 9001

The server exposes:
    POST /api/embeddings  (Ollama-compatible)
    POST /embed           (batch endpoint)
    GET  /health
"""

import logging
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sarathi.embed.serve")

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL = SCRIPT_DIR / "output" / "sarathi-embed"

# Module-level model reference
_model = None


class EmbedRequest(BaseModel):
    """Ollama-compatible embedding request."""
    model: str = "sarathi-embed"
    prompt: str


class BatchEmbedRequest(BaseModel):
    """Batch embedding request."""
    texts: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup."""
    global _model
    from sentence_transformers import SentenceTransformer

    model_path = app.state.model_path
    logger.info("Loading embedding model from %s", model_path)
    _model = SentenceTransformer(str(model_path))

    dim = _model.get_sentence_embedding_dimension()
    logger.info("Model loaded — %d-dimensional embeddings", dim)

    yield

    logger.info("Embedding server shutting down.")


def create_app(model_path: Path) -> FastAPI:
    app = FastAPI(
        title="Sarathi Embedding Server",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.model_path = model_path

    @app.post("/api/embeddings")
    async def ollama_compatible_embed(req: EmbedRequest):
        """
        Ollama-compatible embedding endpoint.
        Matches the /api/embeddings interface so the backend's
        OllamaEmbeddingFunction works without modification.
        """
        embedding = _model.encode(req.prompt).tolist()
        return {"embedding": embedding}

    @app.post("/embed")
    async def batch_embed(req: BatchEmbedRequest):
        """Batch embedding endpoint for efficiency."""
        embeddings = _model.encode(req.texts).tolist()
        return {"embeddings": embeddings}

    @app.get("/health")
    async def health():
        dim = _model.get_sentence_embedding_dimension() if _model else 0
        return {
            "status": "healthy",
            "model_loaded": _model is not None,
            "embedding_dimension": dim,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve Sarathi embedding model")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Path to the trained embedding model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    args = parser.parse_args()

    if not args.model.exists():
        logger.error(
            "Model not found at %s. Train first with: python train.py",
            args.model,
        )
        return

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
