"""
ChromaDB Vector Store — Legal Corpus Retrieval

Persistent, in-process ChromaDB instance for zero-latency RAG.
Uses a pluggable embedding function — you provide your own embedding
model via the Ollama /api/embeddings endpoint (or any HTTP endpoint).

If the embedding endpoint is unavailable, falls back to a no-op
embedding that allows the system to start without a model.
"""

import logging
from typing import Any

import httpx
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from config import get_settings

logger = logging.getLogger("sarathi.db.chroma")

# ── Module-level store (initialized in lifespan) ─────────────────
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that calls your self-hosted model
    via Ollama's /api/embeddings endpoint.

    Plug in your own model by setting OLLAMA_EMBED_MODEL in .env.
    """

    def __init__(self, base_url: str, model: str, timeout: int = 30):
        self._url = f"{base_url}/api/embeddings"
        self._model = model
        self._timeout = timeout

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of documents."""
        embeddings: Embeddings = []

        for text in input:
            try:
                resp = httpx.post(
                    self._url,
                    json={"model": self._model, "prompt": text},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                embedding = resp.json().get("embedding", [])
                embeddings.append(embedding)
            except (httpx.HTTPError, KeyError) as exc:
                logger.warning(
                    "Embedding call failed for text (len=%d): %s. Using zero vector.",
                    len(text),
                    exc,
                )
                # Fallback: zero vector. Won't retrieve well but won't crash.
                embeddings.append([0.0] * 384)

        return embeddings


def initialize() -> None:
    """Initialize the persistent ChromaDB client and collection."""
    global _client, _collection
    settings = get_settings()

    _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)

    embed_fn = OllamaEmbeddingFunction(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_EMBED_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
    )

    _collection = _client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    count = _collection.count()
    logger.info(
        "ChromaDB initialized — collection '%s' has %d documents.",
        settings.CHROMA_COLLECTION,
        count,
    )


def _get_collection() -> chromadb.Collection:
    if _collection is None:
        raise RuntimeError("ChromaDB not initialized. Call initialize() first.")
    return _collection


# ── Query & Ingest ───────────────────────────────────────────────


class LegalCorpusStore:
    """Query and manage the legal document corpus."""

    @staticmethod
    def query(
        text: str,
        n_results: int | None = None,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant legal clauses.

        Returns a list of dicts: {text, act_name, section, score}
        """
        settings = get_settings()
        col = _get_collection()
        k = n_results or settings.RAG_TOP_K

        where_filter = {"domain": domain_filter} if domain_filter else None

        results = col.query(
            query_texts=[text],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "text": doc,
                "act_name": meta.get("act_name", "Unknown"),
                "section": meta.get("section", "Unknown"),
                "relevance_score": round(1 - dist, 4),  # cosine → similarity
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    @staticmethod
    def add_documents(
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Ingest documents into the legal corpus collection."""
        col = _get_collection()
        col.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Added %d documents to ChromaDB.", len(documents))

    @staticmethod
    def get_stats() -> dict:
        """Return collection statistics."""
        col = _get_collection()
        return {
            "collection": col.name,
            "document_count": col.count(),
        }
