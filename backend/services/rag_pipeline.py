"""
RAG Pipeline — Step 2 of the Deterministic AI Pipeline

Retrieves relevant legal clauses and bare acts from ChromaDB based on the
classified intent and user input. Pure retrieval — no LLM call here.
"""

import logging

from models.domain import IntentResult, RAGResult
from db.chromadb_store import LegalCorpusStore

logger = logging.getLogger("sarathi.pipeline.rag")


async def retrieve_legal_context(
    user_input: str,
    intent: IntentResult,
) -> list[RAGResult]:
    """
    Query ChromaDB for relevant legal clauses.

    Combines the user's input with the intent summary for a richer query.
    Filters by legal domain when the classification confidence is high enough.
    """
    # Build a query that combines intent understanding with raw input
    query_text = f"{intent.summary}. {user_input}"

    # Only apply domain filter if confidence is strong
    domain_filter = (
        intent.domain.value if intent.confidence >= 0.7 else None
    )

    try:
        raw_results = LegalCorpusStore.query(
            text=query_text,
            domain_filter=domain_filter,
        )
    except RuntimeError:
        logger.warning("ChromaDB not available or empty. Returning no results.")
        return []

    results = [
        RAGResult(
            text=r["text"],
            act_name=r["act_name"],
            section=r["section"],
            relevance_score=r["relevance_score"],
        )
        for r in raw_results
        if r["relevance_score"] > 0.1  # Filter out noise
    ]

    logger.info(
        "RAG retrieved %d clauses for domain '%s'",
        len(results),
        intent.domain.value,
    )
    return results
