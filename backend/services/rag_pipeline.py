"""
RAG Pipeline — Step 2 of the Deterministic AI Pipeline

Retrieves relevant legal clauses from TWO sources:
  1. ChromaDB (offline vector store with pre-seeded legal corpus)
  2. Live web search (official Indian government sites via DuckDuckGo)

Results from both sources are merged, deduplicated, and sorted by
relevance before being passed to the LLM for explanation.

If either source fails, the other still provides context — the pipeline
is resilient by design.
"""

import asyncio
import logging

from models.domain import IntentResult, RAGResult
from db.chromadb_store import LegalCorpusStore
from services.web_fetcher import search_indian_gov

logger = logging.getLogger("sarathi.pipeline.rag")


async def _query_chromadb(
    query_text: str,
    domain_filter: str | None,
) -> list[RAGResult]:
    """Query the local ChromaDB vector store (offline corpus)."""
    try:
        raw_results = LegalCorpusStore.query(
            text=query_text,
            domain_filter=domain_filter,
        )
    except RuntimeError:
        logger.warning("ChromaDB not available or empty. Returning no local results.")
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

    logger.info("ChromaDB returned %d local results.", len(results))
    return results


async def _query_web(
    query_text: str,
    domain: str | None,
) -> list[RAGResult]:
    """Search official Indian government sites for live legal data."""
    try:
        results = await search_indian_gov(query_text, domain=domain)
        logger.info("Web fetcher returned %d results.", len(results))
        return results
    except Exception as exc:
        logger.error("Web search failed entirely: %s. Continuing without web results.", exc)
        return []


def _deduplicate_results(results: list[RAGResult]) -> list[RAGResult]:
    """
    Remove near-duplicate results based on act_name + section overlap.

    When both ChromaDB and the web return the same section, prefer the
    higher-confidence result (typically the local curated one).
    """
    seen: dict[str, RAGResult] = {}

    for r in results:
        key = f"{r.act_name}::{r.section}".lower().strip()
        if key in seen:
            # Keep the one with higher relevance_score
            if r.relevance_score > seen[key].relevance_score:
                seen[key] = r
        else:
            seen[key] = r

    return list(seen.values())


async def retrieve_legal_context(
    user_input: str,
    intent: IntentResult,
) -> list[RAGResult]:
    """
    Query both ChromaDB AND live government sites concurrently.

    Combines the user's input with the intent summary for a richer query.
    Filters by legal domain when the classification confidence is high enough.
    Results from both sources are merged, deduplicated, and sorted.
    """
    # Build a query that combines intent understanding with raw input
    query_text = f"{intent.summary}. {user_input}"

    # Only apply domain filter if confidence is strong
    domain_filter = (
        intent.domain.value if intent.confidence >= 0.7 else None
    )

    # ── Run both sources concurrently ────────────────────────────
    local_results, web_results = await asyncio.gather(
        _query_chromadb(query_text, domain_filter),
        _query_web(query_text, domain_filter),
    )

    # ── Merge & deduplicate ──────────────────────────────────────
    combined = local_results + web_results
    deduplicated = _deduplicate_results(combined)

    # Sort by relevance (highest first)
    deduplicated.sort(key=lambda r: r.relevance_score, reverse=True)

    logger.info(
        "RAG retrieved %d total results (%d local + %d web, %d after dedup) for domain '%s'",
        len(deduplicated),
        len(local_results),
        len(web_results),
        len(deduplicated),
        intent.domain.value,
    )
    return deduplicated
