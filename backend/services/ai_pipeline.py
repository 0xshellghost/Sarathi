"""
AI Pipeline Orchestrator — Deterministic 4-Step Sequence

Runs the full pipeline: Intent Router → RAG → Entity Extractor → Output Formatter.
Yields SSE events as an async generator — tokens stream from the LLM explanation,
then a form_request event delivers the entity schema, and finally a complete event.

STRICTLY NO AUTONOMOUS AGENTS. No loops. No self-directed tool use.
Each step receives the output of the previous step — fully deterministic.
"""

import json
import logging
from typing import AsyncGenerator

from db.mongodb import SessionStore
from services.intent_router import classify_intent
from services.rag_pipeline import retrieve_legal_context
from services.entity_extractor import extract_entities
from services.output_formatter import build_form_schema
from services.llm_client import get_llm_client
from models.domain import IntentResult, RAGResult

logger = logging.getLogger("sarathi.pipeline")


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def run_pipeline(
    user_input: str,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Execute the full deterministic AI pipeline and yield SSE events.

    Event sequence sent to the frontend:
    1. Multiple `{type: "token", text: "..."}` events (streamed explanation)
    2. One `{type: "form_request", schema: {...}}` event (entity form)
    3. One `{type: "complete", session_id: "..."}` event

    On error, yields `{type: "error", message: "...", recoverable: bool}`.
    """
    # ── Create or resume session ─────────────────────────────────
    try:
        session = await SessionStore.create(user_input, session_id)
        sid = session["session_id"]
    except Exception as exc:
        logger.error("Session creation failed: %s", exc)
        sid = session_id or "temp_session"

    try:
        # ── Step 1: Intent Router ────────────────────────────────
        logger.info("[%s] Step 1: Classifying intent...", sid)
        intent: IntentResult = await classify_intent(user_input)

        await SessionStore.update(
            sid,
            intent={"domain": intent.domain.value, "confidence": intent.confidence, "summary": intent.summary},
        )

        # ── Step 2: RAG Retrieval ────────────────────────────────
        logger.info("[%s] Step 2: Retrieving legal context...", sid)
        rag_results: list[RAGResult] = await retrieve_legal_context(user_input, intent)

        await SessionStore.update(
            sid,
            rag_results=[
                {"text": r.text[:500], "act_name": r.act_name, "section": r.section}
                for r in rag_results
            ],
        )

        # ── Step 3: Stream LLM Explanation ───────────────────────
        logger.info("[%s] Step 3: Streaming explanation...", sid)
        explanation_prompt = _build_explanation_prompt(user_input, intent, rag_results)
        llm = get_llm_client()

        full_explanation = []
        async for token in llm.stream(explanation_prompt):
            full_explanation.append(token)
            yield _sse({"type": "token", "text": token})

        await SessionStore.update(
            sid, explanation="".join(full_explanation)
        )

        # ── Step 4: Entity Extraction (runs silently) ────────────
        logger.info("[%s] Step 4: Extracting entities...", sid)
        entities = await extract_entities(user_input, intent, rag_results)

        await SessionStore.update(
            sid, extracted_entities=entities, status="awaiting_form"
        )

        # ── Step 5: Send form schema ─────────────────────────────
        form_event = build_form_schema(intent)
        yield _sse(form_event)

        # ── Done ─────────────────────────────────────────────────
        yield _sse({"type": "complete", "session_id": sid})
        logger.info("[%s] Pipeline complete.", sid)

    except Exception as exc:
        logger.exception("[%s] Pipeline error: %s", sid, exc)
        yield _sse({
            "type": "error",
            "message": f"An error occurred while processing your request: {str(exc)}",
            "recoverable": True,
        })
        await SessionStore.update(sid, status="error")


def _build_explanation_prompt(
    user_input: str,
    intent: IntentResult,
    rag_results: list[RAGResult],
) -> list[dict]:
    """Build the prompt for the streamed legal explanation."""
    rag_context = ""
    if rag_results:
        clauses = "\n".join(
            f"- {r.act_name}, {r.section}: {r.text[:400]}"
            for r in rag_results[:5]
        )
        rag_context = f"\n\nRelevant Legal Provisions (from official Indian law databases and government sites):\n{clauses}"

    system = f"""You are Sarathi, an AI-powered Indian legal empowerment assistant. You help ordinary citizens understand their legal rights in simple, clear language.

The user has a legal issue classified as: {intent.domain.value}
Summary: {intent.summary}{rag_context}

Instructions:
1. Explain the user's legal rights in plain, empathetic language that anyone can understand — even someone with no legal background.
2. Reference specific acts, sections, and rules where applicable. If the context includes government website sources, mention them so the user can verify.
3. Translate any complex legal jargon into everyday Hindi-English terms (e.g., "notice bhejo" for "send a notice", "aapka haq hai" for "it is your right").
4. Outline the practical steps they should take, in a numbered list.
5. Keep the explanation concise (3-5 paragraphs).
6. If the information comes from an official government source, say so (e.g., "According to the official India Code website...").
7. Do NOT provide specific legal advice — empower the user with knowledge of their rights and the correct legal procedures."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
