"""
Entity Extractor — Step 3 of the Deterministic AI Pipeline

Extracts structured fields from the user's input using an LLM call with
JSON-mode output, strictly enforcing the domain's field schema.

Key feature: SILENT RETRIES — if the LLM returns malformed JSON, this
module catches the error and retries with a progressively stricter prompt.
The frontend never receives a broken payload.
"""

import json
import logging

from models.domain import LegalDomain, IntentResult, RAGResult, DOMAIN_FIELD_SCHEMAS
from services.llm_client import get_llm_client
from config import get_settings

logger = logging.getLogger("sarathi.pipeline.entity")


def _build_schema_description(domain: LegalDomain) -> str:
    """Build a human-readable field description from the domain schema."""
    fields = DOMAIN_FIELD_SCHEMAS.get(domain, [])
    lines = []
    for f in fields:
        req = "REQUIRED" if f.required else "optional"
        lines.append(f'  - "{f.key}" ({f.field_type}, {req}): {f.label}')
    return "\n".join(lines)


def _build_extraction_prompt(
    user_input: str,
    intent: IntentResult,
    rag_results: list[RAGResult],
    attempt: int = 1,
) -> list[dict]:
    """
    Build the extraction prompt. On retries, the prompt gets progressively
    stricter to force valid JSON output.
    """
    schema_desc = _build_schema_description(intent.domain)

    # Combine RAG context
    rag_context = ""
    if rag_results:
        clauses = "\n".join(
            f"- {r.act_name}, {r.section}: {r.text[:300]}"
            for r in rag_results[:5]
        )
        rag_context = f"\n\nRelevant Legal Provisions:\n{clauses}"

    strictness = ""
    if attempt > 1:
        strictness = (
            "\n\nCRITICAL: Your previous response was not valid JSON. "
            "You MUST respond with ONLY a JSON object. No markdown, no explanation, "
            "no code fences. Just the raw JSON object starting with { and ending with }."
        )

    system = f"""You are a legal entity extractor for the Indian legal domain: {intent.domain.value}.

From the user's description, extract the following fields into a JSON object:
{schema_desc}

Rules:
1. Respond with ONLY a valid JSON object — no markdown, no explanation.
2. Use null for fields you cannot determine from the user's input.
3. For required fields where the value is not mentioned, use a reasonable placeholder like "Not Specified".
4. Amounts should be numbers (no currency symbols).
5. Dates should be in ISO format (YYYY-MM-DD) or null.{rag_context}{strictness}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


async def extract_entities(
    user_input: str,
    intent: IntentResult,
    rag_results: list[RAGResult],
) -> dict:
    """
    Extract structured entities from user input.

    Implements silent LLM retries: on malformed JSON, retries up to
    MAX_RETRIES times with stricter prompts. The caller (and therefore
    the frontend) never sees a parsing error.
    """
    settings = get_settings()
    llm = get_llm_client()
    max_attempts = settings.LLM_MAX_RETRIES

    for attempt in range(1, max_attempts + 1):
        messages = _build_extraction_prompt(user_input, intent, rag_results, attempt)

        try:
            raw = await llm.generate(messages, json_mode=True)

            # Strip markdown code fences if the model wraps its output
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            entities = json.loads(cleaned)

            if not isinstance(entities, dict):
                raise ValueError(f"Expected dict, got {type(entities).__name__}")

            logger.info(
                "Entities extracted on attempt %d: %d fields",
                attempt,
                len(entities),
            )
            return entities

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Entity extraction attempt %d/%d failed: %s. %s",
                attempt,
                max_attempts,
                exc,
                "Retrying silently..." if attempt < max_attempts else "Returning defaults.",
            )

    # All retries exhausted — return a safe default with nulls
    logger.error("Entity extraction failed after %d attempts. Returning defaults.", max_attempts)
    fields = DOMAIN_FIELD_SCHEMAS.get(intent.domain, [])
    return {f.key: None for f in fields}
