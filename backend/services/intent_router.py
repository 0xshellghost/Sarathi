"""
Intent Router — Step 1 of the Deterministic AI Pipeline

Classifies the user's plain-text problem into a legal domain using a single
LLM call with JSON-mode output. No chains, no agents — one prompt, one call.
"""

import json
import logging

from models.domain import LegalDomain, IntentResult
from services.llm_client import get_llm_client

logger = logging.getLogger("sarathi.pipeline.intent")

_SYSTEM_PROMPT = """You are a legal domain classifier for an Indian civic empowerment platform called Sarathi.

Given a user's description of their legal problem, classify it into exactly ONE of these domains:
- rent_deposit_dispute: Issues with landlords, security deposits, rental agreements, tenant rights
- consumer_complaint: Defective products, deficient services, unfair trade practices
- employment_dispute: Wrongful termination, unpaid wages, workplace harassment, PF/gratuity issues
- property_dispute: Land/property ownership conflicts, encroachment, title disputes
- cheque_bounce: Dishonoured cheques under Section 138 of the Negotiable Instruments Act
- general_legal_query: Any legal question that doesn't fit the above categories

Respond in JSON with exactly these fields:
{
  "domain": "<one of the domain values above>",
  "confidence": <float between 0.0 and 1.0>,
  "summary": "<one-line summary of the user's problem>"
}"""


async def classify_intent(user_input: str) -> IntentResult:
    """
    Classify user input into a legal domain.

    Returns an IntentResult with the domain, confidence, and summary.
    Always returns a valid result — falls back to general_legal_query
    if the LLM output is unparseable.
    """
    llm = get_llm_client()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    try:
        raw = await llm.generate(messages, json_mode=True)
        data = json.loads(raw)

        domain_str = data.get("domain", "general_legal_query")

        # Validate against enum
        try:
            domain = LegalDomain(domain_str)
        except ValueError:
            logger.warning("LLM returned unknown domain '%s', falling back.", domain_str)
            domain = LegalDomain.GENERAL_LEGAL_QUERY

        result = IntentResult(
            domain=domain,
            confidence=min(max(float(data.get("confidence", 0.5)), 0.0), 1.0),
            summary=data.get("summary", user_input[:100]),
        )
        logger.info("Intent classified: %s (%.0f%%)", result.domain.value, result.confidence * 100)
        return result

    except (json.JSONDecodeError, RuntimeError) as exc:
        logger.error("Intent classification failed: %s. Defaulting to general.", exc)
        return IntentResult(
            domain=LegalDomain.GENERAL_LEGAL_QUERY,
            confidence=0.0,
            summary=user_input[:100],
        )
