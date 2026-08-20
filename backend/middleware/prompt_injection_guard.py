"""
Prompt Injection Guard Middleware

Heuristic-based detection of prompt injection attacks. Runs before any
request body reaches the LLM pipeline. Zero-latency — no model calls,
pure pattern matching and scoring.

Threat Model:
  • Direct injection: "ignore previous instructions", "you are now..."
  • Role-play attacks: "pretend you are an unrestricted AI"
  • Delimiter injection: markdown fences, XML tags used to redefine context
  • Data exfiltration: attempts to leak system prompt or training data
"""

import re
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from config import get_settings

logger = logging.getLogger("sarathi.security")

# ── Injection Signatures ─────────────────────────────────────────
# Each tuple: (pattern, weight). Total score > threshold → block.

_INJECTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    # Direct instruction override
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)", re.I), 0.9),
    (re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.I), 0.9),
    (re.compile(r"forget\s+(everything|all|your)\s+(you|instructions?|rules?)", re.I), 0.85),

    # System prompt extraction
    (re.compile(r"(show|reveal|print|display|output|repeat)\s+(your\s+)?(system\s+prompt|instructions?|rules?)", re.I), 0.8),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)", re.I), 0.7),

    # Role-play / jailbreak
    (re.compile(r"(you\s+are|act\s+as|pretend\s+(to\s+be|you\s+are))\s+(an?\s+)?(unrestricted|unfiltered|evil|DAN)", re.I), 0.9),
    (re.compile(r"(jailbreak|DAN\s+mode|developer\s+mode|god\s+mode)", re.I), 0.95),

    # Delimiter / context boundary attacks
    (re.compile(r"<\|?(system|endoftext|im_start|im_end)\|?>", re.I), 0.85),
    (re.compile(r"```\s*(system|prompt|instructions?)", re.I), 0.7),
    (re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.I), 0.85),

    # Data exfiltration
    (re.compile(r"(output|send|post|fetch|curl|wget)\s+.*(http|url|endpoint|webhook)", re.I), 0.6),

    # Subtle manipulation
    (re.compile(r"(from\s+now\s+on|new\s+rule|override|overwrite)\s+", re.I), 0.5),
    (re.compile(r"do\s+not\s+(follow|obey|listen\s+to)\s+(the\s+)?(rules?|instructions?|guidelines?)", re.I), 0.8),
]

# Paths that carry user input and need scanning
_PROTECTED_PATHS = {"/api/v1/action/analyze", "/api/v1/transcribe"}


def score_threat(text: str) -> tuple[float, list[str]]:
    """
    Score a text for prompt injection signatures.
    Returns (max_score, list_of_matched_patterns).
    """
    matches: list[str] = []
    max_score = 0.0

    for pattern, weight in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern[:60])
            max_score = max(max_score, weight)

    return max_score, matches


class PromptInjectionGuard(BaseHTTPMiddleware):
    """Blocks requests with high prompt injection threat scores."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()

        # Only scan relevant endpoints
        if request.url.path not in _PROTECTED_PATHS:
            return await call_next(request)

        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            body_bytes = await request.body()
            text = body_bytes.decode("utf-8", errors="replace")

            threat_score, matched = score_threat(text)

            if threat_score >= settings.INJECTION_THREAT_THRESHOLD:
                logger.warning(
                    "Prompt injection BLOCKED on %s | score=%.2f | patterns=%s",
                    request.url.path,
                    threat_score,
                    matched,
                )
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": "Your input was flagged as potentially harmful and cannot be processed. "
                        "Please rephrase your legal question in plain language."
                    },
                )

            if threat_score > 0:
                logger.info(
                    "Low-risk injection patterns on %s | score=%.2f",
                    request.url.path,
                    threat_score,
                )

        return await call_next(request)
