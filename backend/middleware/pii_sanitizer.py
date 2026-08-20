"""
PII Sanitizer Middleware

Intercepts request bodies and redacts personally identifiable information
before it reaches the application's logging layer. The *original* body is
passed through untouched to route handlers (the LLM needs real data), but
a sanitized copy is stored on `request.state.sanitized_body` for audit trails.

Detected PII patterns (India-specific):
  • Aadhaar numbers  (12 digits, with optional spaces/hyphens)
  • PAN cards        (ABCDE1234F)
  • Mobile numbers   (+91 / 0 prefixed 10-digit)
  • Email addresses
"""

import re
import logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from config import get_settings

logger = logging.getLogger("sarathi.pii")

# ── Compiled Patterns (compiled once, reused per-request) ────────

_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "AADHAAR",
        re.compile(
            r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
    ),
    (
        "PAN",
        re.compile(
            r"\b[A-Z]{5}\d{4}[A-Z]\b"
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}\b"
        ),
    ),
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
        ),
    ),
]


def redact(text: str) -> str:
    """Replace all PII matches with [REDACTED_<TYPE>] tags."""
    for label, pattern in _PATTERNS:
        text = pattern.sub(f"[REDACTED_{label}]", text)
    return text


class PIISanitizerMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that:
    1. Reads the raw request body.
    2. Produces a sanitized copy for logging.
    3. Passes the original body through to the route handler.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()

        if not settings.PII_REDACTION_ENABLED:
            return await call_next(request)

        # Only process JSON bodies on mutation endpoints
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8", errors="replace")

            sanitized = redact(body_text)
            request.state.sanitized_body = sanitized

            if sanitized != body_text:
                logger.info(
                    "PII redacted from request to %s", request.url.path
                )
                logger.debug("Sanitized body: %s", sanitized)
        else:
            request.state.sanitized_body = None

        return await call_next(request)
