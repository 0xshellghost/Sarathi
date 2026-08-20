"""
LLM Client — Model-Agnostic Interface

Communicates with any OpenAI-compatible API (Ollama, vLLM, LocalAI, etc.)
via async HTTP. Your custom model runs behind the endpoint — this client
doesn't care what model is serving, only that it speaks the chat completions
protocol.

Supports:
  • Blocking calls (for intent routing, entity extraction)
  • Streaming calls (for SSE explanation tokens)
  • JSON-mode forcing (Ollama's format: "json")
  • Exponential backoff on transient failures
"""

import json
import asyncio
import logging
from typing import AsyncGenerator

import httpx

from config import get_settings

logger = logging.getLogger("sarathi.llm")


class LLMClient:
    """Async client for the self-hosted LLM."""

    def __init__(self):
        settings = get_settings()
        self._base_url = settings.OLLAMA_BASE_URL
        self._model = settings.OLLAMA_MODEL
        self._temperature = settings.LLM_TEMPERATURE
        self._max_tokens = settings.LLM_MAX_TOKENS
        self._max_retries = settings.LLM_MAX_RETRIES
        self._timeout = settings.OLLAMA_TIMEOUT

    def _build_payload(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        json_mode: bool = False,
    ) -> dict:
        """Construct the Ollama /api/chat request payload."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    async def generate(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
    ) -> str:
        """
        Blocking LLM call. Returns the full response text.
        Retries with exponential backoff on transient errors.
        """
        payload = self._build_payload(messages, stream=False, json_mode=json_mode)
        url = f"{self._base_url}/api/chat"

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    return content.strip()

            except (httpx.HTTPError, httpx.TimeoutException, KeyError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                if attempt == self._max_retries:
                    raise RuntimeError(
                        f"LLM unreachable after {self._max_retries} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(wait)

        # Unreachable, but keeps type-checker happy
        raise RuntimeError("LLM call failed unexpectedly.")

    async def stream(
        self,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Streaming LLM call. Yields text tokens as they arrive.
        Used for SSE explanation streaming to the frontend.
        """
        payload = self._build_payload(messages, stream=True)
        url = f"{self._base_url}/api/chat"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue


# ── Singleton ────────────────────────────────────────────────────
_instance: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _instance
    if _instance is None:
        _instance = LLMClient()
    return _instance
