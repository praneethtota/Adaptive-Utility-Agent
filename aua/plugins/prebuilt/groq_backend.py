"""
core/plugins/groq_backend.py — Groq backend plugin (Llama 3.3 70B).

Covers: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
OpenAI-compatible API at api.groq.com/openai/v1

Groq is notable for extremely fast inference (tokens-per-second world record)
and a generous free tier — good first model for users without a credit card.
Runs Meta's open-weight Llama models on custom hardware.

Contributed from AUA-Veritas.
Source: core/plugins/groq_backend.py
"""
from __future__ import annotations

import logging
import time

import httpx

from aua.plugins.prebuilt.openai_backend import OpenAIBackend

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Groq context window limits per model
GROQ_CONTEXT_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile":   128_000,
    "llama-3.1-70b-versatile":   128_000,
    "llama-3.1-8b-instant":      128_000,
    "mixtral-8x7b-32768":         32_768,
    "gemma2-9b-it":                8_192,
}


class GroqBackend(OpenAIBackend):
    """
    Groq backend (Llama 3.3 70B and others). Fully OpenAI-compatible API.

    Inherits complete(), stream(), and aclose() from OpenAIBackend.
    Overrides health() to use GET /models (Groq supports this) and to
    expose the model's context window limit for context overflow management.

    Note on speed: Groq typically returns in 0.3–0.8s for 70B models —
    significantly faster than most frontier APIs. This makes it an
    excellent candidate for peer-review judge calls in Balanced mode.

    Args:
        model_id: e.g. "llama-3.3-70b-versatile" or "llama-3.1-8b-instant"
        api_key:  Groq API key (gsk_...)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, api_key: str = "") -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=GROQ_BASE_URL)

    @property
    def context_window(self) -> int:
        """Return context window size for the current model."""
        return GROQ_CONTEXT_LIMITS.get(self.model_id, 8_192)

    async def health(self) -> dict:
        """
        Health check via GET /models.
        Groq supports the OpenAI /models endpoint fully.
        Also returns context_window for the current model.
        """
        t0 = time.time()
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
            data = r.json()
            available = [m["id"] for m in data.get("data", [])]
            if self.model_id not in available:
                return {
                    "status": "error",
                    "error": (
                        f"Model {self.model_id!r} not available on Groq. "
                        f"Try: {', '.join(available[:4])}"
                    ),
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                }
            return {
                "status": "ok",
                "model": self.model_id,
                "context_window": self.context_window,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "note": "Groq has a generous free tier — no credit card needed to start.",
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error = "Invalid Groq API key (gsk_...)"
            elif status == 429:
                error = "Rate limit exceeded — Groq free tier has per-minute limits"
            else:
                error = f"HTTP {status}"
            return {
                "status": "error",
                "error": error,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
