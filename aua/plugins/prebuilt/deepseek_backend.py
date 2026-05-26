"""
core/plugins/deepseek_backend.py — DeepSeek backend plugin.

Covers: deepseek-chat (DeepSeek-V3), deepseek-reasoner (DeepSeek-R1)
OpenAI-compatible API at api.deepseek.com/v1

DeepSeek is notable for:
  - Very low cost (~10x cheaper than GPT-4o per token)
  - Strong coding performance (competitive with GPT-4o on code benchmarks)
  - deepseek-reasoner exposes its chain-of-thought reasoning tokens

Contributed from AUA-Veritas.
Source: core/plugins/deepseek_backend.py
"""

from __future__ import annotations

import logging
import time

import httpx

from aua.plugins.prebuilt.openai_backend import OpenAIBackend

log = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekBackend(OpenAIBackend):
    """
    DeepSeek backend (DeepSeek-V3 and DeepSeek-R1). Fully OpenAI-compatible.

    Inherits complete(), stream(), and aclose() from OpenAIBackend.
    Overrides health() with a minimal chat completion check.

    Special handling for deepseek-reasoner:
      The reasoner model returns an additional 'reasoning_content' field
      alongside the regular 'content'. complete() strips this out to stay
      OpenAI-compatible, but logs the reasoning length for debugging.

    Args:
        model_id: "deepseek-chat" (V3) or "deepseek-reasoner" (R1)
        api_key:  DeepSeek API key (sk-...)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, api_key: str = "") -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    async def complete(self, request: dict) -> dict:
        """
        Non-streaming completion. Extends OpenAIBackend.complete() to handle
        deepseek-reasoner's additional reasoning_content field.
        """
        result = await super().complete(request)

        # deepseek-reasoner includes reasoning_content in the message
        # Log it for debugging but don't expose it in the standard response
        try:
            msg = result["choices"][0].get("message", {})
            if "reasoning_content" in msg:
                rc = msg.pop("reasoning_content", "")
                log.debug(
                    "DeepSeek reasoning tokens: %d chars (stripped from response)",
                    len(rc or ""),
                )
        except (KeyError, IndexError):
            pass

        return result

    async def health(self) -> dict:
        """
        Health check via a minimal chat completion (1 token).
        DeepSeek's /models endpoint sometimes has auth quirks —
        using a cheap chat call is more reliable.
        """
        t0 = time.time()
        try:
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": "Say ok"}],
                "max_tokens": 5,
            }
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return {
                "status": "ok",
                "model": self.model_id,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "note": "DeepSeek is ~10x cheaper than GPT-4o per token.",
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error = "Invalid DeepSeek API key"
            elif status == 402:
                error = "Insufficient DeepSeek credits — top up at platform.deepseek.com"
            elif status == 429:
                error = "Rate limit exceeded"
            elif status == 503:
                error = "DeepSeek servers overloaded — try again shortly"
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
