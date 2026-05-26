"""
core/plugins/xai_backend.py — xAI Grok backend plugin.

Covers: grok-2, grok-2-mini
OpenAI-compatible API endpoint at api.x.ai/v1

Contributed from AUA-Veritas.
Source: core/plugins/xai_backend.py
"""

from __future__ import annotations

import logging
import time

import httpx

from aua.plugins.prebuilt.openai_backend import OpenAIBackend

log = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-2"


class XAIBackend(OpenAIBackend):
    """
    xAI Grok-2 backend. Fully OpenAI-compatible API.

    Inherits complete(), stream(), and aclose() from OpenAIBackend.
    Overrides health() with a lightweight chat-based check since
    xAI's /models endpoint requires additional permissions.

    Args:
        model_id: e.g. "grok-2" or "grok-2-mini"
        api_key:  xAI API key (xai-...)
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, api_key: str = "") -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=XAI_BASE_URL)

    async def health(self) -> dict:
        """
        Health check via a minimal chat completion (1 token).
        xAI's /models endpoint requires extra permissions — use a cheap
        chat call instead, same as the Anthropic approach.
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
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error = "Invalid API key"
            elif status == 402:
                error = "Insufficient credits — check xAI billing"
            elif status == 404:
                error = f"Model {self.model_id!r} not found"
            elif status == 429:
                error = "Rate limit exceeded"
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
