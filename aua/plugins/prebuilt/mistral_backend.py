"""
core/plugins/mistral_backend.py — Mistral AI backend plugin.

Covers: mistral-large-latest, mistral-small-latest, open-mistral-nemo
OpenAI-compatible API at api.mistral.ai/v1

Good choice for: European data residency requirements (Mistral is French),
users who prefer open-weight models, coding tasks (Codestral).

Contributed from AUA-Veritas.
Source: core/plugins/mistral_backend.py
"""

from __future__ import annotations

import logging
import time

import httpx

from aua.plugins.prebuilt.openai_backend import OpenAIBackend

log = logging.getLogger(__name__)

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "mistral-large-latest"


class MistralBackend(OpenAIBackend):
    """
    Mistral AI backend. Fully OpenAI-compatible API.

    Inherits complete(), stream(), and aclose() from OpenAIBackend.
    Overrides health() to use /models endpoint which Mistral supports
    and returns a useful list of available models.

    Args:
        model_id: e.g. "mistral-large-latest", "mistral-small-latest",
                  "open-mistral-nemo", "codestral-latest"
        api_key:  Mistral API key
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, api_key: str = "") -> None:
        super().__init__(model_id=model_id, api_key=api_key, base_url=MISTRAL_BASE_URL)

    async def health(self) -> dict:
        """
        Health check via GET /models — Mistral supports this endpoint fully.
        Verifies the API key and that the requested model is available.
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
                    "error": f"Model {self.model_id!r} not in your Mistral account. "
                    f"Available: {', '.join(available[:5])}",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                }
            return {
                "status": "ok",
                "model": self.model_id,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 401:
                error = "Invalid Mistral API key"
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
