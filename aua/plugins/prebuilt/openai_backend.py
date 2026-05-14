"""
core/plugins/openai_backend.py — OpenAI backend plugin.

Covers: gpt-4o, gpt-4o-mini
Compatible models: any OpenAI chat completion endpoint.

To contribute back to AUA: copy this file to
  aua/plugins/prebuilt/openai_backend.py
and register in aua/plugins/registry.py.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIBackend:
    """
    ModelBackendPlugin for OpenAI models (GPT-4o, GPT-4o mini).

    Args:
        model_id: The OpenAI model string e.g. "gpt-4o" or "gpt-4o-mini".
        api_key:  OpenAI API key (sk-...).
        base_url: Override base URL for Azure or compatible endpoints.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str = "",
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def complete(self, request: dict) -> dict:
        """
        Non-streaming chat completion.

        Args:
            request: OpenAI-compatible dict with at minimum:
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "..."}],
                    "temperature": 0.2,
                    "max_tokens": 2048
                }

        Returns:
            OpenAI-compatible response dict. Caller reads:
                result["choices"][0]["message"]["content"]
        """
        # Always use the backend's model_id, not whatever is in the request
        payload = {**request, "model": self.model_id, "stream": False}
        try:
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            log.error("OpenAI API error %d: %s", e.response.status_code, e.response.text[:200])
            raise
        except httpx.RequestError as e:
            log.error("OpenAI request error: %s", e)
            raise

    async def stream(self, request: dict) -> AsyncIterator[str]:
        """
        Streaming chat completion. Yields token strings as they arrive.

        Args:
            request: Same shape as complete().

        Yields:
            str — individual token chunks.
        """
        import json

        payload = {**request, "model": self.model_id, "stream": True}
        async with self._client.stream("POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    token = chunk["choices"][0]["delta"].get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def health(self) -> dict:
        """
        Health check — verifies API key is valid and model is accessible.

        Returns:
            {"status": "ok", "model": "gpt-4o", "latency_ms": 123.4}
            {"status": "error", "error": "...", "latency_ms": 0.0}
        """
        t0 = time.time()
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
            models = [m["id"] for m in r.json().get("data", [])]
            if self.model_id not in models:
                return {
                    "status": "error",
                    "error": f"Model {self.model_id!r} not found in account",
                    "latency_ms": round((time.time() - t0) * 1000, 1),
                }
            return {
                "status": "ok",
                "model": self.model_id,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }

    async def aclose(self) -> None:
        await self._client.aclose()
