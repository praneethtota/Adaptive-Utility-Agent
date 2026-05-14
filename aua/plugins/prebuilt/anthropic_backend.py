"""
core/plugins/anthropic_backend.py — Anthropic backend plugin.

Covers: claude-sonnet-4-6, claude-haiku-4-5-20251001
Uses the Anthropic messages API (not OpenAI-compatible).

To contribute back to AUA: copy this file to
  aua/plugins/prebuilt/anthropic_backend.py
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicBackend:
    """
    ModelBackendPlugin for Anthropic models (Claude Sonnet, Claude Haiku).

    The Anthropic API is NOT OpenAI-compatible — it uses a different
    request/response format. This plugin adapts it to the OpenAI-compatible
    interface that Veritas uses internally.

    Args:
        model_id: Anthropic model string e.g. "claude-sonnet-4-6".
        api_key:  Anthropic API key (sk-ant-...).
    """

    def __init__(self, model_id: str = "claude-sonnet-4-6", api_key: str = "") -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def _to_anthropic_request(self, request: dict) -> dict:
        """
        Convert OpenAI-compatible request dict to Anthropic messages API format.

        OpenAI format:
            {"model": "...", "messages": [{"role": "user", "content": "..."}], "max_tokens": 2048}

        Anthropic format:
            {"model": "...", "messages": [...], "max_tokens": 2048, "system": "..."}
        """
        messages = request.get("messages", [])
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append({"role": m["role"], "content": m["content"]})

        payload = {
            "model": self.model_id,
            "messages": filtered,
            "max_tokens": request.get("max_tokens", 2048),
        }
        if system:
            payload["system"] = system
        if "temperature" in request:
            payload["temperature"] = request["temperature"]
        return payload

    def _from_anthropic_response(self, response: dict) -> dict:
        """
        Convert Anthropic response to OpenAI-compatible format.

        Anthropic: {"content": [{"type": "text", "text": "..."}], ...}
        OpenAI:    {"choices": [{"message": {"content": "..."}}], ...}
        """
        text = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": response.get("model", self.model_id),
            "usage": response.get("usage", {}),
        }

    async def complete(self, request: dict) -> dict:
        """
        Non-streaming completion.

        Args:
            request: OpenAI-compatible format (converted internally).

        Returns:
            OpenAI-compatible response dict.
        """
        payload = self._to_anthropic_request(request)
        try:
            r = await self._client.post("/v1/messages", json=payload)
            r.raise_for_status()
            return self._from_anthropic_response(r.json())
        except httpx.HTTPStatusError as e:
            log.error("Anthropic API error %d: %s", e.response.status_code, e.response.text[:200])
            raise
        except httpx.RequestError as e:
            log.error("Anthropic request error: %s", e)
            raise

    async def stream(self, request: dict) -> AsyncIterator[str]:
        """
        Streaming completion. Yields token strings as they arrive.
        """
        import json

        payload = {**self._to_anthropic_request(request), "stream": True}
        async with self._client.stream("POST", "/v1/messages", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                try:
                    event = json.loads(data)
                    if event.get("type") == "content_block_delta":
                        token = event.get("delta", {}).get("text", "")
                        if token:
                            yield token
                except (json.JSONDecodeError, KeyError):
                    continue

    async def health(self) -> dict:
        """
        Health check — sends a minimal message to verify the API key works.
        """
        t0 = time.time()
        try:
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": "Say 'ok'"}],
                "max_tokens": 5,
            }
            r = await self._client.post("/v1/messages", json=payload)
            r.raise_for_status()
            return {
                "status": "ok",
                "model": self.model_id,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            if e.response.status_code == 401:
                error_msg = "Invalid API key"
            elif e.response.status_code == 403:
                error_msg = "API key lacks access to this model"
            return {
                "status": "error",
                "error": error_msg,
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
