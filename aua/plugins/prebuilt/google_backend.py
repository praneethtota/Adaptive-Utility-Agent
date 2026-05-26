"""
core/plugins/google_backend.py — Google Gemini backend plugin.

Covers: gemini-1.5-pro, gemini-2.0-flash
Uses Google Generative Language REST API directly (no SDK dependency).
NOT OpenAI-compatible — different auth (API key as query param), different
request/response format.

Contributed from AUA-Veritas.
Source: core/plugins/google_backend.py
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

log = logging.getLogger(__name__)

GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleBackend:
    """
    Google Gemini backend. Uses the Google Generative Language REST API.

    Auth: API key passed as query parameter (not Authorization header).
    Endpoint pattern: /v1beta/models/{model}:generateContent?key={api_key}

    Args:
        model_id: e.g. "gemini-1.5-pro" or "gemini-2.0-flash"
        api_key:  Google AI Studio API key (AIza...)
    """

    def __init__(self, model_id: str = "gemini-1.5-pro", api_key: str = "") -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=60.0)

    # ── URL helpers ───────────────────────────────────────────────────────────

    def _url(self, method: str) -> str:
        """Build the full endpoint URL for a given method."""
        # Gemini model IDs use hyphens; the API accepts them directly
        return f"{GOOGLE_BASE_URL}/models/{self.model_id}:{method}?key={self._api_key}"

    # ── Request/response conversion ───────────────────────────────────────────

    @staticmethod
    def _to_google(request: dict) -> dict:
        """
        Convert OpenAI-compatible request dict to Google format.

        OpenAI: {"messages": [{"role": "user", "content": "..."}], "max_tokens": N}
        Google: {"contents": [{"role": "user", "parts": [{"text": "..."}]}],
                 "generationConfig": {"maxOutputTokens": N}}
        """
        messages = request.get("messages", [])
        contents: list[dict] = []
        system_text = None

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                # Google doesn't have a system role — prepend to first user message
                system_text = content
                continue
            # Google uses "model" not "assistant"
            g_role = "model" if role == "assistant" else "user"
            contents.append({"role": g_role, "parts": [{"text": content}]})

        # Prepend system text to first user turn if present
        if system_text and contents:
            first_user = contents[0]
            first_user["parts"] = [{"text": system_text + "\n\n" + first_user["parts"][0]["text"]}]

        payload: dict = {"contents": contents}
        gen_config: dict = {}

        if "max_tokens" in request:
            gen_config["maxOutputTokens"] = request["max_tokens"]
        if "temperature" in request:
            gen_config["temperature"] = request["temperature"]

        if gen_config:
            payload["generationConfig"] = gen_config

        return payload

    @staticmethod
    def _from_google(response: dict) -> dict:
        """
        Convert Google response to OpenAI-compatible format.

        Google:
          {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        OpenAI:
          {"choices": [{"message": {"role": "assistant", "content": "..."}}]}
        """
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # Handle safety-blocked or empty responses
            finish_reason = "unknown"
            try:
                finish_reason = response["candidates"][0].get("finishReason", "unknown")
            except (KeyError, IndexError):
                pass
            if finish_reason == "SAFETY":
                text = "[Response blocked by Google safety filters]"
            else:
                text = ""
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "model": response.get("modelVersion", ""),
        }

    # ── Public interface (OpenAI-compatible) ──────────────────────────────────

    async def complete(self, request: dict) -> dict:
        """
        Non-streaming completion.

        Args:
            request: OpenAI-compatible dict (model, messages, temperature, max_tokens).
                     The 'model' key is ignored — self.model_id is always used.

        Returns:
            OpenAI-compatible response dict. Caller reads:
                result["choices"][0]["message"]["content"]
        """
        payload = self._to_google(request)
        try:
            r = await self._client.post(
                self._url("generateContent"),
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            return self._from_google(r.json())
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:300]
            log.error("Google API error %d: %s", status, body)
            if status == 400:
                raise ValueError(f"Google API bad request: {body}") from e
            if status == 403:
                raise PermissionError("Google API key invalid or quota exceeded") from e
            raise

    async def stream(self, request: dict) -> AsyncIterator[str]:
        """
        Streaming completion via Google Server-Sent Events.

        Yields token strings as they arrive. Google wraps SSE in a JSON array
        — each frame is a complete generateContent response chunk.

        Args:
            request: Same shape as complete().

        Yields:
            str — individual text chunks.
        """
        payload = self._to_google(request)
        async with self._client.stream(
            "POST",
            self._url("streamGenerateContent") + "&alt=sse",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]  # strip "data: "
                if data in ("[DONE]", ""):
                    continue
                try:
                    chunk = json.loads(data)
                    text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def health(self) -> dict:
        """
        Health check — sends a minimal request to verify the API key works.

        Returns:
            {"status": "ok", "model": "...", "latency_ms": N}
            {"status": "error", "error": "...", "latency_ms": N}
        """
        t0 = time.time()
        try:
            payload = {
                "contents": [{"role": "user", "parts": [{"text": "Say ok"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            }
            r = await self._client.post(
                self._url("generateContent"),
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            return {
                "status": "ok",
                "model": self.model_id,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 400:
                error = "Invalid request — check model ID"
            elif status == 403:
                error = "Invalid API key or quota exceeded"
            elif status == 404:
                error = f"Model {self.model_id!r} not found"
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

    async def aclose(self) -> None:
        await self._client.aclose()
