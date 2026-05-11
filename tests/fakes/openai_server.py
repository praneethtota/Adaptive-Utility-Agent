"""
tests/fakes/openai_server.py — Fake OpenAI-compatible specialist server.

Implements the minimal OpenAI-compatible API subset that AUA uses:
    GET  /v1/models                 → model list (health check endpoint)
    POST /v1/chat/completions       → buffered and streaming responses

Also implements the Ollama-compatible subset:
    GET  /api/tags                  → model list (Ollama health check)
    POST /api/chat                  → buffered and streaming (NDJSON)

Usage in tests:
    from tests.fakes.openai_server import make_fake_specialist, start_fake_server

    app = make_fake_specialist(model_name="swe", response="hello world")
    port, stop = start_fake_server(app)
    # ... run tests ...
    stop()
"""

import json
import socket
import threading
import time
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


def _find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_fake_specialist(
    model_name: str = "fake-model",
    response: str = "This is a fake specialist response.",
    stream_tokens: list[str] | None = None,
) -> FastAPI:
    """
    Build a FastAPI app that mimics an OpenAI-compatible specialist endpoint.

    Args:
        model_name:    The model name returned in /v1/models.
        response:      The full response text for non-streaming requests.
        stream_tokens: Optional list of tokens to emit when streaming.
                       Defaults to splitting `response` into words.
    """
    if stream_tokens is None:
        stream_tokens = response.split()

    app = FastAPI(title=f"Fake specialist: {model_name}")

    # ── OpenAI-compatible endpoints ───────────────────────────────────────────

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": 0, "owned_by": "aua-test"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        is_stream = body.get("stream", False)

        if is_stream:
            # Server-Sent Events format
            async def generate():
                for i, token in enumerate(stream_tokens):
                    text = token if i == 0 else " " + token
                    chunk = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                # Final chunk
                final = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        # Buffered response
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    # ── Ollama-compatible endpoints ───────────────────────────────────────────

    @app.get("/api/tags")
    def ollama_tags():
        return {"models": [{"name": model_name, "size": 0}]}

    @app.post("/api/chat")
    async def ollama_chat(body: dict):
        is_stream = body.get("stream", True)

        if is_stream:

            async def generate():
                for i, token in enumerate(stream_tokens):
                    text = token if i == 0 else " " + token
                    chunk = {
                        "model": model_name,
                        "message": {"role": "assistant", "content": text},
                        "done": False,
                    }
                    yield json.dumps(chunk) + "\n"
                # Final chunk
                final = {
                    "model": model_name,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                }
                yield json.dumps(final) + "\n"

            return StreamingResponse(generate(), media_type="application/x-ndjson")

        return {
            "model": model_name,
            "message": {"role": "assistant", "content": response},
            "done": True,
        }

    return app


def start_fake_server(app: FastAPI, port: int | None = None) -> tuple[int, Callable]:
    """
    Start `app` in a background daemon thread.

    Returns:
        (port, stop_fn) — port the server is listening on, and a function to stop it.

    The server runs until `stop_fn()` is called or the process exits.
    """
    port = port or _find_free_port()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to become ready (up to 5s)
    import httpx

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError(f"Fake specialist did not start on port {port} within 5s")

    def stop():
        server.should_exit = True
        thread.join(timeout=3)

    return port, stop
