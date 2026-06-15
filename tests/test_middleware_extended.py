"""
tests/test_middleware_extended.py — Tests for #52 extended middleware pipeline.

Covers:
  MiddlewarePipeline.on_chunk() — SSE token interception
    calls on_chunk on every token when middleware registered
    passes metadata dict with session_id, domain, routing_mode, chunk_index
    suppressed chunks (return "") are not yielded
    skips on_chunk when no middleware registered
    falls back if on_chunk raises

  MiddlewarePipeline.before_batch() / after_batch()
    before_batch called with job metadata before dispatch
    after_batch called with job and results after dispatch
    before_batch can modify the job dict
    after_batch can modify the results list
    skipped when no middleware registered

  MiddlewarePipeline.on_error()
    called when query pipeline raises
    first middleware to return a dict wins (reverse order)
    returning None lets exception propagate
    on_error itself raising is caught gracefully

  AUAMiddleware Protocol
    on_chunk / before_batch / after_batch / on_error are NOT required by Protocol
    existing middleware (before_query + after_response only) still validates

  BatchWorker middleware injection
    BatchWorker receives middleware kwarg
    before_batch fires before items are dispatched
    after_batch fires after all items complete
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from aua.middleware import MiddlewarePipeline

# ── Fixtures ──────────────────────────────────────────────────────────────────


class MinimalMiddleware:
    """Only implements the two required Protocol methods."""

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response


class ChunkRedactor:
    """Implements on_chunk — replaces 'SECRET' with '[REDACTED]'."""

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response

    def on_chunk(self, chunk: str, metadata: dict) -> str:
        return chunk.replace("SECRET", "[REDACTED]")


class ChunkSuppressor:
    """Suppresses any chunk containing 'SKIP'."""

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response

    async def on_chunk(self, chunk: str, metadata: dict) -> str:
        if "SKIP" in chunk:
            return ""  # suppress
        return chunk


class ChunkCrasher:
    """on_chunk always raises — should fall through gracefully."""

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response

    def on_chunk(self, chunk: str, metadata: dict) -> str:
        raise RuntimeError("chunk processing failed")


class BatchLogger:
    """Logs before_batch and after_batch calls."""

    def __init__(self):
        self.before_calls = []
        self.after_calls = []

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response

    async def before_batch(self, job: dict) -> dict:
        self.before_calls.append(job.copy())
        return {**job, "_tagged": True}

    async def after_batch(self, job: dict, results: list) -> list:
        self.after_calls.append((job.copy(), [r.copy() for r in results]))
        return [{**r, "_post": True} for r in results]


class ErrorRecoverer:
    """on_error returns a fallback response."""

    def __init__(self, recovery: dict | None = None):
        self.recovery = recovery
        self.called_with = []

    async def before_query(self, request: dict) -> dict:
        return request

    async def after_response(self, response: dict) -> dict:
        return response

    async def on_error(self, exc: Exception, request: dict) -> dict | None:
        self.called_with.append((exc, request))
        return self.recovery


# ── on_chunk tests ────────────────────────────────────────────────────────────


class TestOnChunk:
    def test_on_chunk_called_for_each_token(self) -> None:
        pipeline = MiddlewarePipeline()
        calls = []

        class Recorder:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            def on_chunk(self, chunk, meta):
                calls.append(chunk)
                return chunk

        pipeline.add(Recorder())

        tokens = ["Hello", " ", "world", "!"]
        results = [
            asyncio.run(pipeline.on_chunk(t, {"chunk_index": i})) for i, t in enumerate(tokens)
        ]

        assert calls == tokens
        assert results == tokens

    def test_on_chunk_receives_metadata_keys(self) -> None:
        pipeline = MiddlewarePipeline()
        received_meta = []

        class MetaCapture:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            def on_chunk(self, chunk, meta):
                received_meta.append(meta.copy())
                return chunk

        pipeline.add(MetaCapture())

        asyncio.run(
            pipeline.on_chunk(
                "token",
                {
                    "session_id": "s1",
                    "domain": "software_engineering",
                    "routing_mode": "single",
                    "chunk_index": 3,
                },
            )
        )

        assert received_meta[0]["session_id"] == "s1"
        assert received_meta[0]["domain"] == "software_engineering"
        assert received_meta[0]["chunk_index"] == 3

    def test_on_chunk_redacts_content(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(ChunkRedactor())

        result = asyncio.run(pipeline.on_chunk("my SECRET data", {}))
        assert result == "my [REDACTED] data"

    def test_on_chunk_suppression_returns_empty_string(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(ChunkSuppressor())

        assert asyncio.run(pipeline.on_chunk("SKIP this", {})) == ""
        assert asyncio.run(pipeline.on_chunk("keep this", {})) == "keep this"

    def test_on_chunk_chaining_multiple_middleware(self) -> None:
        """Each middleware sees the previous one's output."""
        pipeline = MiddlewarePipeline()

        class Upper:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            def on_chunk(self, chunk, meta):
                return chunk.upper()

        class Exclaim:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            def on_chunk(self, chunk, meta):
                return chunk + "!"

        pipeline.add(Upper())
        pipeline.add(Exclaim())

        result = asyncio.run(pipeline.on_chunk("hello", {}))
        assert result == "HELLO!"

    def test_on_chunk_skipped_when_no_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        # No middleware registered — should pass through
        result = asyncio.run(pipeline.on_chunk("unchanged", {}))
        assert result == "unchanged"

    def test_on_chunk_fallback_on_exception(self) -> None:
        """When on_chunk raises, the current chunk value is preserved and pipeline continues."""
        pipeline = MiddlewarePipeline()
        pipeline.add(ChunkCrasher())

        # Should not raise — chunk passes through unchanged
        result = asyncio.run(pipeline.on_chunk("safe chunk", {}))
        assert result == "safe chunk"

    def test_on_chunk_skipped_for_middleware_without_hook(self) -> None:
        """Middleware that doesn't implement on_chunk is silently skipped."""
        pipeline = MiddlewarePipeline()
        pipeline.add(MinimalMiddleware())  # no on_chunk

        result = asyncio.run(pipeline.on_chunk("token", {}))
        assert result == "token"

    def test_on_chunk_async_implementation_works(self) -> None:
        """on_chunk can be async."""
        pipeline = MiddlewarePipeline()
        pipeline.add(ChunkSuppressor())  # ChunkSuppressor.on_chunk is async

        result = asyncio.run(pipeline.on_chunk("SKIP me", {}))
        assert result == ""

        result = asyncio.run(pipeline.on_chunk("keep me", {}))
        assert result == "keep me"


# ── before_batch / after_batch tests ─────────────────────────────────────────


class TestBatchHooks:
    def test_before_batch_called_with_job_dict(self) -> None:
        pipeline = MiddlewarePipeline()
        logger = BatchLogger()
        pipeline.add(logger)

        job = {"job_id": "j1", "n_queries": 3, "priority": "high"}
        result = asyncio.run(pipeline.before_batch(job))

        assert len(logger.before_calls) == 1
        assert logger.before_calls[0]["job_id"] == "j1"
        # before_batch can modify job
        assert result.get("_tagged") is True

    def test_after_batch_called_with_results(self) -> None:
        pipeline = MiddlewarePipeline()
        logger = BatchLogger()
        pipeline.add(logger)

        job = {"job_id": "j1"}
        results = [{"response": "r1", "u_score": 0.8}, {"response": "r2", "u_score": 0.7}]
        modified = asyncio.run(pipeline.after_batch(job, results))

        assert len(logger.after_calls) == 1
        # after_batch can modify results
        assert all(r.get("_post") is True for r in modified)

    def test_before_batch_stack_order(self) -> None:
        """before_batch runs in registration order."""
        pipeline = MiddlewarePipeline()
        order = []

        class A:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def before_batch(self, job):
                order.append("A")
                return job

        class B:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def before_batch(self, job):
                order.append("B")
                return job

        pipeline.add(A())
        pipeline.add(B())
        asyncio.run(pipeline.before_batch({"job_id": "j"}))
        assert order == ["A", "B"]

    def test_after_batch_reverse_order(self) -> None:
        """after_batch runs in reverse registration order."""
        pipeline = MiddlewarePipeline()
        order = []

        class A:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def after_batch(self, job, results):
                order.append("A")
                return results

        class B:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def after_batch(self, job, results):
                order.append("B")
                return results

        pipeline.add(A())
        pipeline.add(B())
        asyncio.run(pipeline.after_batch({}, []))
        assert order == ["B", "A"]

    def test_batch_hooks_skipped_when_no_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        job = {"job_id": "j1"}
        result = asyncio.run(pipeline.before_batch(job))
        assert result == job  # unchanged

        results = [{"r": 1}]
        modified = asyncio.run(pipeline.after_batch(job, results))
        assert modified == results

    def test_batch_hooks_skipped_for_middleware_without_them(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(MinimalMiddleware())  # no before/after_batch

        job = {"job_id": "j"}
        result = asyncio.run(pipeline.before_batch(job))
        assert result == job


# ── on_error tests ────────────────────────────────────────────────────────────


class TestOnError:
    def test_on_error_called_when_registered(self) -> None:
        pipeline = MiddlewarePipeline()
        recoverer = ErrorRecoverer(recovery={"response": "fallback", "u_score": 0.0})
        pipeline.add(recoverer)

        exc = RuntimeError("pipeline failed")
        req = {"query": "test", "session_id": "s1"}
        result = asyncio.run(pipeline.on_error(exc, req))

        assert len(recoverer.called_with) == 1
        assert recoverer.called_with[0][0] is exc
        assert result == {"response": "fallback", "u_score": 0.0}

    def test_on_error_returns_none_propagates(self) -> None:
        pipeline = MiddlewarePipeline()
        pipeline.add(ErrorRecoverer(recovery=None))

        result = asyncio.run(pipeline.on_error(RuntimeError("err"), {}))
        assert result is None

    def test_on_error_first_dict_wins_reverse_order(self) -> None:
        """on_error calls in reverse order; first to return a dict wins."""
        pipeline = MiddlewarePipeline()
        calls = []

        class First:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def on_error(self, exc, req):
                calls.append("First")
                return {"from": "First"}

        class Second:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def on_error(self, exc, req):
                calls.append("Second")
                return {"from": "Second"}

        pipeline.add(First())
        pipeline.add(Second())

        result = asyncio.run(pipeline.on_error(RuntimeError("e"), {}))
        # Reverse order: Second runs first, returns a dict → wins
        assert calls == ["Second"]
        assert result == {"from": "Second"}

    def test_on_error_none_when_no_middleware(self) -> None:
        pipeline = MiddlewarePipeline()
        result = asyncio.run(pipeline.on_error(RuntimeError("e"), {}))
        assert result is None

    def test_on_error_self_raising_caught(self) -> None:
        """If on_error itself raises, it's caught and pipeline continues."""
        pipeline = MiddlewarePipeline()

        class CrashingRecoverer:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def on_error(self, exc, req):
                raise RuntimeError("recoverer also crashed")

        class GoodRecoverer:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

            async def on_error(self, exc, req):
                return {"recovered": True}

        pipeline.add(GoodRecoverer())
        pipeline.add(CrashingRecoverer())  # runs first (reverse), crashes, then Good runs

        result = asyncio.run(pipeline.on_error(RuntimeError("e"), {}))
        assert result == {"recovered": True}


# ── Protocol compliance ───────────────────────────────────────────────────────


class TestProtocolCompliance:
    def test_minimal_middleware_still_validates(self) -> None:
        """before_query + after_response is sufficient — new hooks are optional."""
        from aua.plugins.interfaces import AUAMiddleware

        assert isinstance(MinimalMiddleware(), AUAMiddleware)

    def test_full_middleware_validates(self) -> None:
        from aua.plugins.interfaces import AUAMiddleware

        assert isinstance(ChunkRedactor(), AUAMiddleware)
        assert isinstance(BatchLogger(), AUAMiddleware)
        assert isinstance(ErrorRecoverer(), AUAMiddleware)

    def test_new_hooks_not_in_protocol_requirement(self) -> None:
        """on_chunk, before_batch, after_batch, on_error not required by Protocol."""
        from aua.plugins.interfaces import AUAMiddleware

        class BareMinimum:
            async def before_query(self, r):
                return r

            async def after_response(self, r):
                return r

        assert isinstance(BareMinimum(), AUAMiddleware)


# ── BatchWorker middleware injection ──────────────────────────────────────────


class TestBatchWorkerMiddleware:
    def test_batch_worker_accepts_middleware_kwarg(self, tmp_path) -> None:
        from aua.batch_queue import BatchQueue, BatchWorker
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(str(tmp_path / "test.db"))
        queue = BatchQueue(store)
        pipeline = MiddlewarePipeline()

        async def dummy_handle(req):
            return MagicMock(model_dump=lambda: {"response": "ok", "u_score": 0.8})

        worker = BatchWorker(queue, dummy_handle, middleware=pipeline)
        assert worker._middleware is pipeline

    def test_batch_worker_fires_before_and_after_batch(self, tmp_path) -> None:
        from aua.batch_queue import BatchQueue, BatchWorker
        from aua.state import SQLiteStateStore

        store = SQLiteStateStore(str(tmp_path / "test.db"))
        queue = BatchQueue(store)
        pipeline = MiddlewarePipeline()
        logger = BatchLogger()
        pipeline.add(logger)

        call_count = [0]

        async def dummy_handle(req):
            call_count[0] += 1
            resp = MagicMock()
            resp.model_dump.return_value = {"response": "ok", "u_score": 0.8}
            return resp

        worker = BatchWorker(queue, dummy_handle, middleware=pipeline)

        # Submit a job with 2 items
        job_id = queue.submit(["query one", "query two"], priority="high")
        job_row = queue.get_job(job_id)

        asyncio.run(worker._dispatch_job(job_id, job_row))

        # before_batch was called once
        assert len(logger.before_calls) == 1
        assert logger.before_calls[0]["n_queries"] == 2

        # after_batch was called once with 2 results
        assert len(logger.after_calls) == 1
        job_arg, results_arg = logger.after_calls[0]
        assert len(results_arg) == 2
        assert call_count[0] == 2
