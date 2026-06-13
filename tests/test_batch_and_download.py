"""
tests/test_batch_and_download.py — Tests for #56 (persistent batch queue)
and #57 (model auto-download).

#56 tests
─────────
- BatchQueue.submit() returns a UUID job_id and creates DB rows
- Priority ordering: high before normal before low
- Cold-start recovery: interrupted 'running' jobs reset to 'pending' on init
- Partial results available before job completes
- Item lifecycle: pending → running → done / error
- BatchWorker dispatches all items and transitions job to 'done'
- list_jobs() filterable by status
- Unknown job_id returns None from get_job()

#57 tests
─────────
- _ollama_model_present(): returns True when tag is in /api/tags JSON
- _ollama_model_present(): returns False when tag is absent or server down
- _ollama_pull(): skips pull when model already present
- _ollama_pull(): calls subprocess.run when model absent
- _hf_download(): skips when try_to_load_from_cache succeeds
- _hf_download(): exits on 401/403 HfHubHTTPError (gated model)
- _hf_download(): warns but continues when HF_TOKEN is unset
- serve() passes no_download=True through to _start_specialist (dry_run path)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aua.batch_queue import BatchQueue, BatchWorker
from aua.state import SQLiteStateStore

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> SQLiteStateStore:
    """In-memory-equivalent: fresh SQLite DB per test."""
    return SQLiteStateStore(db_path=tmp_path / "test.db")


@pytest.fixture
def bq(tmp_db: SQLiteStateStore) -> BatchQueue:
    return BatchQueue(tmp_db)


# ── #56: BatchQueue unit tests ────────────────────────────────────────────────


class TestBatchQueueSubmit:
    def test_returns_uuid_string(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q1", "q2"])
        uuid.UUID(job_id)  # raises if not a valid UUID

    def test_creates_job_row(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q1", "q2"], priority="high")
        data = bq.get_job(job_id)
        assert data is not None
        assert data["n_queries"] == 2
        assert data["priority"] == "high"
        assert data["status"] == "pending"

    def test_creates_item_rows(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["alpha", "beta", "gamma"])
        data = bq.get_job(job_id)
        assert data["n_pending"] == 3
        assert data["n_running"] == 0

    def test_invalid_priority_defaults_to_normal(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"], priority="turbo")
        data = bq.get_job(job_id)
        assert data["priority"] == "normal"

    def test_meta_stored_and_returned(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"], meta={"caller": "test", "ref": 42})
        data = bq.get_job(job_id)
        assert data["meta"]["caller"] == "test"
        assert data["meta"]["ref"] == 42

    def test_empty_queries_raises(self, bq: BatchQueue) -> None:
        with pytest.raises(ValueError, match="1–500"):
            bq.submit([])

    def test_too_many_queries_raises(self, bq: BatchQueue) -> None:
        with pytest.raises(ValueError, match="1–500"):
            bq.submit(["q"] * 501)


class TestBatchQueuePriority:
    def test_high_before_normal_before_low(self, bq: BatchQueue) -> None:
        low_id = bq.submit(["low"], priority="low")
        normal_id = bq.submit(["normal"], priority="normal")
        high_id = bq.submit(["high"], priority="high")

        first = bq.next_pending_job()
        assert first["job_id"] == high_id

        bq.claim_job(high_id)
        bq.finish_job(high_id)

        second = bq.next_pending_job()
        assert second["job_id"] == normal_id

        bq.claim_job(normal_id)
        bq.finish_job(normal_id)

        third = bq.next_pending_job()
        assert third["job_id"] == low_id

    def test_fifo_within_same_priority(self, bq: BatchQueue) -> None:
        id1 = bq.submit(["first"], priority="normal")
        time.sleep(0.01)
        _ = bq.submit(["second"], priority="normal")

        first = bq.next_pending_job()
        assert first["job_id"] == id1

    def test_no_pending_returns_none(self, bq: BatchQueue) -> None:
        assert bq.next_pending_job() is None


class TestBatchQueueClaim:
    def test_claim_transitions_to_running(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"])
        assert bq.claim_job(job_id) is True
        data = bq.get_job(job_id)
        assert data["status"] == "running"

    def test_claim_already_running_returns_false(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"])
        bq.claim_job(job_id)
        assert bq.claim_job(job_id) is False


class TestBatchQueueItemLifecycle:
    def test_mark_item_done_increments_n_done(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q1", "q2"])
        bq.claim_job(job_id)
        items = bq.pending_items(job_id)

        bq.mark_item_running(items[0]["item_id"])
        bq.mark_item_done(items[0]["item_id"], {"response": "ok"})

        data = bq.get_job(job_id)
        assert data["n_done"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["response"] == "ok"

    def test_mark_item_error_increments_n_errors(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"])
        bq.claim_job(job_id)
        items = bq.pending_items(job_id)

        bq.mark_item_error(items[0]["item_id"], "specialist timeout")

        data = bq.get_job(job_id)
        assert data["n_errors"] == 1
        assert data["errors"][0]["error"] == "specialist timeout"

    def test_partial_results_before_finish(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q1", "q2"])
        bq.claim_job(job_id)
        items = bq.pending_items(job_id)

        # Only complete the first item
        bq.mark_item_done(items[0]["item_id"], {"response": "partial"})

        data = bq.get_job(job_id)
        assert data["status"] == "running"  # not finished yet
        assert len(data["results"]) == 1  # but result is visible now

    def test_finish_job_sets_done(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"])
        bq.claim_job(job_id)
        bq.finish_job(job_id)
        assert bq.get_job(job_id)["status"] == "done"


class TestBatchQueueRecovery:
    def test_recover_interrupted_resets_running_to_pending(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q1", "q2"])
        bq.claim_job(job_id)  # simulate interrupted mid-flight
        items = bq.pending_items(job_id)
        bq.mark_item_running(items[0]["item_id"])

        n = bq.recover_interrupted()
        assert n == 1  # one job was reset

        data = bq.get_job(job_id)
        assert data["status"] == "pending"

    def test_recover_no_interrupted_returns_zero(self, bq: BatchQueue) -> None:
        job_id = bq.submit(["q"])
        bq.claim_job(job_id)
        bq.finish_job(job_id)  # done — not interrupted
        assert bq.recover_interrupted() == 0


class TestBatchQueueList:
    def test_list_returns_all_jobs(self, bq: BatchQueue) -> None:
        bq.submit(["q1"])
        bq.submit(["q2"])
        jobs = bq.list_jobs()
        assert len(jobs) == 2

    def test_list_filtered_by_status(self, bq: BatchQueue) -> None:
        id1 = bq.submit(["q1"])
        id2 = bq.submit(["q2"])
        bq.claim_job(id1)
        bq.finish_job(id1)

        done_jobs = bq.list_jobs(status="done")
        assert len(done_jobs) == 1
        assert done_jobs[0]["job_id"] == id1

        pending_jobs = bq.list_jobs(status="pending")
        assert len(pending_jobs) == 1
        assert pending_jobs[0]["job_id"] == id2

    def test_unknown_job_returns_none(self, bq: BatchQueue) -> None:
        assert bq.get_job("nonexistent-job-id") is None


# ── #56: BatchWorker integration test ────────────────────────────────────────


class TestBatchWorker:
    @pytest.mark.asyncio
    async def test_worker_dispatches_all_items(self, bq: BatchQueue) -> None:
        """Worker picks up a job, runs all items, and marks it done."""
        call_count = 0

        async def fake_handle(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.model_dump.return_value = {"response": f"answer to: {req.query}"}
            return result

        job_id = bq.submit(["q1", "q2", "q3"], max_parallel=2)

        worker = BatchWorker(bq, fake_handle)
        worker.start()

        # Give the worker time to drain the job (poll interval is 2s but job
        # should complete immediately with a fake handle)
        deadline = time.time() + 10
        while time.time() < deadline:
            await asyncio.sleep(0.1)
            data = bq.get_job(job_id)
            if data["status"] == "done":
                break

        worker.stop()

        data = bq.get_job(job_id)
        assert data["status"] == "done"
        assert data["n_done"] == 3
        assert data["n_errors"] == 0
        assert len(data["results"]) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_worker_handles_item_error_gracefully(self, bq: BatchQueue) -> None:
        """A failing handle records item error without crashing the worker."""

        async def failing_handle(req: Any) -> Any:
            raise RuntimeError("specialist exploded")

        job_id = bq.submit(["q1"])
        worker = BatchWorker(bq, failing_handle)
        worker.start()

        deadline = time.time() + 10
        while time.time() < deadline:
            await asyncio.sleep(0.1)
            if bq.get_job(job_id)["status"] == "done":
                break

        worker.stop()

        data = bq.get_job(job_id)
        assert data["status"] == "done"
        assert data["n_errors"] == 1
        assert "specialist exploded" in data["errors"][0]["error"]


# ── #57: model download unit tests ───────────────────────────────────────────


class TestOllamaModelPresent:
    def test_returns_true_when_model_in_tags(self) -> None:
        from aua.serve import _ollama_model_present

        tags_response = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "llama3:8b"}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                json=lambda: tags_response,
                status_code=200,
            )
            assert _ollama_model_present("qwen2.5-coder:7b") is True

    def test_returns_false_when_model_absent(self) -> None:
        from aua.serve import _ollama_model_present

        tags_response = {"models": [{"name": "llama3:8b"}]}
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(json=lambda: tags_response)
            assert _ollama_model_present("qwen2.5-coder:7b") is False

    def test_returns_false_on_connection_error(self) -> None:
        from aua.serve import _ollama_model_present

        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert _ollama_model_present("any-model") is False


class TestOllamaPull:
    def test_skips_pull_when_model_present(self) -> None:
        from aua.serve import _ollama_pull

        with patch("aua.serve._ollama_model_present", return_value=True):
            with patch("subprocess.run") as mock_run:
                _ollama_pull("qwen2.5-coder:7b")
                mock_run.assert_not_called()

    def test_runs_pull_when_model_absent(self) -> None:
        from aua.serve import _ollama_pull

        with patch("aua.serve._ollama_model_present", return_value=False):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                _ollama_pull("qwen2.5-coder:7b")
                mock_run.assert_called_once()
                assert "pull" in mock_run.call_args[0][0]

    def test_exits_on_pull_failure(self) -> None:
        from aua.serve import _ollama_pull

        with patch("aua.serve._ollama_model_present", return_value=False):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                with pytest.raises(SystemExit):
                    _ollama_pull("nonexistent-model")


class TestHfDownload:
    """
    Tests for _hf_download().

    huggingface_hub is an optional dependency (not in [dev] extras), so
    every test injects a fake module into sys.modules before calling
    _hf_download(). This makes the suite pass in clean CI environments.
    """

    def _fake_hf_module(self, cached_path=None, download_side_effect=None):
        """Return a MagicMock that stands in for the huggingface_hub module."""
        fake = MagicMock()
        fake.try_to_load_from_cache.return_value = cached_path
        if download_side_effect is not None:
            fake.snapshot_download.side_effect = download_side_effect
        return fake

    def test_skips_when_cached(self, tmp_path: Path) -> None:
        from aua.serve import _hf_download

        fake_hf = self._fake_hf_module(cached_path=str(tmp_path / "config.json"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            _hf_download("meta-llama/Llama-3-8B")
        fake_hf.snapshot_download.assert_not_called()

    def test_downloads_when_not_cached(self) -> None:
        from aua.serve import _hf_download

        fake_hf = self._fake_hf_module(cached_path=None)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch("shutil.disk_usage") as mock_du:
                mock_du.return_value = MagicMock(free=50 * 1024**3)
                _hf_download("Qwen/Qwen2.5-Coder-7B-AWQ")
        fake_hf.snapshot_download.assert_called_once()
        call_kwargs = fake_hf.snapshot_download.call_args[1]
        assert call_kwargs["repo_id"] == "Qwen/Qwen2.5-Coder-7B-AWQ"

    def test_exits_on_401_gated_model(self) -> None:
        from aua.serve import _hf_download

        fake_hf = self._fake_hf_module(
            cached_path=None,
            download_side_effect=Exception("401 Unauthorized"),
        )
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch("shutil.disk_usage") as mock_du:
                mock_du.return_value = MagicMock(free=50 * 1024**3)
                with pytest.raises(SystemExit):
                    _hf_download("meta-llama/Llama-3-8B")

    def test_warns_on_low_disk_space(self) -> None:
        from aua.serve import _hf_download

        fake_hf = self._fake_hf_module(cached_path=None)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            with patch("shutil.disk_usage") as mock_du:
                mock_du.return_value = MagicMock(free=2 * 1024**3)  # only 2 GB
                # Should complete without raising (warning only)
                _hf_download("small/model")

    def test_graceful_when_huggingface_hub_missing(self) -> None:
        from aua.serve import _hf_download

        # Simulate package not installed: inject None so the import raises ImportError
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            # Should not raise — prints a warning and returns
            _hf_download("any/model")


class TestNoDownloadFlag:
    def test_no_download_skips_hf_download_on_vllm(self) -> None:
        """When no_download=True, _hf_download should not be called."""
        from aua.config import (
            SpecialistConfig,
        )

        spec = SpecialistConfig(
            name="swe",
            model="Qwen/Qwen2.5-Coder-7B-AWQ",
            port=19001,
            field="software_engineering",
            backend="vllm",
        )

        import aua.serve as serve_mod
        from aua.serve import _start_specialist

        with patch.object(serve_mod, "_hf_download") as mock_hf:
            # dry_run=True so no actual subprocess, but no_download should gate _hf_download
            _start_specialist(spec, dry_run=True, timeout=10, runtime=MagicMock(), no_download=True)
            mock_hf.assert_not_called()

    def test_download_called_when_flag_absent(self) -> None:
        import aua.serve as serve_mod
        from aua.config import SpecialistConfig
        from aua.serve import _start_specialist

        spec = SpecialistConfig(
            name="swe",
            model="Qwen/Qwen2.5-Coder-7B-AWQ",
            port=19001,
            field="software_engineering",
            backend="vllm",
        )

        with patch.object(serve_mod, "_hf_download") as mock_hf:
            # dry_run=False so download path is active; stub out subprocess
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock(pid=999, poll=lambda: None)
                with patch.object(serve_mod, "_wait_healthy"):
                    with patch.object(serve_mod, "_write_pid_file"):
                        with patch.object(serve_mod, "_open_log", return_value=None):
                            _start_specialist(
                                spec,
                                dry_run=False,
                                timeout=5,
                                runtime=MagicMock(),
                                no_download=False,
                            )
            mock_hf.assert_called_once_with("Qwen/Qwen2.5-Coder-7B-AWQ")
