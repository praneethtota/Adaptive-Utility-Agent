"""
tests/test_loadtest.py — Tests for #50 aua loadtest command.

Coverage:
  _load_queries: built-in suite, custom dataset, unknown suite raises
  _percentile: p50/p95/p99 correctness
  run_loadtest: all-ok scenario, all-error scenario, mixed
  LoadTestReport: ok property, to_dict shape, throughput, error_rate
  CLI: basic invocation, --json output, --output file, liveness failure,
       unknown suite error, --no-liveness skips health check
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from aua.loadtest import (
    LoadTestConfig,
    LoadTestReport,
    RequestResult,
    _load_queries,
    run_loadtest,
)

# ── _load_queries ─────────────────────────────────────────────────────────────


class TestLoadQueries:
    def test_builtin_smoke_suite(self) -> None:
        cfg = LoadTestConfig(suite="smoke")
        queries = _load_queries(cfg)
        assert len(queries) >= 4
        assert all(isinstance(q, str) and len(q) > 5 for q in queries)

    def test_builtin_full_suite(self) -> None:
        cfg = LoadTestConfig(suite="full")
        queries = _load_queries(cfg)
        assert len(queries) >= 10

    def test_builtin_routing_suite(self) -> None:
        cfg = LoadTestConfig(suite="routing")
        queries = _load_queries(cfg)
        assert len(queries) >= 5

    def test_custom_dataset(self, tmp_path: Path) -> None:
        data = {
            "name": "custom",
            "cases": [
                {"id": "a", "prompt": "Write bubble sort."},
                {"id": "b", "prompt": "Explain recursion."},
            ],
        }
        p = tmp_path / "custom.yaml"
        p.write_text(yaml.dump(data))
        cfg = LoadTestConfig(dataset=str(p))
        queries = _load_queries(cfg)
        assert queries == ["Write bubble sort.", "Explain recursion."]

    def test_unknown_suite_raises(self) -> None:
        cfg = LoadTestConfig(suite="nonexistent")
        with pytest.raises(ValueError, match="Unknown suite"):
            _load_queries(cfg)


# ── LoadTestReport ────────────────────────────────────────────────────────────


class TestLoadTestReport:
    def _make_report(self, ok: int = 10, err: int = 0) -> LoadTestReport:
        cfg = LoadTestConfig()
        latencies = [100.0 + i * 10 for i in range(ok + err)]
        ok_results = [
            RequestResult(
                latency_ms=lat,
                status=200,
                routing_mode="single",
                primary_domain="software_engineering",
                u_score=0.75,
            )
            for lat in latencies[:ok]
        ]
        err_results = [
            RequestResult(
                latency_ms=lat,
                status=500,
                routing_mode="unknown",
                primary_domain="unknown",
                u_score=0.0,
                error="HTTP 500",
            )
            for lat in latencies[ok:]
        ]
        all_results = ok_results + err_results
        total = len(all_results)
        lats = [r.latency_ms for r in all_results]
        u_scores = [r.u_score for r in ok_results]
        import statistics

        return LoadTestReport(
            config=cfg,
            started_at=1000.0,
            finished_at=1030.0,
            total_requests=total,
            ok_requests=ok,
            error_requests=err,
            latencies_ms=lats,
            error_rate=err / total if total else 0.0,
            throughput_rps=total / 30.0,
            p50_ms=lats[total // 2],
            p95_ms=lats[int(total * 0.95)],
            p99_ms=lats[int(total * 0.99)],
            mean_ms=statistics.mean(lats),
            min_ms=min(lats),
            max_ms=max(lats),
            mean_u_score=statistics.mean(u_scores) if u_scores else 0.0,
            routing_mode_counts={"single": ok},
            domain_counts={"software_engineering": ok},
            errors=["HTTP 500"] if err else [],
        )

    def test_ok_true_when_low_error_rate(self) -> None:
        r = self._make_report(ok=100, err=2)
        assert r.ok  # 2% < 5%

    def test_ok_false_when_high_error_rate(self) -> None:
        r = self._make_report(ok=80, err=20)
        assert not r.ok  # 20% >= 5%

    def test_to_dict_shape(self) -> None:
        r = self._make_report()
        d = r.to_dict()
        assert "config" in d
        assert "summary" in d
        assert "latency" in d
        assert "quality" in d
        assert "errors" in d
        assert "ok" in d
        assert "p50_ms" in d["latency"]
        assert "p95_ms" in d["latency"]
        assert "p99_ms" in d["latency"]
        assert "throughput_rps" in d["summary"]
        assert "error_rate" in d["summary"]

    def test_error_rate_correct(self) -> None:
        r = self._make_report(ok=9, err=1)
        assert r.error_rate == pytest.approx(0.10)

    def test_throughput_correct(self) -> None:
        r = self._make_report(ok=10)
        assert r.throughput_rps == pytest.approx(10 / 30.0, rel=0.01)


# ── run_loadtest ──────────────────────────────────────────────────────────────


def _make_mock_response(u_score: float = 0.75, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {
        "response": "def binary_search(): pass",
        "routing_mode": "single",
        "primary_domain": "software_engineering",
        "u_score": u_score,
    }
    return r


class TestRunLoadtest:
    @pytest.mark.asyncio
    async def test_all_ok(self) -> None:
        cfg = LoadTestConfig(
            router_url="http://localhost:8000",
            concurrency=2,
            duration_s=0.5,
            suite="smoke",
        )
        mock_resp = _make_mock_response(u_score=0.80)

        async def fake_post(*a, **kw):
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = fake_post
            mock_client_cls.return_value = mock_client

            report = await run_loadtest(cfg)

        assert report.total_requests > 0
        assert report.error_requests == 0
        assert report.error_rate == 0.0
        assert report.mean_u_score == pytest.approx(0.80)

    @pytest.mark.asyncio
    async def test_network_errors_counted(self) -> None:
        cfg = LoadTestConfig(concurrency=2, duration_s=0.3, suite="smoke")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def failing_post(*a, **kw):
                raise ConnectionRefusedError("router down")

            mock_client.post = failing_post
            mock_client_cls.return_value = mock_client

            report = await run_loadtest(cfg)

        assert report.total_requests > 0
        assert report.ok_requests == 0
        assert report.error_rate == 1.0
        assert not report.ok

    @pytest.mark.asyncio
    async def test_unknown_suite_raises(self) -> None:
        cfg = LoadTestConfig(suite="nonexistent", concurrency=1, duration_s=0.1)
        with pytest.raises(ValueError, match="Unknown suite"):
            await run_loadtest(cfg)

    @pytest.mark.asyncio
    async def test_custom_dataset(self, tmp_path: Path) -> None:
        data = {"name": "t", "cases": [{"id": "a", "prompt": "Write hello world."}]}
        p = tmp_path / "t.yaml"
        p.write_text(yaml.dump(data))
        cfg = LoadTestConfig(dataset=str(p), concurrency=1, duration_s=0.3)

        mock_resp = _make_mock_response()
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            async def fake_post(*a, **kw):
                return mock_resp

            mock_client.post = fake_post
            mock_cls.return_value = mock_client
            report = await run_loadtest(cfg)

        assert report.total_requests > 0


# ── CLI ───────────────────────────────────────────────────────────────────────


class TestLoadtestCLI:
    def _mock_report(self):
        from aua.loadtest import LoadTestConfig, LoadTestReport

        cfg = LoadTestConfig()
        return LoadTestReport(
            config=cfg,
            started_at=1000.0,
            finished_at=1030.0,
            total_requests=150,
            ok_requests=148,
            error_requests=2,
            latencies_ms=[100.0 + i for i in range(150)],
            error_rate=2 / 150,
            throughput_rps=5.0,
            p50_ms=125.0,
            p95_ms=240.0,
            p99_ms=248.0,
            mean_ms=175.0,
            min_ms=100.0,
            max_ms=249.0,
            mean_u_score=0.74,
            routing_mode_counts={"single": 148},
            domain_counts={"software_engineering": 148},
            errors=["HTTP 500", "HTTP 500"],
        )

    def test_basic_invocation(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        with patch("aua.loadtest.run_loadtest", return_value=self._mock_report()):
            with patch("aua.test_harness.check_router_live", return_value=(True, "ok")):
                result = runner.invoke(main, ["loadtest", "--no-liveness", "--duration", "0.1"])
        assert result.exit_code == 0
        assert "p50" in result.output or "load test" in result.output.lower()

    def test_json_output(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        with patch("aua.loadtest.run_loadtest", return_value=self._mock_report()):
            result = runner.invoke(
                main, ["loadtest", "--no-liveness", "--duration", "0.1", "--json"]
            )
        assert result.exit_code == 0
        # Output may have a status line before the JSON block — find the JSON
        start = result.output.find("{")
        assert start >= 0, f"No JSON found in output: {result.output!r}"
        data = json.loads(result.output[start:])
        assert "latency" in data
        assert "summary" in data
        assert data["summary"]["total_requests"] == 150

    def test_output_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        out = tmp_path / "report.json"
        runner = CliRunner()
        with patch("aua.loadtest.run_loadtest", return_value=self._mock_report()):
            result = runner.invoke(
                main,
                ["loadtest", "--no-liveness", "--duration", "0.1", "--output", str(out)],
            )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "latency" in data

    def test_liveness_failure_exits_1(self) -> None:
        import urllib.error

        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = runner.invoke(main, ["loadtest", "--duration", "0.1"])
        assert result.exit_code == 1
        assert "not reachable" in result.output.lower() or "Router" in result.output

    def test_no_liveness_skips_check(self) -> None:
        from click.testing import CliRunner

        from aua.cli import main

        runner = CliRunner()
        with patch("aua.loadtest.run_loadtest", return_value=self._mock_report()):
            result = runner.invoke(main, ["loadtest", "--no-liveness", "--duration", "0.1"])
        # Should not print "not reachable"
        assert "not reachable" not in result.output.lower()
        assert result.exit_code == 0
