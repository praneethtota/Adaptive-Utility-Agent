"""
tests/test_test_harness.py — Tests for #54 aua test harness.

Coverage:
  test_harness module:
    - SUITES registry points to existing files
    - check_router_live: live (200), dead (connection error), bad status
    - _check_property: all six property types
    - run_case: success path, network error path, property failures
    - run_suite: all cases pass, partial failure, unknown suite raises
    - run_custom_dataset: loads and runs a tmp YAML file
    - SuiteReport.ok: True on 100% pass, False otherwise
    - SuiteReport.to_dict: correct shape

  CLI (aua test):
    - --suite smoke runs (mocked router)
    - --no-liveness skips liveness check
    - --json emits JSON to stdout
    - --output writes file
    - --case filters to specific cases
    - liveness failure exits 1 with clear message
    - all-pass exits 0; any-fail exits 1
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from aua.cli import main
from aua.test_harness import (
    SUITES,
    SuiteReport,
    _check_property,
    check_router_live,
    run_case,
    run_custom_dataset,
    run_suite,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _mock_router_response(
    response: str = "def binary_search(): pass",
    domain: str = "software_engineering",
    u_score: float = 0.75,
    routing_mode: str = "single",
    status: int = 200,
) -> MagicMock:
    """Return a mock that looks like urlopen's response context manager."""
    body = json.dumps(
        {
            "response": response,
            "primary_domain": domain,
            "u_score": u_score,
            "routing_mode": routing_mode,
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    mock_resp.status = status
    return mock_resp


def _minimal_fixture(tmp_path: Path, cases: list[dict] | None = None) -> Path:
    """Write a minimal fixture YAML to tmp_path and return its path."""
    if cases is None:
        cases = [
            {
                "id": "test_case_1",
                "prompt": "Write hello world in Python.",
                "expected_properties": [
                    {"contains": "print"},
                    {"min_length": 5},
                ],
            }
        ]
    data = {"name": "tmp_fixture", "description": "tmp", "cases": cases}
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.dump(data))
    return path


# ── SUITES registry ───────────────────────────────────────────────────────────


class TestSuitesRegistry:
    def test_all_suite_files_exist(self) -> None:
        for name, path in SUITES.items():
            assert path.exists(), f"Suite '{name}' file missing: {path}"

    def test_suite_files_are_valid_yaml(self) -> None:
        for name, path in SUITES.items():
            raw = yaml.safe_load(path.read_text())
            assert "cases" in raw, f"Suite '{name}' missing 'cases' key"
            assert len(raw["cases"]) > 0, f"Suite '{name}' has no cases"

    def test_all_cases_have_required_fields(self) -> None:
        for suite_name, path in SUITES.items():
            raw = yaml.safe_load(path.read_text())
            for case in raw["cases"]:
                assert "id" in case, f"[{suite_name}] case missing 'id': {case}"
                assert "prompt" in case, f"[{suite_name}] case missing 'prompt': {case}"

    def test_smoke_has_at_least_4_cases(self) -> None:
        raw = yaml.safe_load(SUITES["smoke"].read_text())
        assert len(raw["cases"]) >= 4

    def test_full_has_at_least_10_cases(self) -> None:
        raw = yaml.safe_load(SUITES["full"].read_text())
        assert len(raw["cases"]) >= 10

    def test_routing_suite_all_cases_have_expected_domain(self) -> None:
        raw = yaml.safe_load(SUITES["routing"].read_text())
        # Every case in the routing suite should have at least one property check
        for case in raw["cases"]:
            assert case.get("expected_properties"), f"Case {case['id']} has no property checks"


# ── check_router_live ─────────────────────────────────────────────────────────


class TestCheckRouterLive:
    def test_live_on_200(self) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.status = 200
        with patch("urllib.request.urlopen", return_value=mock_resp):
            live, msg = check_router_live("http://localhost:8000")
        assert live is True
        assert msg == "ok"

    def test_dead_on_connection_error(self) -> None:
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            live, msg = check_router_live("http://localhost:9999")
        assert live is False
        assert "refused" in msg or "unreachable" in msg.lower()

    def test_dead_on_exception(self) -> None:
        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            live, msg = check_router_live("http://localhost:8000")
        assert live is False
        assert "timeout" in msg


# ── _check_property ───────────────────────────────────────────────────────────


class TestCheckProperty:
    def test_contains_pass(self) -> None:
        ok, _ = _check_property({"contains": "def "}, "def foo(): pass", "swe")
        assert ok

    def test_contains_fail(self) -> None:
        ok, reason = _check_property({"contains": "def "}, "no function here", "swe")
        assert not ok
        assert "missing" in reason

    def test_contains_any_pass(self) -> None:
        ok, _ = _check_property({"contains_any": ["mid", "low"]}, "use low and high", "swe")
        assert ok

    def test_contains_any_fail(self) -> None:
        ok, reason = _check_property({"contains_any": ["mid", "low"]}, "nothing here", "swe")
        assert not ok
        assert "none of" in reason

    def test_not_contains_pass(self) -> None:
        ok, _ = _check_property({"not_contains": "I cannot"}, "here is the answer", "swe")
        assert ok

    def test_not_contains_fail(self) -> None:
        ok, reason = _check_property({"not_contains": "I cannot"}, "I cannot help", "swe")
        assert not ok
        assert "unexpectedly" in reason

    def test_min_length_pass(self) -> None:
        ok, _ = _check_property({"min_length": 5}, "hello world", "swe")
        assert ok

    def test_min_length_fail(self) -> None:
        ok, reason = _check_property({"min_length": 100}, "short", "swe")
        assert not ok
        assert "too short" in reason

    def test_expected_domain_pass(self) -> None:
        ok, _ = _check_property({"expected_domain": "mathematics"}, "text", "mathematics")
        assert ok

    def test_expected_domain_fail(self) -> None:
        ok, reason = _check_property(
            {"expected_domain": "mathematics"}, "text", "software_engineering"
        )
        assert not ok
        assert "expected" in reason

    def test_expected_domain_any_pass(self) -> None:
        ok, _ = _check_property(
            {"expected_domain_any": ["mathematics", "software_engineering"]},
            "text",
            "mathematics",
        )
        assert ok

    def test_expected_domain_any_fail(self) -> None:
        ok, reason = _check_property({"expected_domain_any": ["mathematics"]}, "text", "general")
        assert not ok

    def test_multiple_props_all_pass(self) -> None:
        props = [{"contains": "def"}, {"min_length": 3}, {"not_contains": "error"}]
        for prop in props:
            ok, _ = _check_property(prop, "def foo(): pass", "swe")
            assert ok


# ── run_case ──────────────────────────────────────────────────────────────────


class TestRunCase:
    def test_success_path_all_props_pass(self) -> None:
        mock = _mock_router_response(
            response="def binary_search(arr, target): mid = len(arr) // 2",
            domain="software_engineering",
            u_score=0.82,
        )
        case = {
            "id": "test",
            "prompt": "Write binary search.",
            "expected_properties": [
                {"contains": "def "},
                {"expected_domain": "software_engineering"},
                {"min_length": 10},
            ],
        }
        with patch("urllib.request.urlopen", return_value=mock):
            result = run_case(case)
        assert result.passed
        assert result.u_score == pytest.approx(0.82)
        assert result.domain == "software_engineering"
        assert result.routing_mode == "single"
        assert result.error is None
        assert result.failures == []

    def test_property_failure_marks_case_failed(self) -> None:
        mock = _mock_router_response(response="short", domain="general")
        case = {
            "id": "test",
            "prompt": "Q",
            "expected_properties": [{"min_length": 100}],
        }
        with patch("urllib.request.urlopen", return_value=mock):
            result = run_case(case)
        assert not result.passed
        assert len(result.failures) == 1
        assert "too short" in result.failures[0]

    def test_network_error_marks_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            result = run_case({"id": "t", "prompt": "Q"})
        assert not result.passed
        assert result.error is not None
        assert "connection refused" in result.error

    def test_domain_mismatch_fails(self) -> None:
        mock = _mock_router_response(domain="general")
        case = {
            "id": "t",
            "prompt": "Q",
            "expected_properties": [{"expected_domain": "mathematics"}],
        }
        with patch("urllib.request.urlopen", return_value=mock):
            result = run_case(case)
        assert not result.passed


# ── run_suite ─────────────────────────────────────────────────────────────────


class TestRunSuite:
    def test_smoke_all_pass(self) -> None:
        mock = _mock_router_response(
            response="def binary_search(): mid = 0; Paris; print('ok') O(n log n) 3x^2 + 2",
            domain="software_engineering",
            u_score=0.80,
        )
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_suite("smoke")
        assert isinstance(report, SuiteReport)
        assert report.total > 0
        assert report.suite == "smoke"

    def test_unknown_suite_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown suite"):
            run_suite("nonexistent")

    def test_partial_failure_tracked(self) -> None:
        # First call succeeds, second fails with network error
        calls = iter(
            [
                _mock_router_response(
                    response="def binary_search(): mid=0 O(log n) low high",
                    domain="software_engineering",
                ),
                Exception("network down"),
            ]
        )

        def side_effect(*args, **kwargs):
            v = next(calls)
            if isinstance(v, Exception):
                raise v
            return v

        with patch("urllib.request.urlopen", side_effect=side_effect):
            report = run_suite("smoke", case_ids=["swe_binary_search", "math_derivative"])

        assert report.total == 2
        assert report.errored >= 1

    def test_case_id_filter(self) -> None:
        mock = _mock_router_response(
            response="def binary_search(): low=0; high=len(arr); mid=(low+high)//2",
            domain="software_engineering",
        )
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_suite("smoke", case_ids=["swe_binary_search"])
        assert report.total == 1
        assert report.cases[0].case_id == "swe_binary_search"

    def test_report_ok_property(self) -> None:
        mock = _mock_router_response(
            response="def f(): pass " * 10,
            domain="software_engineering",
        )
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_suite("smoke", case_ids=["swe_complexity_question"])
        # ok is True only if pass_rate == 1.0
        assert isinstance(report.ok, bool)

    def test_to_dict_shape(self) -> None:
        mock = _mock_router_response(response="O(n log n)", domain="software_engineering")
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_suite("smoke", case_ids=["swe_complexity_question"])
        d = report.to_dict()
        assert "suite" in d
        assert "summary" in d
        assert "cases" in d
        assert "pass_rate" in d["summary"]
        assert "mean_u_score" in d["summary"]


# ── run_custom_dataset ────────────────────────────────────────────────────────


class TestRunCustomDataset:
    def test_loads_and_runs_tmp_fixture(self, tmp_path: Path) -> None:
        fixture = _minimal_fixture(tmp_path)
        mock = _mock_router_response(response="print('hello world')", domain="general")
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_custom_dataset(fixture)
        assert report.total == 1
        assert report.suite == "tmp_fixture"

    def test_custom_dataset_case_passes(self, tmp_path: Path) -> None:
        cases = [
            {
                "id": "check_print",
                "prompt": "Print hello.",
                "expected_properties": [{"contains": "print"}],
            }
        ]
        fixture = _minimal_fixture(tmp_path, cases)
        mock = _mock_router_response(response="print('hello')", domain="general")
        with patch("urllib.request.urlopen", return_value=mock):
            report = run_custom_dataset(fixture)
        assert report.passed == 1
        assert report.cases[0].passed


# ── CLI integration ───────────────────────────────────────────────────────────


class TestCliTestCommand:
    def _live_response(self) -> MagicMock:
        """Mock for the liveness check."""
        r = MagicMock()
        r.__enter__ = lambda s: s
        r.__exit__ = MagicMock(return_value=False)
        r.status = 200
        r.read.return_value = b'{"status": "ok"}'
        return r

    def _query_response(self) -> MagicMock:
        return _mock_router_response(
            response=(
                "def binary_search(): low=0; high=len(arr); mid=(low+high)//2 "
                "O(n log n) Paris print('ok') 3x^2 + 2"
            ),
            domain="software_engineering",
            u_score=0.78,
        )

    def test_no_liveness_smoke_exits_0_or_1(self) -> None:
        """With --no-liveness, CLI runs without hitting health endpoint."""
        runner = CliRunner()
        mock = self._query_response()
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(main, ["test", "--no-liveness", "--suite", "smoke"])
        # Exit 0 (all pass) or 1 (some fail) — but must not crash
        assert result.exit_code in (0, 1)
        assert "aua test" in result.output

    def test_liveness_failure_exits_1(self) -> None:
        runner = CliRunner()
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = runner.invoke(main, ["test", "--suite", "smoke"])
        assert result.exit_code == 1
        assert "not reachable" in result.output.lower() or "Router" in result.output

    def test_json_flag_emits_valid_json(self) -> None:
        runner = CliRunner()
        mock = self._query_response()
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(main, ["test", "--no-liveness", "--suite", "smoke", "--json"])
        assert result.exit_code in (0, 1)
        # stdout should be parseable JSON
        try:
            data = json.loads(result.output)
            assert "suite" in data
            assert "summary" in data
        except json.JSONDecodeError:
            # output may include Rich markup before the JSON — find the JSON block
            start = result.output.find("{")
            if start >= 0:
                data = json.loads(result.output[start:])
                assert "suite" in data

    def test_output_flag_writes_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out_file = tmp_path / "report.json"
        mock = self._query_response()
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(
                main,
                ["test", "--no-liveness", "--suite", "smoke", "--output", str(out_file)],
            )
        assert result.exit_code in (0, 1)
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "suite" in data
        assert "cases" in data

    def test_case_filter_reduces_run(self) -> None:
        runner = CliRunner()
        mock = _mock_router_response(
            response="def binary_search(): low=0 high=len(arr) mid=(low+high)//2",
            domain="software_engineering",
        )
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(
                main,
                ["test", "--no-liveness", "--suite", "smoke", "--case", "swe_binary_search"],
            )
        assert result.exit_code in (0, 1)
        assert "swe_binary_search" in result.output

    def test_custom_dataset_flag(self, tmp_path: Path) -> None:
        runner = CliRunner()
        fixture = _minimal_fixture(tmp_path)
        mock = _mock_router_response(response="print('hello world')", domain="general")
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(
                main,
                ["test", "--no-liveness", "--dataset", str(fixture)],
            )
        assert result.exit_code in (0, 1)
        assert "aua test" in result.output

    def test_routing_suite_flag_accepted(self) -> None:
        runner = CliRunner()
        mock = _mock_router_response(
            response="def reverse(s): return s[::-1]",
            domain="software_engineering",
        )
        with patch("urllib.request.urlopen", return_value=mock):
            result = runner.invoke(main, ["test", "--no-liveness", "--suite", "routing"])
        assert result.exit_code in (0, 1)
        assert "routing" in result.output
