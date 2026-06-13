"""
aua/test_harness.py — Built-in integration test harness (#54).

Runs the bundled fixture datasets against a live AUA router.
Called by `aua test`; also importable for programmatic use.

Suites (bundled in aua/fixtures/):
    smoke   — 6 cases, < 60 s  — router liveness + basic routing
    full    — 15 cases, 3-10 min — regression, edge cases, routing
    routing — 9 cases, 1-3 min  — domain classification correctness

Usage:
    from aua.test_harness import run_suite, SUITES
    report = run_suite("smoke", router_url="http://localhost:8000")
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Suite registry ────────────────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent / "fixtures"

SUITES: dict[str, Path] = {
    "smoke": _FIXTURES_DIR / "smoke.yaml",
    "full": _FIXTURES_DIR / "full.yaml",
    "routing": _FIXTURES_DIR / "routing.yaml",
}

DEFAULT_SUITE = "smoke"

# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class TestCaseResult:
    case_id: str
    prompt: str
    passed: bool
    domain: str
    u_score: float
    latency_ms: float
    response_len: int
    routing_mode: str
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "passed": self.passed,
            "domain": self.domain,
            "routing_mode": self.routing_mode,
            "u_score": round(self.u_score, 4),
            "latency_ms": round(self.latency_ms, 1),
            "response_len": self.response_len,
            "failures": self.failures,
            "error": self.error,
        }


@dataclass
class SuiteReport:
    suite: str
    router_url: str
    run_at: float
    total: int
    passed: int
    failed: int
    errored: int
    mean_u_score: float
    mean_latency_ms: float
    pass_rate: float
    cases: list[TestCaseResult]

    @property
    def ok(self) -> bool:
        return self.pass_rate == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "router_url": self.router_url,
            "run_at": self.run_at,
            "run_at_human": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.run_at)),
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errored": self.errored,
                "pass_rate": round(self.pass_rate, 3),
                "mean_u_score": round(self.mean_u_score, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
            },
            "cases": [c.to_dict() for c in self.cases],
        }


# ── Property checker (reuses eval.py logic, independent copy for clarity) ─────


def _check_property(prop: dict[str, Any], response: str, domain: str) -> tuple[bool, str]:
    """Return (passed, failure_reason)."""
    if "contains" in prop:
        if prop["contains"].lower() not in response.lower():
            return False, f"missing '{prop['contains']}'"

    if "contains_any" in prop:
        if not any(s.lower() in response.lower() for s in prop["contains_any"]):
            return False, f"none of {prop['contains_any']} found"

    if "not_contains" in prop:
        if prop["not_contains"].lower() in response.lower():
            return False, f"unexpectedly contains '{prop['not_contains']}'"

    if "min_length" in prop:
        if len(response) < prop["min_length"]:
            return False, f"response too short ({len(response)} < {prop['min_length']})"

    if "expected_domain" in prop:
        if domain != prop["expected_domain"]:
            return False, f"domain={domain!r}, expected={prop['expected_domain']!r}"

    if "expected_domain_any" in prop:
        if domain not in prop["expected_domain_any"]:
            return False, f"domain={domain!r}, expected one of {prop['expected_domain_any']}"

    return True, "ok"


# ── Liveness check ────────────────────────────────────────────────────────────


def check_router_live(router_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """
    Return (is_live, message).
    Hits GET /health/live — no auth required.
    """
    url = f"{router_url}/health/live"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, "ok"
            return False, f"GET /health/live returned HTTP {resp.status}"
    except urllib.error.URLError as e:
        return False, f"Router unreachable at {router_url}: {e.reason}"
    except Exception as e:
        return False, f"Health check failed: {e}"


# ── Single case runner ────────────────────────────────────────────────────────


def run_case(
    case: dict[str, Any],
    router_url: str = "http://localhost:8000",
    timeout: float = 120.0,
) -> TestCaseResult:
    case_id = case["id"]
    prompt = case["prompt"]
    props = case.get("expected_properties", [])

    t0 = time.time()
    try:
        payload = json.dumps({"query": prompt}).encode()
        req = urllib.request.Request(
            f"{router_url}/query",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        return TestCaseResult(
            case_id=case_id,
            prompt=prompt,
            passed=False,
            domain="unknown",
            u_score=0.0,
            latency_ms=(time.time() - t0) * 1000,
            response_len=0,
            routing_mode="unknown",
            error=str(e),
        )

    latency_ms = (time.time() - t0) * 1000
    response = data.get("response", "")
    domain = data.get("primary_domain", "unknown")
    u_score = float(data.get("u_score", 0.0))
    routing_mode = data.get("routing_mode", "unknown")

    failures = []
    for prop in props:
        passed, reason = _check_property(prop, response, domain)
        if not passed:
            failures.append(reason)

    return TestCaseResult(
        case_id=case_id,
        prompt=prompt,
        passed=len(failures) == 0,
        domain=domain,
        u_score=u_score,
        latency_ms=latency_ms,
        response_len=len(response),
        routing_mode=routing_mode,
        failures=failures,
    )


# ── Suite runner ──────────────────────────────────────────────────────────────


def run_suite(
    suite: str = DEFAULT_SUITE,
    router_url: str = "http://localhost:8000",
    timeout: float = 120.0,
    case_ids: list[str] | None = None,
) -> SuiteReport:
    """
    Run a built-in test suite against a live router.

    Args:
        suite:      'smoke' | 'full' | 'routing'
        router_url: base URL of the running AUA router
        timeout:    per-case timeout in seconds
        case_ids:   optional list of case IDs to run (default: all)

    Returns:
        SuiteReport with per-case results and aggregate stats.

    Raises:
        ValueError: if suite name is not recognised
        RuntimeError: if the router is not reachable (caller should check
                      check_router_live() first for a nicer error message)
    """
    if suite not in SUITES:
        raise ValueError(
            f"Unknown suite '{suite}'. Available: {sorted(SUITES)}. "
            "Use --suite smoke|full|routing or --dataset for a custom file."
        )

    fixture_path = SUITES[suite]
    raw = yaml.safe_load(fixture_path.read_text())
    cases_raw = raw.get("cases", [])

    if case_ids:
        cases_raw = [c for c in cases_raw if c["id"] in case_ids]

    results: list[TestCaseResult] = []
    for case_raw in cases_raw:
        result = run_case(case_raw, router_url=router_url, timeout=timeout)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error is None)
    errored = sum(1 for r in results if r.error is not None)
    u_scores = [r.u_score for r in results if r.error is None]
    latencies = [r.latency_ms for r in results if r.error is None]

    return SuiteReport(
        suite=suite,
        router_url=router_url,
        run_at=time.time(),
        total=len(results),
        passed=passed,
        failed=failed,
        errored=errored,
        mean_u_score=sum(u_scores) / len(u_scores) if u_scores else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        pass_rate=passed / len(results) if results else 0.0,
        cases=results,
    )


def run_custom_dataset(
    dataset_path: str | Path,
    router_url: str = "http://localhost:8000",
    timeout: float = 120.0,
) -> SuiteReport:
    """
    Run a user-supplied YAML dataset through the test harness.
    Same format as the built-in fixtures.
    """
    path = Path(dataset_path)
    raw = yaml.safe_load(path.read_text())
    suite_name = raw.get("name", path.stem)
    cases_raw = raw.get("cases", [])

    results: list[TestCaseResult] = []
    for case_raw in cases_raw:
        result = run_case(case_raw, router_url=router_url, timeout=timeout)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error is None)
    errored = sum(1 for r in results if r.error is not None)
    u_scores = [r.u_score for r in results if r.error is None]
    latencies = [r.latency_ms for r in results if r.error is None]

    return SuiteReport(
        suite=suite_name,
        router_url=router_url,
        run_at=time.time(),
        total=len(results),
        passed=passed,
        failed=failed,
        errored=errored,
        mean_u_score=sum(u_scores) / len(u_scores) if u_scores else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        pass_rate=passed / len(results) if results else 0.0,
        cases=results,
    )
