"""
aua/eval.py — Evaluation harness for AUA Framework.

Routes each eval case through the live framework, scores with the utility
function, checks expected properties, and produces a JSON report.

CLI:
    aua eval run --dataset evals/coding_smoke.yaml --config aua_config.yaml
    aua eval report .aua/evals/latest.json
    aua eval compare --baseline .aua/evals/blue.json --candidate .aua/evals/green.json

Dataset format (YAML):
    name: coding_smoke
    field: software_engineering
    description: Basic coding correctness

    cases:
      - id: binary_search
        prompt: "Write binary search in Python."
        expected_properties:
          - contains: "def binary_search"
          - contains_any: ["mid", "low", "high"]
          - min_length: 50
          - expected_domain: software_engineering
          - not_contains: "I cannot"

Property checkers:
    contains: str          response must contain this substring (case-insensitive)
    contains_any: [str]    response must contain at least one
    not_contains: str      response must NOT contain this
    min_length: int        response must be at least N chars
    expected_domain: str   primary_domain in response must match
    expected_domain_any: [str]  primary_domain must be one of these
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

# ── Property checkers ─────────────────────────────────────────────────────────


def _check_property(prop: dict[str, Any], response: str, domain: str) -> tuple[bool, str]:
    """Return (passed, reason)."""
    if "contains" in prop:
        needle = prop["contains"].lower()
        if needle not in response.lower():
            return False, f"missing '{prop['contains']}'"

    if "contains_any" in prop:
        found = any(s.lower() in response.lower() for s in prop["contains_any"])
        if not found:
            return False, f"none of {prop['contains_any']} found"

    if "not_contains" in prop:
        needle = prop["not_contains"].lower()
        if needle in response.lower():
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


# ── Case runner ───────────────────────────────────────────────────────────────


@dataclass
class CaseResult:
    case_id: str
    prompt: str
    passed: bool
    domain: str
    u_score: float
    latency_ms: float
    response_len: int
    failures: list[str] = field(default_factory=list)
    error: str | None = None


def run_case(
    case: dict[str, Any],
    router_url: str = "http://localhost:8000",
    timeout: float = 120.0,
) -> CaseResult:
    """Run a single eval case against the live router."""
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
        return CaseResult(
            case_id=case_id,
            prompt=prompt,
            passed=False,
            domain="unknown",
            u_score=0.0,
            latency_ms=(time.time() - t0) * 1000,
            response_len=0,
            error=str(e),
        )

    latency_ms = (time.time() - t0) * 1000
    response = data.get("response", "")
    domain = data.get("primary_domain", "unknown")
    u_score = float(data.get("u_score", 0.0))

    failures = []
    for prop in props:
        passed, reason = _check_property(prop, response, domain)
        if not passed:
            failures.append(reason)

    return CaseResult(
        case_id=case_id,
        prompt=prompt,
        passed=len(failures) == 0,
        domain=domain,
        u_score=u_score,
        latency_ms=latency_ms,
        response_len=len(response),
        failures=failures,
    )


# ── Dataset runner ────────────────────────────────────────────────────────────


@dataclass
class EvalReport:
    dataset_name: str
    field: str
    description: str
    run_at: float
    router_url: str
    total: int
    passed: int
    failed: int
    error: int
    mean_u_score: float
    mean_latency_ms: float
    pass_rate: float
    cases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "field": self.field,
            "description": self.description,
            "run_at": self.run_at,
            "run_at_human": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.run_at)),
            "router_url": self.router_url,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "error": self.error,
                "pass_rate": round(self.pass_rate, 3),
                "mean_u_score": round(self.mean_u_score, 4),
                "mean_latency_ms": round(self.mean_latency_ms, 1),
            },
            "cases": self.cases,
        }

    def regression_vs(self, baseline: EvalReport) -> dict[str, Any]:
        """Compare this report against a baseline. Returns regression summary."""
        delta_pass = self.pass_rate - baseline.pass_rate
        delta_u = self.mean_u_score - baseline.mean_u_score
        delta_latency = self.mean_latency_ms - baseline.mean_latency_ms
        regressed = delta_pass < -0.05 or delta_u < -0.02
        return {
            "regressed": regressed,
            "delta_pass_rate": round(delta_pass, 3),
            "delta_u_score": round(delta_u, 4),
            "delta_latency_ms": round(delta_latency, 1),
            "verdict": "REGRESSION" if regressed else "OK",
        }


def run_dataset(
    dataset_path: str | Path,
    router_url: str = "http://localhost:8000",
    timeout: float = 120.0,
) -> EvalReport:
    """Run all cases in a dataset YAML file."""
    path = Path(dataset_path)
    raw = yaml.safe_load(path.read_text())

    name = raw.get("name", path.stem)
    field_ = raw.get("field", "any")
    description = raw.get("description", "")
    cases_raw = raw.get("cases", [])

    results = []
    for case_raw in cases_raw:
        result = run_case(case_raw, router_url=router_url, timeout=timeout)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.error is None)
    errors = sum(1 for r in results if r.error is not None)
    u_scores = [r.u_score for r in results if r.error is None]
    latencies = [r.latency_ms for r in results if r.error is None]

    return EvalReport(
        dataset_name=name,
        field=field_,
        description=description,
        run_at=time.time(),
        router_url=router_url,
        total=len(results),
        passed=passed,
        failed=failed,
        error=errors,
        mean_u_score=sum(u_scores) / len(u_scores) if u_scores else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        pass_rate=passed / len(results) if results else 0.0,
        cases=[
            {
                "id": r.case_id,
                "passed": r.passed,
                "domain": r.domain,
                "u_score": round(r.u_score, 4),
                "latency_ms": round(r.latency_ms, 1),
                "response_len": r.response_len,
                "failures": r.failures,
                "error": r.error,
            }
            for r in results
        ],
    )


def save_report(report: EvalReport, output_dir: str = ".aua/evals") -> Path:
    """Save report to .aua/evals/ and create a 'latest.json' symlink."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime(report.run_at))
    fname = out / f"{report.dataset_name}_{ts}.json"
    fname.write_text(json.dumps(report.to_dict(), indent=2))
    latest = out / "latest.json"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(fname.name)
    except Exception:
        import shutil

        shutil.copy(str(fname), str(latest))
    return fname
