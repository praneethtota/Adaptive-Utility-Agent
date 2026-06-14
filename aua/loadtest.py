"""
aua/loadtest.py — Built-in load testing engine (#50).

Fires concurrent POST /query requests against a live router and reports
p50/p95/p99 latency, throughput, and error rate.

Usage (programmatic):
    from aua.loadtest import LoadTestConfig, run_loadtest
    cfg = LoadTestConfig(router_url="http://localhost:8000", concurrency=10,
                         duration_s=30, suite="smoke")
    report = asyncio.run(run_loadtest(cfg))
    print(report.to_dict())

CLI:
    aua loadtest
    aua loadtest --concurrency 20 --duration 60 --suite full
    aua loadtest --dataset my_queries.yaml --concurrency 5
    aua loadtest --url http://prod:8000 --json
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class LoadTestConfig:
    router_url: str = "http://localhost:8000"
    concurrency: int = 10  # simultaneous in-flight requests
    duration_s: float = 30.0  # wall-clock duration to run
    ramp_s: float = 0.0  # ramp-up: linearly increase workers over N seconds
    suite: str = "smoke"  # built-in fixture suite (or ignored if dataset set)
    dataset: str | None = None  # path to custom YAML query file (overrides suite)
    timeout_s: float = 60.0  # per-request timeout
    think_ms: float = 0.0  # ms between each worker's requests (0 = fire continuously)


# ── Per-request result ────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    latency_ms: float
    status: int  # HTTP status or 0 on network error
    routing_mode: str
    primary_domain: str
    u_score: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class LoadTestReport:
    config: LoadTestConfig
    started_at: float
    finished_at: float
    total_requests: int
    ok_requests: int
    error_requests: int
    latencies_ms: list[float]
    error_rate: float
    throughput_rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    mean_u_score: float
    routing_mode_counts: dict[str, int]
    domain_counts: dict[str, int]
    errors: list[str]  # up to 20 distinct error messages

    @property
    def ok(self) -> bool:
        return self.error_rate < 0.05  # < 5% errors = passing

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "router_url": self.config.router_url,
                "concurrency": self.config.concurrency,
                "duration_s": self.config.duration_s,
                "suite": self.config.suite,
                "dataset": self.config.dataset,
            },
            "run": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "wall_time_s": round(self.finished_at - self.started_at, 2),
            },
            "summary": {
                "total_requests": self.total_requests,
                "ok_requests": self.ok_requests,
                "error_requests": self.error_requests,
                "error_rate": round(self.error_rate, 4),
                "throughput_rps": round(self.throughput_rps, 2),
            },
            "latency": {
                "p50_ms": round(self.p50_ms, 1),
                "p95_ms": round(self.p95_ms, 1),
                "p99_ms": round(self.p99_ms, 1),
                "mean_ms": round(self.mean_ms, 1),
                "min_ms": round(self.min_ms, 1),
                "max_ms": round(self.max_ms, 1),
            },
            "quality": {
                "mean_u_score": round(self.mean_u_score, 4),
                "routing_mode_counts": self.routing_mode_counts,
                "domain_counts": self.domain_counts,
            },
            "errors": self.errors[:20],
            "ok": self.ok,
        }


# ── Query loading ─────────────────────────────────────────────────────────────


def _load_queries(config: LoadTestConfig) -> list[str]:
    """
    Return the list of query strings to sample from during the load test.

    Priority:
      1. config.dataset — user-supplied YAML (same format as aua test fixtures)
      2. config.suite   — built-in fixture (smoke / full / routing)
    """
    if config.dataset:
        path = Path(config.dataset)
        raw = yaml.safe_load(path.read_text())
        cases = raw.get("cases", [])
        return [c["prompt"] for c in cases if c.get("prompt")]

    # Built-in suites
    from aua.test_harness import SUITES

    if config.suite not in SUITES:
        raise ValueError(f"Unknown suite '{config.suite}'. Available: {sorted(SUITES)}.")
    raw = yaml.safe_load(SUITES[config.suite].read_text())
    return [c["prompt"] for c in raw.get("cases", [])]


# ── Core engine ───────────────────────────────────────────────────────────────


async def _worker(
    worker_id: int,
    queries: list[str],
    router_url: str,
    deadline: float,
    timeout_s: float,
    think_s: float,
    results: list[RequestResult],
    client: httpx.AsyncClient,
) -> None:
    """Single worker coroutine — fires requests until deadline."""
    idx = worker_id  # start at a different query per worker to spread the mix
    while time.monotonic() < deadline:
        query = queries[idx % len(queries)]
        idx += 1
        t0 = time.monotonic()
        try:
            resp = await client.post(
                f"{router_url}/query",
                json={"query": query},
                timeout=timeout_s,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            try:
                data = resp.json()
            except Exception:
                data = {}
            results.append(
                RequestResult(
                    latency_ms=latency_ms,
                    status=resp.status_code,
                    routing_mode=data.get("routing_mode", "unknown"),
                    primary_domain=data.get("primary_domain", "unknown"),
                    u_score=float(data.get("u_score", 0.0)),
                    error=None if resp.status_code < 400 else f"HTTP {resp.status_code}",
                )
            )
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000
            results.append(
                RequestResult(
                    latency_ms=latency_ms,
                    status=0,
                    routing_mode="unknown",
                    primary_domain="unknown",
                    u_score=0.0,
                    error=str(e)[:120],
                )
            )

        if think_s > 0:
            await asyncio.sleep(think_s)


async def run_loadtest(config: LoadTestConfig) -> LoadTestReport:
    """
    Run the load test and return a LoadTestReport.

    Args:
        config: LoadTestConfig controlling all parameters.

    Returns:
        LoadTestReport with full latency distribution and quality metrics.

    Raises:
        ValueError: if suite name is unknown or dataset file not found.
        httpx.ConnectError: if the router is unreachable at startup.
    """
    queries = _load_queries(config)
    if not queries:
        raise ValueError("No queries found in dataset/suite.")

    results: list[RequestResult] = []
    started_at = time.time()
    deadline = time.monotonic() + config.duration_s
    think_s = config.think_ms / 1000.0

    async with httpx.AsyncClient(timeout=config.timeout_s) as client:
        if config.ramp_s > 0:
            # Ramp: launch workers one at a time over ramp_s seconds
            ramp_interval = config.ramp_s / max(config.concurrency, 1)
            tasks = []
            for i in range(config.concurrency):
                await asyncio.sleep(ramp_interval)
                tasks.append(
                    asyncio.create_task(
                        _worker(
                            i,
                            queries,
                            config.router_url,
                            deadline,
                            config.timeout_s,
                            think_s,
                            results,
                            client,
                        )
                    )
                )
            await asyncio.gather(*tasks)
        else:
            # Full concurrency from the start
            await asyncio.gather(
                *[
                    _worker(
                        i,
                        queries,
                        config.router_url,
                        deadline,
                        config.timeout_s,
                        think_s,
                        results,
                        client,
                    )
                    for i in range(config.concurrency)
                ]
            )

    finished_at = time.time()
    wall_s = finished_at - started_at

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = len(results)
    ok_results = [r for r in results if r.ok]
    err_results = [r for r in results if not r.ok]
    latencies = [r.latency_ms for r in results]

    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    routing_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for r in ok_results:
        routing_counts[r.routing_mode] = routing_counts.get(r.routing_mode, 0) + 1
        domain_counts[r.primary_domain] = domain_counts.get(r.primary_domain, 0) + 1

    distinct_errors = list(dict.fromkeys(r.error for r in err_results if r.error))

    u_scores = [r.u_score for r in ok_results]

    return LoadTestReport(
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        total_requests=total,
        ok_requests=len(ok_results),
        error_requests=len(err_results),
        latencies_ms=latencies,
        error_rate=len(err_results) / total if total else 0.0,
        throughput_rps=total / wall_s if wall_s > 0 else 0.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        min_ms=min(latencies) if latencies else 0.0,
        max_ms=max(latencies) if latencies else 0.0,
        mean_u_score=statistics.mean(u_scores) if u_scores else 0.0,
        routing_mode_counts=routing_counts,
        domain_counts=domain_counts,
        errors=distinct_errors[:20],
    )
