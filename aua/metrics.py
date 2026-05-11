"""
aua/metrics.py — Prometheus metrics for AUA Framework.

Exposes GET /metrics (Prometheus text format) with all standard AUA metrics.
Uses prometheus_client if available; returns a stub if not installed.

Install: pip install prometheus-client  (or pip install aua[otel])

Metrics emitted:
    aua_queries_total{domain, routing_mode, status}
    aua_query_latency_seconds{domain, routing_mode}
    aua_utility_score{domain}              # gauge: last U score
    aua_contradiction_rate{domain}         # gauge
    aua_routing_field_distribution{field}  # counter
    aua_specialist_confidence{specialist}  # gauge
    aua_bluegreen_traffic_split{specialist, variant}
    aua_correction_count{domain}           # counter
    aua_abstention_rate{domain}            # gauge
    aua_arbiter_verdict_distribution{case} # counter
    aua_dpo_pairs_accumulated              # gauge
    aua_token_requests_total{scope, status}
    aua_token_requests_rejected{scope}
    aua_plugin_execution_seconds{plugin, kind}
    aua_hook_failures_total{hook_point}
    aua_specialist_vram_utilization{specialist}  # gauge (0-1)

Cost metrics (GET /metrics/cost):
    aua_cost_gpu_hours_total{specialist}
    aua_cost_usd_total{specialist}
    aua_query_cost_usd{domain}  # gauge: cost per query

Configuration:
    cost:
      gpu_hour_rates:
        single-4090: 0.50
        a100-cluster: 2.50
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Try importing prometheus_client
try:
    from prometheus_client import (  # noqa: F401
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    log.info(
        "prometheus_client not installed — /metrics will return a stub. "
        "Install with: pip install prometheus-client"
    )


class AUAMetrics:
    """
    Central metrics registry for AUA Framework.

    If prometheus_client is not installed, all record_* methods are no-ops
    and /metrics returns a minimal stub response.
    """

    def __init__(self) -> None:
        self._available = _PROMETHEUS_AVAILABLE
        self._cost_config: dict[str, float] = {}  # tier → USD/hour

        if not self._available:
            # In-memory counters for stub /metrics response
            self._counts: dict[str, int] = {}
            self._gauges: dict[str, float] = {}  # type annotation explicit
            return

        # ── Prometheus metrics ────────────────────────────────────────────
        self.queries_total = Counter(
            "aua_queries_total",
            "Total queries processed",
            ["domain", "routing_mode", "status"],
        )
        self.query_latency = Histogram(
            "aua_query_latency_seconds",
            "Query latency in seconds",
            ["domain", "routing_mode"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )
        self.utility_score = Gauge(
            "aua_utility_score",
            "Last utility score for domain",
            ["domain"],
        )
        self.contradiction_rate = Gauge(
            "aua_contradiction_rate",
            "Contradiction rate for domain (fraction of queries)",
            ["domain"],
        )
        self.routing_distribution = Counter(
            "aua_routing_field_distribution",
            "Routing events per field",
            ["field"],
        )
        self.specialist_confidence = Gauge(
            "aua_specialist_confidence",
            "Kalman-filtered confidence score",
            ["specialist"],
        )
        self.correction_count = Counter(
            "aua_correction_count",
            "Corrections stored",
            ["domain"],
        )
        self.arbiter_verdict = Counter(
            "aua_arbiter_verdict_distribution",
            "Arbiter verdict by case",
            ["case"],
        )
        self.dpo_pairs = Gauge("aua_dpo_pairs_accumulated", "Total DPO pairs in store")
        self.token_requests = Counter(
            "aua_token_requests_total",
            "API requests by scope and status",
            ["scope", "status"],
        )
        self.hook_failures = Counter(
            "aua_hook_failures_total",
            "Hook failures by hook point",
            ["hook_point"],
        )
        self.plugin_latency = Histogram(
            "aua_plugin_execution_seconds",
            "Plugin execution time",
            ["plugin", "kind"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        )
        self.vram_utilization = Gauge(
            "aua_specialist_vram_utilization",
            "VRAM utilization fraction (0-1)",
            ["specialist"],
        )
        # Cost metrics
        self.cost_gpu_hours = Counter(
            "aua_cost_gpu_hours_total",
            "Accumulated GPU-hours by specialist",
            ["specialist"],
        )
        self.cost_usd = Counter(
            "aua_cost_usd_total",
            "Accumulated USD cost by specialist",
            ["specialist"],
        )

    def configure_cost(self, gpu_hour_rates: dict[str, float]) -> None:
        """Set GPU hour rates: {tier: usd_per_hour}"""
        self._cost_config.update(gpu_hour_rates)

    def record_query(
        self,
        domain: str,
        routing_mode: str,
        latency_s: float,
        u_score: float,
        status: str = "ok",
        specialist: str | None = None,
        contradictions: int = 0,
    ) -> None:
        """Record a completed query."""
        if not self._available:
            self._counts[f"queries.{domain}"] = self._counts.get(f"queries.{domain}", 0) + 1
            self._gauges[f"u_score.{domain}"] = float(u_score)
            return

        self.queries_total.labels(domain=domain, routing_mode=routing_mode, status=status).inc()
        self.query_latency.labels(domain=domain, routing_mode=routing_mode).observe(latency_s)
        self.utility_score.labels(domain=domain).set(u_score)
        self.routing_distribution.labels(field=domain).inc()
        if specialist:
            self.specialist_confidence.labels(specialist=specialist).set(u_score)

    def record_correction(self, domain: str) -> None:
        if not self._available:
            return
        self.correction_count.labels(domain=domain).inc()

    def record_arbiter_verdict(self, case: str) -> None:
        if not self._available:
            return
        self.arbiter_verdict.labels(case=case).inc()

    def record_hook_failure(self, hook_point: str) -> None:
        if not self._available:
            return
        self.hook_failures.labels(hook_point=hook_point).inc()

    def record_token_request(self, scope: str, status: str = "ok") -> None:
        if not self._available:
            return
        self.token_requests.labels(scope=scope, status=status).inc()

    def get_prometheus_output(self) -> tuple[str, str]:
        """
        Return (content, content_type) for the /metrics endpoint.

        If prometheus_client is not installed, returns a minimal stub.
        """
        if not self._available:
            stub = (
                "# AUA Framework metrics stub\n"
                "# Install prometheus-client to enable full metrics:\n"
                "#   pip install prometheus-client\n"
                f"# Queries processed: {sum(v for k, v in self._counts.items() if 'queries' in k)}\n"
            )
            for k, v in self._counts.items():
                stub += f"aua_{k.replace('.', '_')} {v}\n"
            for k, v_raw in self._gauges.items():
                stub += f"aua_{k.replace('.', '_')} {float(v_raw):.4f}\n"
            return stub, "text/plain; version=0.0.4"

        return generate_latest().decode(), CONTENT_TYPE_LATEST

    def get_cost_summary(self, config: Any | None = None) -> dict[str, Any]:
        """Return cost summary for GET /metrics/cost."""
        gpu_rates = self._cost_config
        if config:
            cost_cfg = getattr(config, "cost", None)
            if cost_cfg:
                gpu_rates = getattr(cost_cfg, "gpu_hour_rates", gpu_rates)

        if not self._available:
            return {
                "status": "stub",
                "message": "Install prometheus-client for full cost tracking",
                "gpu_hour_rates": gpu_rates,
                "total_cost_usd": 0.0,
            }

        return {
            "gpu_hour_rates": gpu_rates,
            "note": "Cost accumulation requires GPU utilization data from specialists",
            "total_cost_usd": 0.0,  # requires live specialist VRAM data
        }


# Global metrics instance
_metrics: AUAMetrics | None = None


def get_metrics() -> AUAMetrics:
    global _metrics
    if _metrics is None:
        _metrics = AUAMetrics()
    return _metrics
