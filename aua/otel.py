"""
aua/otel.py — OpenTelemetry instrumentation for AUA Framework.

Optional — requires: pip install aua[otel]
  opentelemetry-sdk
  opentelemetry-instrumentation-fastapi
  opentelemetry-exporter-otlp

Configuration:
    observability:
      otel:
        enabled: false
        endpoint: http://localhost:4317   # OTEL collector gRPC endpoint
        service_name: aua-router
        headers: {}

W3C trace context:
    Trace IDs from aua.session.SessionContext are W3C-compatible (32 hex chars).
    When OTEL is enabled, each request's trace_id is used to correlate spans
    with the existing AUA session tracking.

Usage (automatic when enabled in config):
    from aua.otel import setup_otel, instrument_app
    setup_otel(config)          # called at serve startup
    instrument_app(app)         # instruments the FastAPI app
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def setup_otel(config: Any | None = None) -> bool:
    """
    Configure the OTEL SDK from AUAConfig.

    Returns True if OTEL was successfully configured, False otherwise.
    """
    if config is None:
        return False

    obs_cfg = getattr(config, "observability", None)
    if obs_cfg is None:
        return False

    otel_cfg = getattr(obs_cfg, "otel", None)
    if otel_cfg is None:
        return False

    if not getattr(otel_cfg, "enabled", False):
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "OTEL enabled in config but opentelemetry packages not installed. "
            "Install with: pip install 'adaptive-utility-agent[otel]'"
        )
        return False

    service_name = getattr(otel_cfg, "service_name", "aua-router")
    endpoint = getattr(otel_cfg, "endpoint", "http://localhost:4317")
    headers = getattr(otel_cfg, "headers", {})

    resource = Resource.create({SERVICE_NAME: service_name})
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    log.info("OTEL configured: service=%s endpoint=%s", service_name, endpoint)
    return True


def instrument_app(app: Any) -> None:
    """Apply FastAPI OTEL instrumentation to the router app."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        log.info("FastAPI OTEL instrumentation active")
    except ImportError:
        log.debug("opentelemetry-instrumentation-fastapi not installed — skipping")


def get_tracer(name: str = "aua") -> Any:
    """Return an OTEL tracer (no-op if OTEL not configured)."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpTracer:
    """Stub tracer used when OTEL is not installed."""

    def start_as_current_span(self, name: str, **kwargs: Any) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield None

        return _noop()

    def start_span(self, name: str, **kwargs: Any) -> Any:
        return _NoOpSpan()


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass
