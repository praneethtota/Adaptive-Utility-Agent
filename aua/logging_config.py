"""
aua/logging_config.py — Structured JSON logging for AUA Framework.

Configures Python's logging to emit JSON with all required fields:
  timestamp, level, message, session_id, trace_id, request_id,
  token_id, field, specialist, utility_score, confidence,
  contradiction, verdict, latency_ms, plugin, hook, middleware.

Configuration:
    logging:
      level: INFO        # DEBUG | INFO | WARNING | ERROR
      format: json       # "json" (default) | "text" (human-readable)
      output: stdout     # "stdout" (default) | "/path/to/file.log"

Usage:
    from aua.logging_config import configure_logging
    configure_logging(level="INFO", format="json")

    import logging
    log = logging.getLogger("aua.router")
    log.info("Query routed", extra={"field": "swe", "latency_ms": 312.4})
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class AUAJsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON with all AUA standard fields.

    Output format:
        {"ts": 1234567890.123, "level": "INFO", "msg": "...",
         "logger": "aua.router", "session_id": "...", ...}
    """

    # Fields to promote from LogRecord's extra dict
    AUA_FIELDS = {
        "session_id",
        "trace_id",
        "request_id",
        "token_id",
        "field",
        "specialist",
        "utility_score",
        "confidence",
        "contradiction",
        "verdict",
        "latency_ms",
        "plugin",
        "hook",
        "middleware",
        "domain",
        "routing_mode",
        "error_code",
    }

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": record.created,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Pull AUA-specific fields from extra
        for field in self.AUA_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                doc[field] = val

        # Include exception info if present
        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)

        # Try to inject current session context automatically
        try:
            from aua.session import get_current_or_none

            ctx = get_current_or_none()
            if ctx:
                doc.setdefault("session_id", ctx.session_id)
                doc.setdefault("trace_id", ctx.trace_id)
                doc.setdefault("request_id", ctx.request_id)
        except Exception:
            pass

        return json.dumps(doc, default=str)


class AUATextFormatter(logging.Formatter):
    """Human-readable log formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        base = f"{ts} [{record.levelname:7s}] {record.name}: {record.getMessage()}"

        extras = []
        for field in AUAJsonFormatter.AUA_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                extras.append(f"{field}={val}")

        if extras:
            base += f"  ({', '.join(extras)})"

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    output: str = "stdout",
) -> None:
    """
    Configure AUA structured logging.

    Args:
        level:  log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: "json" for structured JSON, "text" for human-readable
        output: "stdout", "stderr", or a file path
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if format == "json":
        formatter: logging.Formatter = AUAJsonFormatter()
    else:
        formatter = AUATextFormatter()

    if output in ("stdout", "-"):
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    elif output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(output, mode="a", encoding="utf-8")

    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    # Configure the root aua logger
    aua_logger = logging.getLogger("aua")
    aua_logger.setLevel(numeric_level)
    aua_logger.handlers.clear()
    aua_logger.addHandler(handler)
    aua_logger.propagate = False  # don't bubble to root

    # Suppress noisy third-party loggers at WARNING+
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
