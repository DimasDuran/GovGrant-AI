"""Observability: LangSmith tracing + OpenTelemetry (Grafana Cloud).

Usage:
    from govgrant.core.telemetry import setup_telemetry, langsmith_client

    setup_telemetry()  # call once at startup before any agent run
    client = langsmith_client()  # for evals, prompt hub, etc.

LangSmith auto-tracing:
  LangGraph's ``app.invoke()`` is traced automatically when ``langsmith`` is
  installed and ``LANGSMITH_TRACING=true`` is set. Raw Anthropic calls in
  ``ChatLLM`` are also traced via ``@traceable``.

OpenTelemetry (Grafana Cloud):
  When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, the OTel SDK initialises and
  instruments httpx, logging, and system metrics. No-op otherwise.
"""

from __future__ import annotations

import os
from urllib.parse import urljoin

_LANGSMITH_READY: bool = False
_OTEL_READY: bool = False


def _parse_headers(raw: str | None) -> dict[str, str]:
    """Parse ``OTEL_EXPORTER_OTLP_HEADERS`` into a dict."""
    headers: dict[str, str] = {}
    if not raw:
        return headers
    for part in raw.split(","):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            headers[key.strip()] = val.strip()
    return headers


def setup_langsmith() -> bool:
    """Initialise LangSmith auto-tracing.

    Returns ``True`` if tracing is active, ``False`` if env vars are missing.
    Safe to call multiple times (idempotent).
    """
    global _LANGSMITH_READY
    if _LANGSMITH_READY:
        return True

    api_key = os.getenv("LANGSMITH_API_KEY")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")

    if not api_key or not tracing:
        _LANGSMITH_READY = False
        return False

    try:
        import langsmith  # noqa: F401 — auto-registers LangChain tracer

        _LANGSMITH_READY = True
        return True
    except ImportError:
        _LANGSMITH_READY = False
        return False


def setup_otel() -> bool:
    """Initialise OpenTelemetry SDK with OTLP HTTP exporter.

    Instruments:
      - httpx (HTTP client calls — Anthropic, SBIR.gov, Ollama, Qdrant)
      - logging (trace correlation in logs)
      - system metrics (CPU, memory)

    Returns ``True`` if OTel is active, ``False`` if no endpoint configured.
    Safe to call multiple times (idempotent).
    """
    global _OTEL_READY
    if _OTEL_READY:
        return True

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        _OTEL_READY = False
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.instrumentation.system_metrics import (
            SystemMetricsInstrumentor,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        trace_url = urljoin(endpoint.rstrip("/") + "/", "v1/traces")
        raw_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")

        resource = Resource.create({
            SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "govgrant"),
        })
        extra_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if extra_attrs:
            parts = {}
            for attr in extra_attrs.split(","):
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    parts[k.strip()] = v.strip()
            resource = resource.merge(Resource.create(parts))

        provider = TracerProvider(
            resource=resource,
            active_span_processor=BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=trace_url,
                    headers=_parse_headers(raw_headers),
                ),
            ),
        )
        trace.set_tracer_provider(provider)

        LoggingInstrumentor().instrument()
        SystemMetricsInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()

        _OTEL_READY = True
        return True
    except ImportError:
        _OTEL_READY = False
        return False


def setup_telemetry() -> bool:
    """Convenience: initialise both LangSmith and OpenTelemetry.

    Returns ``True`` if at least one observability backend is active.
    """
    ls = setup_langsmith()
    ot = setup_otel()
    return ls or ot


def langsmith_enabled() -> bool:
    """Whether LangSmith tracing has been successfully initialised."""
    return _LANGSMITH_READY


def langsmith_client():
    """Return a ``langsmith.Client`` for evals / prompt hub.

    Returns ``None`` if LangSmith is not configured.
    """
    if not _LANGSMITH_READY:
        return None
    from langsmith import Client

    return Client()
