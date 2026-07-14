"""Test isolation: clear observability env vars to prevent accidental remote tracing."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clear_telemetry_env() -> None:
    """Remove LangSmith + OpenTelemetry env vars during tests."""
    for key in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_SERVICE_NAME",
        "OTEL_RESOURCE_ATTRIBUTES",
    ):
        os.environ.pop(key, None)
