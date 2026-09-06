"""Span exporter configuration helpers."""

from __future__ import annotations

from typing import Any, Optional


def configure_tracing(
    exporter: Optional[Any] = None,
    service_name: str = "loong",
) -> Any:
    """Configure the global tracer provider.

    By default installs a ``ConsoleSpanExporter`` for local development. Pass a
    custom exporter (e.g. an OTLP exporter) for production setups.

    Returns the configured ``TracerProvider`` (or ``None`` if OpenTelemetry is
    unavailable).
    """
    try:  # pragma: no cover - depends on optional runtime
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if exporter is None:
            exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider
    except Exception:  # noqa: BLE001
        return None


__all__ = ["configure_tracing"]
