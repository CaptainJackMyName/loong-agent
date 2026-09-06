"""OpenTelemetry tracer wrapper with graceful degradation."""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator, Dict, Optional

try:  # pragma: no cover - depends on optional runtime
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Span, Tracer, TracerProvider

    _HAS_OTEL = True
except Exception:  # noqa: BLE001
    otel_trace = None  # type: ignore[assignment]
    _HAS_OTEL = False


class _NoopSpan:
    def __init__(self) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: Any = None) -> None:
        pass

    def set_status(self, status: Any, description: Any = None) -> None:
        pass

    def end(self) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass


class LoongAgentTracer:
    """Thin, dependency-safe facade over an OpenTelemetry ``Tracer``."""

    SERVICE_NAME = "loong"

    def __init__(self, tracer: Optional[Any] = None) -> None:
        if tracer is not None:
            self._tracer = tracer
        elif _HAS_OTEL:
            self._tracer = otel_trace.get_tracer(self.SERVICE_NAME)
        else:
            self._tracer = None

    @property
    def available(self) -> bool:
        return self._tracer is not None

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        if self._tracer is None:
            return _NoopSpan()
        return self._tracer.start_as_current_span(name, attributes=attributes)

    @contextlib.asynccontextmanager
    async def span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> AsyncIterator[Any]:
        """Async context manager yielding an active span-like object."""
        if self._tracer is None:
            yield _NoopSpan()
            return
        with self._tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span

    def set_attribute(self, key: str, value: Any) -> None:
        if _HAS_OTEL and otel_trace is not None:
            span = otel_trace.get_current_span()
            if span is not None:
                span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        if _HAS_OTEL and otel_trace is not None:
            span = otel_trace.get_current_span()
            if span is not None:
                span.add_event(name, attributes=attributes)


# Module-level singleton used by the SDK internals.
_default_tracer = LoongAgentTracer()


def get_tracer() -> LoongAgentTracer:
    return _default_tracer


__all__ = ["LoongAgentTracer", "get_tracer", "_NoopSpan"]
