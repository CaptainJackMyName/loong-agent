"""Tests for the tracing facade."""

from __future__ import annotations

import pytest

from loong_agent._agent.tracing import LoongAgentTracer
from loong_agent._agent.tracing.spans import LOOP_TURN, LLM_REQUEST, TOOL_EXECUTE
from loong_agent._agent.tracing.tracer import _NoopSpan


def test_noop_span_methods_do_not_raise():
    span = _NoopSpan()
    span.set_attribute("k", "v")
    span.add_event("e", {"a": 1})
    span.set_status(None)
    span.record_exception(Exception("x"))
    span.end()


class _FakeSpanContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_attribute(self, key, value):
        pass


class _FakeTracer:
    def __init__(self):
        self.started = []

    def start_as_current_span(self, name, attributes=None):
        self.started.append(name)
        return _FakeSpanContext()


@pytest.mark.asyncio
async def test_span_uses_custom_tracer():
    fake = _FakeTracer()
    tracer = LoongAgentTracer(tracer=fake)

    async with tracer.span(TOOL_EXECUTE):
        pass

    assert fake.started == [TOOL_EXECUTE]


def test_span_name_constants():
    assert LOOP_TURN == "loong_agent..loop.turn"
    assert LLM_REQUEST == "loong_agent..llm.request"
    assert TOOL_EXECUTE == "loong_agent..tool.execute"


def test_tracer_reports_availability():
    tracer = LoongAgentTracer()
    # With or without OpenTelemetry, the object must be usable.
    assert tracer.available in (True, False)
