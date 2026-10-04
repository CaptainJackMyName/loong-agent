"""Tests for the core message types."""

from __future__ import annotations

import pytest

from loong_agent.types.messages import (
    Message,
    ToolCall,
    ToolResult,
    Usage,
    assistant_message,
    result_message,
    system_message,
    user_message,
)


def test_factory_constructors():
    sys_msg = system_message(subtype="init", session_id="abc", model="m")
    assert sys_msg.type == "system"
    assert sys_msg.session_id == "abc"

    assistant = assistant_message(
        text="hi", tool_calls=[ToolCall(id="1", name="T", arguments={"x": 1})]
    )
    assert assistant.has_tool_calls()

    user = user_message(tool_results=[ToolResult(tool_use_id="1", name="T", content="ok")])
    assert user.tool_results[0].content == "ok"

    result = result_message(text="done", usage=Usage(total_tokens=10), cost=0.5)
    assert result.type == "result"
    assert result.usage.total_tokens == 10


def test_has_tool_calls_empty():
    assert not Message.assistant(text="plain").has_tool_calls()


def test_usage_merge():
    merged = Usage(10, 5, 15, 0.1).merge(Usage(20, 3, 23, 0.2))
    assert (merged.prompt_tokens, merged.completion_tokens) == (30, 8)
    assert merged.total_tokens == 38
    assert merged.cost == pytest.approx(0.3)
