"""Tests for the OpenAI-compatible provider (no network)."""

from __future__ import annotations

import pytest

from loong_agent.llm.base import ToolDefinition
from loong_agent.llm.openai_compatible import OpenAICompatibleProvider
from loong_agent.types.messages import Message, ToolCall, ToolResult


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://localhost:1/v1", api_key="k", model="m"
    )


def test_to_openai_messages(provider):
    messages = [
        Message.system(text="sys"),
        Message.user(text="hi"),
        Message.assistant(
            text="calling", tool_calls=[ToolCall(id="1", name="T", arguments={"a": 1})]
        ),
        Message.user(tool_results=[ToolResult(tool_use_id="1", name="T", content="res")]),
    ]
    out = provider._to_openai_messages(messages)

    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["role"] == "assistant"
    assert out[2]["tool_calls"][0]["function"]["name"] == "T"
    assert out[3]["role"] == "tool"
    assert out[3]["tool_call_id"] == "1"


def test_to_openai_tool(provider):
    definition = ToolDefinition(name="T", description="d", input_schema={"type": "object"})
    out = provider._to_openai_tool(definition)
    assert out["type"] == "function"
    assert out["function"]["name"] == "T"


def test_parse_response_with_tool_calls(provider):
    data = {
        "choices": [
            {
                "message": {
                    "content": "hello",
                    "tool_calls": [
                        {"id": "1", "function": {"name": "T", "arguments": '{"a": 1}'}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "model": "m",
    }
    resp = provider._parse_response(data)
    assert resp.text == "hello"
    assert resp.tool_calls[0].name == "T"
    assert resp.tool_calls[0].arguments == {"a": 1}
    assert resp.usage.total_tokens == 3


def test_parse_response_invalid_arguments_falls_back(provider):
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "function": {"name": "T", "arguments": "not-json"}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {},
    }
    resp = provider._parse_response(data)
    assert resp.tool_calls[0].arguments == {"_raw": "not-json"}


def test_clone_preserves_config(provider):
    clone = provider.clone(model="m2")
    assert clone.model == "m2"
    assert clone.base_url == provider.base_url
    assert clone.api_key == provider.api_key


def test_get_model_name(provider):
    assert provider.get_model_name() == "m"
