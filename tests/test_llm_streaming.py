"""Tests for native OpenAI chunk aggregation, SSE parsing and fallback."""

from __future__ import annotations

import json

import httpx
import pytest
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from loong_agent.llm.base import ChatResponse
from loong_agent.llm.openai_compatible import OpenAICompatibleProvider
from loong_agent.llm.streaming import ChatCompletionAggregator, chunk_from_chat_response
from loong_agent.types.messages import ToolCall, Usage


def _chunk(
    content=None,
    tool_call=None,
    finish_reason=None,
    usage=None,
    index=0,
):
    delta = ChoiceDelta(content=content)
    if tool_call is not None:
        delta = ChoiceDelta(
            content=content,
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=tool_call["index"],
                    id=tool_call.get("id"),
                    function=ChoiceDeltaToolCallFunction(
                        name=tool_call.get("name"),
                        arguments=tool_call.get("arguments"),
                    ),
                )
            ],
        )
    return ChatCompletionChunk(
        id="chatcmpl-1",
        choices=[Choice(delta=delta, finish_reason=finish_reason, index=index)],
        created=1700000000,
        model="gpt-test",
        object="chat.completion.chunk",
        usage=usage,
    )


def test_aggregator_assembles_text_and_fragmented_tool_calls():
    agg = ChatCompletionAggregator()
    agg.add(_chunk(content="Hel"))
    agg.add(_chunk(content="lo"))
    agg.add(_chunk(tool_call={"index": 0, "id": "c1", "name": "search", "arguments": '{"qu'}))
    agg.add(_chunk(tool_call={"index": 0, "arguments": 'ery": "ai"}'}))
    agg.add(_chunk(tool_call={"index": 1, "id": "c2", "name": "read", "arguments": "{}"}))
    agg.add(
        _chunk(
            finish_reason="tool_calls",
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
    )

    response = agg.finish()
    assert response.text == "Hello"
    assert response.finish_reason == "tool_calls"
    assert response.usage.total_tokens == 10
    assert response.reasoning == ""
    assert [tc.id for tc in response.tool_calls] == ["c1", "c2"]
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].arguments == {"query": "ai"}
    assert response.tool_calls[1].arguments == {}


def test_aggregator_invalid_tool_arguments_fall_back_to_raw():
    agg = ChatCompletionAggregator()
    agg.add(_chunk(tool_call={"index": 0, "id": "c1", "name": "x", "arguments": "{oops"}))
    response = agg.finish()
    assert response.tool_calls[0].arguments == {"_raw": "{oops"}


def test_aggregator_collects_reasoning_content():
    agg = ChatCompletionAggregator()
    chunk = _chunk(content=None)
    chunk.choices[0].delta.reasoning_content = "thinking hard"
    agg.add(chunk)
    response = agg.finish()
    assert response.reasoning == "thinking hard"


def test_fallback_chunk_roundtrip_from_non_streaming_response():
    response = ChatResponse(
        text="hi",
        tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model="mock-model",
        finish_reason="tool_calls",
    )
    agg = ChatCompletionAggregator()
    agg.add(chunk_from_chat_response(response))
    rebuilt = agg.finish()

    assert rebuilt.text == "hi"
    assert rebuilt.finish_reason == "tool_calls"
    assert rebuilt.usage.total_tokens == 3
    assert len(rebuilt.tool_calls) == 1
    assert rebuilt.tool_calls[0].name == "echo"
    assert rebuilt.tool_calls[0].arguments == {"text": "a"}


def _sse_response(chunks) -> httpx.Response:
    lines = [f"data: {c.model_dump_json()}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    body = "".join(lines)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_openai_compatible_provider_streams_native_chunks():
    chunks = [
        _chunk(content="He"),
        _chunk(content="y"),
        _chunk(
            finish_reason="stop",
            usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        return _sse_response(chunks)

    provider = OpenAICompatibleProvider(
        base_url="http://test/v1", api_key="k", model="gpt-test"
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    collected = []
    async for chunk in provider.chat_completion_stream(messages=[]):
        collected.append(chunk)
    await provider.aclose()

    assert len(collected) == 3
    agg = ChatCompletionAggregator()
    for chunk in collected:
        agg.add(chunk)
    response = agg.finish()
    assert response.text == "Hey"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 4


@pytest.mark.asyncio
async def test_openai_compatible_provider_skips_malformed_lines():
    good = _chunk(content="ok")

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            ": keep-alive comment\n\n"
            "data: not-json\n\n"
            f"data: {good.model_dump_json()}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    provider = OpenAICompatibleProvider(
        base_url="http://test/v1", api_key="k", model="gpt-test"
    )
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    collected = [c async for c in provider.chat_completion_stream(messages=[])]
    await provider.aclose()
    assert len(collected) == 1
    agg = ChatCompletionAggregator()
    agg.add(collected[0])
    assert agg.finish().text == "ok"
