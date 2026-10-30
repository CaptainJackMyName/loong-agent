"""Tests for Agent.run / Agent.run_streamed and the three-tier event model."""

from __future__ import annotations

import json

import pytest

from loong_agent import (
    Agent,
    AgentUpdatedStreamEvent,
    LoongAgentOptions,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    StreamEvent,
)
from loong_agent.llm.base import ChatResponse, LLMProvider
from loong_agent.llm.responses_converter import ChatCompletionToResponsesConverter
from loong_agent.llm.streaming import chunk_from_chat_response
from loong_agent.tools import tool
from loong_agent.types.messages import ToolCall, Usage


@tool(name="echo", description="Echo the given text back.")
async def echo(text: str) -> str:
    return f"echo:{text}"


class ScriptedLLM(LLMProvider):
    """Provider yielding one canned ChatResponse per call via the native chunk path."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get_model_name(self) -> str:
        return "mock-model"

    async def chat_completion(self, messages, tools=None, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response

    async def chat_completion_stream(self, messages, tools=None, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if response.has_tool_calls() or not response.text:
            yield chunk_from_chat_response(response)
            return
        # Split text into fine-grained deltas, then a terminal chunk.
        for part in (response.text[:2], response.text[2:]):
            if part:
                delta_chunk = chunk_from_chat_response(
                    ChatResponse(text=part, model=response.model)
                )
                delta_chunk.choices[0].finish_reason = None
                delta_chunk.usage = None
                yield delta_chunk
        terminal = ChatResponse(
            text="",
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason or "stop",
        )
        yield chunk_from_chat_response(terminal)


def _scripted_agent(responses, **agent_kwargs):
    return Agent(llm=ScriptedLLM(responses), **agent_kwargs)


async def _collect(streamed) -> list:
    events = []
    async for event in streamed.stream_events():
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_stream_event_types_match_openai_agents_model():
    agent = _scripted_agent(
        [
            ChatResponse(
                text="calling",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                usage=Usage(total_tokens=5),
                model="mock-model",
            ),
            ChatResponse(text="done", usage=Usage(total_tokens=3), model="mock-model"),
        ],
        name="worker",
        tools=[echo],
    )
    streamed = agent.run_streamed("hello", options=LoongAgentOptions(max_turns=5))
    events = await _collect(streamed)

    assert events, "stream must yield events"
    assert isinstance(events[0], AgentUpdatedStreamEvent)
    assert events[0].new_agent.name == "worker"

    for event in events:
        assert isinstance(event, StreamEvent.__args__) or event.type in {
            "raw_response_event",
            "run_item_stream_event",
            "agent_updated_stream_event",
        }
        assert event.run_id  # unified metadata envelope

    raw_events = [e for e in events if isinstance(e, RawResponsesStreamEvent)]
    assert raw_events, "raw native chunks must be forwarded"
    assert all(e.data.type.startswith("response.") for e in raw_events)

    names = [e.name for e in events if isinstance(e, RunItemStreamEvent)]
    # message_output_created is only emitted for the final, user-facing
    # message — the intermediate (tool-calling) turn produces no message item.
    assert names == [
        "tool_called",
        "tool_output",
        "message_output_created",
    ]

    assert streamed.is_complete
    assert streamed.final_output == "done"
    assert streamed.status == "completed"
    assert streamed.current_turn == 2
    assert streamed.usage.total_tokens == 8


def _split_raw_turns(raw_events):
    """Group raw events into per-turn blocks delimited by ``response.created``."""
    turns: list[list] = []
    current: list = []
    for event in raw_events:
        if event.data.type == "response.created":
            if current:
                turns.append(current)
            current = [event]
        else:
            current.append(event)
    if current:
        turns.append(current)
    return turns


@pytest.mark.asyncio
async def test_raw_events_carry_responses_stream_semantics():
    """Raw events mirror the openai-agents Responses event vocabulary."""
    agent = _scripted_agent(
        [
            ChatResponse(
                text="calling",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
                usage=Usage(total_tokens=5),
                model="mock-model",
            ),
            ChatResponse(text="done", usage=Usage(total_tokens=3), model="mock-model"),
        ],
        name="worker",
        tools=[echo],
    )
    streamed = agent.run_streamed("hello", options=LoongAgentOptions(max_turns=5))
    events = await _collect(streamed)

    raw = [e for e in events if isinstance(e, RawResponsesStreamEvent)]
    assert raw, "raw events must be forwarded"

    # Every turn starts with response.created and ends with response.completed.
    turns = _split_raw_turns(raw)
    assert len(turns) == 2
    for turn_events in turns:
        types = [e.data.type for e in turn_events]
        assert types[0] == "response.created"
        assert types[-1] == "response.completed"
        # Per-turn event sequence numbers are strictly monotonic.
        seqs = [e.data.sequence_number for e in turn_events]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    # Tool-call turn: function call identity + streamed arguments delta.
    arg_deltas = [
        e.data
        for e in turns[0]
        if e.data.type == "response.function_call_arguments.delta"
    ]
    assert arg_deltas
    assert json.loads("".join(e.delta for e in arg_deltas)) == {"text": "hi"}

    # Final turn: full text is reconstructable from output_text.delta events.
    text_deltas = [
        e.data for e in turns[1] if e.data.type == "response.output_text.delta"
    ]
    assert "".join(e.delta for e in text_deltas) == "done"

    # Terminal responses carry usage and the assembled output items.
    completed = [e.data for e in raw if e.data.type == "response.completed"]
    assert len(completed) == 2
    assert completed[0].response.usage.total_tokens == 5
    assert completed[1].response.usage.total_tokens == 3
    first_output_types = [item.type for item in completed[0].response.output]
    assert "function_call" in first_output_types
    assert "message" in first_output_types


def test_converter_maps_fallback_chunk_and_reasoning():
    """Non-streaming fallback chunk yields a full sequence; reasoning maps to deltas."""
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

    # Fallback path: one synthesized chunk from a non-streaming response.
    converter = ChatCompletionToResponsesConverter()
    response = ChatResponse(
        text="hi",
        tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "yo"})],
        usage=Usage(total_tokens=7),
        model="mock-model",
    )
    events = converter.add(chunk_from_chat_response(response)) + converter.finish()
    types = [e.type for e in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert "response.output_text.delta" in types
    assert "response.function_call_arguments.delta" in types
    assert events[-1].response.usage.total_tokens == 7
    output_types = [item.type for item in events[-1].response.output]
    assert "message" in output_types
    assert "function_call" in output_types

    # Vendor reasoning_content deltas map onto reasoning summary text deltas.
    converter = ChatCompletionToResponsesConverter()
    reasoning_delta = ChoiceDelta(role="assistant", content="")
    reasoning_delta.reasoning_content = "thinking..."  # type: ignore[attr-defined]
    text_delta = ChoiceDelta(role="assistant", content="answer")
    chunk_a = ChatCompletionChunk(
        id="chunk-1",
        choices=[Choice(delta=reasoning_delta, index=0)],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
    )
    chunk_b = ChatCompletionChunk(
        id="chunk-1",
        choices=[Choice(delta=text_delta, index=0)],
        created=0,
        model="mock-model",
        object="chat.completion.chunk",
    )
    events = (
        converter.add(chunk_a)
        + converter.add(chunk_b)
        + converter.finish()
    )
    types = [e.type for e in events]
    assert "response.reasoning_summary_text.delta" in types
    assert "response.output_text.delta" in types
    completed = events[-1]
    output_types = [item.type for item in completed.response.output]
    assert output_types[0] == "reasoning"
    assert "message" in output_types


@pytest.mark.asyncio
async def test_run_and_run_streamed_produce_consistent_results():
    responses = [
        ChatResponse(
            text="thinking",
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            usage=Usage(total_tokens=5),
            model="mock-model",
        ),
        ChatResponse(text="done", usage=Usage(total_tokens=3), model="mock-model"),
    ]
    agent = _scripted_agent(list(responses), name="worker", tools=[echo])
    options = LoongAgentOptions(max_turns=5)

    result = await agent.run("hello", options=options)
    assert result.final_output == "done"
    assert result.status == "completed"
    assert result.last_agent is agent
    assert len(result.raw_responses) == 2
    assert {item.type for item in result.new_items} >= {
        "message_output_item",
        "tool_call_item",
        "tool_call_output_item",
    }

    streamed_agent = _scripted_agent(list(responses), name="worker", tools=[echo])
    streamed = streamed_agent.run_streamed("hello", options=options)
    await _collect(streamed)

    assert streamed.final_output == result.final_output
    assert streamed.usage.total_tokens == result.usage.total_tokens
    assert [i.type for i in streamed.new_items] == [i.type for i in result.new_items]


@pytest.mark.asyncio
async def test_handoff_emits_agent_update_and_switches_context():
    agent_b = Agent(name="b", instructions="Agent B instructions")
    agent_a = _scripted_agent(
        [
            ChatResponse(
                text="",
                tool_calls=[ToolCall(id="h1", name="transfer_to_b", arguments={})],
                model="mock-model",
            ),
            ChatResponse(text="B done", model="mock-model"),
        ],
        name="a",
        handoffs=[agent_b],
    )

    streamed = agent_a.run_streamed("go", options=LoongAgentOptions(max_turns=5))
    events = await _collect(streamed)

    names = [e.name for e in events if isinstance(e, RunItemStreamEvent)]
    assert "handoff_requested" in names
    assert "handoff_occurred" in names
    assert "tool_called" not in names  # handoff is not a regular tool call

    updates = [e for e in events if isinstance(e, AgentUpdatedStreamEvent)]
    assert len(updates) == 2
    assert updates[0].new_agent.name == "a"
    assert updates[1].new_agent.name == "b"

    assert streamed.final_output == "B done"
    assert streamed.current_agent is agent_b
    assert streamed.status == "completed"

    # Non-streaming parity
    agent_a2 = _scripted_agent(
        [
            ChatResponse(
                text="",
                tool_calls=[ToolCall(id="h1", name="transfer_to_b", arguments={})],
                model="mock-model",
            ),
            ChatResponse(text="B done", model="mock-model"),
        ],
        name="a",
        handoffs=[agent_b],
    )
    result = await agent_a2.run("go", options=LoongAgentOptions(max_turns=5))
    assert result.final_output == "B done"
    assert result.last_agent is agent_b


@pytest.mark.asyncio
async def test_final_message_emitted_only_after_loop_exit():
    """The intermediate (tool-calling) turn must NOT emit message_output_created;
    the single user-facing message item appears only after the tools ran."""
    order: list[str] = []

    @tool(name="slow_echo", description="Slow echo tool.")
    async def slow_echo(text: str) -> str:
        order.append("tool_executed")
        return text

    agent = _scripted_agent(
        [
            ChatResponse(
                text="",
                tool_calls=[ToolCall(id="c1", name="slow_echo", arguments={"text": "x"})],
                model="mock-model",
            ),
            ChatResponse(text="done", model="mock-model"),
        ],
        name="worker",
        tools=[slow_echo],
    )

    streamed = agent.run_streamed("go", options=LoongAgentOptions(max_turns=5))
    message_count = 0
    async for event in streamed.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if event.name == "message_output_created":
                order.append("message_emitted")
                message_count += 1
            elif event.name == "tool_called":
                order.append("tool_called")

    # Tool execution happens inside the intermediate turn; the only
    # message_output_created arrives afterwards, exactly once.
    assert message_count == 1
    assert order == ["tool_called", "tool_executed", "message_emitted"]


@pytest.mark.asyncio
async def test_early_break_cancels_background_run():
    from loong_agent.types.messages import Usage

    class SlowLLM(LLMProvider):
        def get_model_name(self):
            return "mock-model"

        async def chat_completion(self, messages, tools=None, **kwargs):
            import asyncio

            await asyncio.sleep(10)
            return ChatResponse(text="done", usage=Usage(), model="mock-model")

    agent = Agent(name="slow", llm=SlowLLM())
    streamed = agent.run_streamed("go", options=LoongAgentOptions(max_turns=5))
    events = streamed.stream_events()
    async for _event in events:
        break  # exit early — does NOT implicitly cancel (matches openai-agents)
    await events.aclose()

    assert streamed._run_task.cancelled() or streamed._run_task.done()
    assert streamed.is_complete
