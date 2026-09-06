"""End-to-end ReAct loop test using a mock LLM provider."""

from __future__ import annotations

import pytest

from loong_agent import LoongAgentOptions, query
from loong_agent.llm.base import ChatResponse, LLMProvider
from loong_agent.types.messages import ToolCall, Usage


class MockLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    def get_model_name(self) -> str:
        return "mock-model"

    async def chat_completion(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                text="listing",
                tool_calls=[ToolCall(id="c1", name="Glob", arguments={"pattern": "*.py"})],
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        return ChatResponse(
            text="done",
            usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            model="mock-model",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_react_loop_runs_tools_and_terminates():
    options = LoongAgentOptions(llm=MockLLM(), allowed_tools=["Glob"], max_turns=5)
    messages = [m async for m in query("list files", options=options)]

    assert messages[0].type == "system"
    assert any(m.type == "assistant" and m.has_tool_calls() for m in messages)
    assert any(m.type == "user" and m.tool_results for m in messages)
    assert messages[-1].type == "result"
    assert not messages[-1].is_error


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_as_error():
    class BadLLM(MockLLM):
        async def chat_completion(self, messages, tools=None, **kwargs):
            return ChatResponse(
                text="x",
                tool_calls=[ToolCall(id="c1", name="Nope", arguments={})],
                usage=Usage(),
                model="mock-model",
            )

    options = LoongAgentOptions(llm=BadLLM(), allowed_tools=["*"], max_turns=2)
    messages = [m async for m in query("go", options=options)]
    user_msg = next(m for m in messages if m.type == "user")
    assert user_msg.tool_results[0].is_error
