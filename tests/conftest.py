"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from loong_agent.llm.base import ChatResponse, LLMProvider
from loong_agent.types.messages import Usage

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class MockLLM(LLMProvider):
    """An LLM stub returning canned responses in sequence."""

    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.calls = 0

    def get_model_name(self) -> str:
        return "mock-model"

    async def chat_completion(self, messages, tools=None, **kwargs):
        idx = min(self.calls, len(self.responses) - 1) if self.responses else 0
        self.calls += 1
        if self.responses:
            return self.responses[idx]
        return ChatResponse(text="done", usage=Usage(), model="mock-model", finish_reason="stop")


@pytest.fixture
def make_llm():
    def _factory(responses=None):
        return MockLLM(responses)

    return _factory
