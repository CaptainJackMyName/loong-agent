"""LLM provider abstraction.

Any model that exposes an OpenAI-compatible chat completion endpoint (or can be
wrapped to do so) plugs into the SDK through :class:`LLMProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loong_agent.types.messages import Message, ToolCall, Usage


@dataclass
class ChatResponse:
    """Normalized response returned by an :class:`LLMProvider`."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw: Any = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolDefinition:
    """A tool description exposed to the LLM as a callable function."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Base interface every LLM adapter must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Produce the next assistant turn given the conversation."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical model name."""

    def clone(self, model: Optional[str] = None) -> "LLMProvider":
        """Return a provider configured for a different model (default: self)."""
        return self


__all__ = ["LLMProvider", "ChatResponse", "ToolDefinition"]
