"""LLM provider abstraction.

Any model that exposes an OpenAI-compatible chat completion endpoint (or can be
wrapped to do so) plugs into the SDK through :class:`LLMProvider`.

Streaming is optional: providers may override :meth:`LLMProvider.chat_completion_stream`
to emit native OpenAI ``ChatCompletionChunk`` deltas. The default implementation
falls back to :meth:`chat_completion` and synthesizes a single native chunk, so
existing custom providers and test mocks keep working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from loong_agent.types.messages import Message, ToolCall, Usage

if TYPE_CHECKING:  # pragma: no cover
    from openai.types.chat import ChatCompletionChunk


@dataclass
class ChatResponse:
    """Normalized response returned by an :class:`LLMProvider`."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    # Vendor-extended reasoning text (e.g. ``reasoning_content`` deltas).
    reasoning: str = ""
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

    async def chat_completion_stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator["ChatCompletionChunk"]:
        """Yield native OpenAI ``ChatCompletionChunk`` deltas.

        The default implementation delegates to the non-streaming
        :meth:`chat_completion` and synthesizes a single complete chunk, so
        providers without streaming support still participate in the unified
        event pipeline (without fine-grained deltas).
        """
        from loong_agent.llm.streaming import chunk_from_chat_response

        response = await self.chat_completion(messages, tools, **kwargs)
        yield chunk_from_chat_response(response)

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical model name."""

    def clone(self, model: Optional[str] = None) -> "LLMProvider":
        """Return a provider configured for a different model (default: self)."""
        return self


__all__ = ["LLMProvider", "ChatResponse", "ToolDefinition"]
