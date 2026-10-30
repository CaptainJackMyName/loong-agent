"""Streaming event model, modeled after the openai-agents SDK.

The public stream is exactly one of three event kinds::

    StreamEvent = RawResponsesStreamEvent | RunItemStreamEvent | AgentUpdatedStreamEvent

- ``RawResponsesStreamEvent``: the native OpenAI streaming result, converted to
  Responses API semantics (``openai.types.responses.ResponseStreamEvent``) and
  forwarded as-is, for token-level consumption.
- ``RunItemStreamEvent``: a complete, semantic run item from this SDK layer
  (final message output, tool called, tool output, handoff, reasoning).
  ``message_output_created`` is only emitted when the agent exits the ReAct
  loop with a user-facing message — never for intermediate assistant messages
  that carry tool calls.
- ``AgentUpdatedStreamEvent``: the starting point of an agent's work — emitted
  once when the run begins and again after each successful handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

from openai.types.responses import ResponseStreamEvent

from loong_agent.types.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ReasoningItem,
    RunItem,
    ToolCallItem,
    ToolCallOutputItem,
)

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.agent import Agent


RunItemStreamEventName = Literal[
    "message_output_created",
    "reasoning_item_created",
    "tool_called",
    "tool_output",
    "handoff_requested",
    "handoff_occurred",
]


@dataclass
class RawResponsesStreamEvent:
    """A Responses API streaming event forwarded from the model adapter.

    The underlying provider streams Chat Completions chunks; they are converted
    to ``openai.types.responses.ResponseStreamEvent`` (mirroring the
    openai-agents SDK event vocabulary) before being wrapped here, so consumers
    can rely on ``data.type`` values like ``response.output_text.delta``.
    """

    data: ResponseStreamEvent
    run_id: str = ""
    sequence: int = 0
    turn_index: int = 0
    type: Literal["raw_response_event"] = "raw_response_event"


@dataclass
class RunItemStreamEvent:
    """A complete run item has been generated (semantic, not delta-level)."""

    name: RunItemStreamEventName
    item: RunItem
    run_id: str = ""
    sequence: int = 0
    turn_index: int = 0
    type: Literal["run_item_stream_event"] = "run_item_stream_event"


@dataclass
class AgentUpdatedStreamEvent:
    """A new agent starts working (run start or post-handoff continuation)."""

    new_agent: "Agent"
    run_id: str = ""
    sequence: int = 0
    turn_index: int = 0
    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


StreamEvent: TypeAlias = (
    RawResponsesStreamEvent
    | RunItemStreamEvent
    | AgentUpdatedStreamEvent
)

__all__ = [
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "AgentUpdatedStreamEvent",
    "StreamEvent",
    "RunItemStreamEventName",
    "MessageOutputItem",
    "ReasoningItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "RunItem",
]
