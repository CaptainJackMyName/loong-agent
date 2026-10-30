"""Run items — complete, semantically meaningful execution artifacts.

Items are the payload of :class:`loong_agent.types.events.RunItemStreamEvent`.
Unlike raw model deltas, an item is only emitted once it is fully resolved
(finished message, parsed tool call, executed tool output, completed handoff).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

from loong_agent.types.messages import Message, ToolCall, ToolResult

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.agent import Agent


@dataclass
class MessageOutputItem:
    """The final, user-facing assistant message produced when the agent exits
    the ReAct loop (final response, follow-up question or refusal).

    Intermediate turns — assistant messages that carry tool calls — never
    produce this item; their text is only available through the raw delta
    stream.
    """

    agent: "Agent"
    message: Message
    type: Literal["message_output_item"] = "message_output_item"


@dataclass
class ReasoningItem:
    """A complete reasoning block returned by the model (when available)."""

    agent: "Agent"
    text: str
    type: Literal["reasoning_item"] = "reasoning_item"


@dataclass
class ToolCallItem:
    """A fully parsed tool invocation requested by the model (pre-execution)."""

    agent: "Agent"
    tool_call: ToolCall
    type: Literal["tool_call_item"] = "tool_call_item"


@dataclass
class ToolCallOutputItem:
    """The executed outcome of a single tool call."""

    agent: "Agent"
    tool_call: ToolCall
    output: ToolResult
    type: Literal["tool_call_output_item"] = "tool_call_output_item"


@dataclass
class HandoffCallItem:
    """The model requested a transfer to another agent."""

    agent: "Agent"
    tool_call: ToolCall
    type: Literal["handoff_call_item"] = "handoff_call_item"


@dataclass
class HandoffOutputItem:
    """The transfer was accepted and recorded; the run switches agents."""

    agent: "Agent"
    tool_call: ToolCall
    output: Optional[ToolResult] = None
    type: Literal["handoff_output_item"] = "handoff_output_item"


RunItem = Union[
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
    HandoffCallItem,
    HandoffOutputItem,
]

__all__ = [
    "MessageOutputItem",
    "ReasoningItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    "RunItem",
]
