"""Core message types exchanged across the SDK.

These types model the ReAct turn structure::

    SystemMessage (init) -> AssistantMessage (text + tool calls)
        -> UserMessage (tool results) -> ... -> ResultMessage

Every message in a turn is represented by :class:`Message`; the factory
classmethods (:meth:`Message.system`, :meth:`Message.assistant`, etc.) and the
module-level helper constructors keep construction ergonomic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

MessageType = Literal["system", "assistant", "user", "result"]


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of executing a tool call."""

    tool_use_id: str
    name: str
    content: str = ""
    is_error: bool = False


@dataclass
class Usage:
    """Token usage and cost accounting for an LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def merge(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost,
        )


@dataclass
class Message:
    """A generic message that can represent any ReAct turn element.

    The ``type`` discriminator selects the active payload fields:

    - ``system``: session metadata via ``subtype``/``session_id``.
    - ``assistant``: model output via ``text`` and ``tool_calls``.
    - ``user``: tool execution feedback via ``tool_results``.
    - ``result``: terminal summary via ``usage`` and ``cost``.
    """

    type: MessageType
    text: str = ""

    # Assistant payload
    tool_calls: List[ToolCall] = field(default_factory=list)

    # User payload
    tool_results: List[ToolResult] = field(default_factory=list)

    # System payload
    subtype: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = None

    # Result payload
    usage: Optional[Usage] = None
    cost: Optional[float] = None

    # Error metadata (any type)
    is_error: bool = False
    error: Optional[str] = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    # -- Factory constructors -------------------------------------------------
    @classmethod
    def system(
        cls,
        subtype: str = "init",
        text: str = "",
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        **extra: Any,
    ) -> "Message":
        return cls(
            type="system",
            subtype=subtype,
            text=text,
            session_id=session_id,
            model=model,
            **extra,
        )

    @classmethod
    def assistant(
        cls,
        text: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        model: Optional[str] = None,
        **extra: Any,
    ) -> "Message":
        return cls(
            type="assistant",
            text=text,
            tool_calls=tool_calls or [],
            model=model,
            **extra,
        )

    @classmethod
    def user(
        cls,
        tool_results: Optional[List[ToolResult]] = None,
        text: str = "",
        **extra: Any,
    ) -> "Message":
        return cls(
            type="user",
            tool_results=tool_results or [],
            text=text,
            **extra,
        )

    @classmethod
    def result(
        cls,
        text: str = "",
        usage: Optional[Usage] = None,
        cost: Optional[float] = None,
        **extra: Any,
    ) -> "Message":
        return cls(type="result", text=text, usage=usage, cost=cost, **extra)

    @classmethod
    def error(cls, text: str, error: Optional[str] = None, **extra: Any) -> "Message":
        return cls(type="result", text=text, is_error=True, error=error or text, **extra)


# -- Module-level convenience constructors -----------------------------------


def system_message(
    subtype: str = "init",
    text: str = "",
    session_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Message:
    return Message.system(subtype=subtype, text=text, session_id=session_id, model=model)


def assistant_message(
    text: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    model: Optional[str] = None,
) -> Message:
    return Message.assistant(text=text, tool_calls=tool_calls, model=model)


def user_message(
    tool_results: Optional[List[ToolResult]] = None,
    text: str = "",
) -> Message:
    return Message.user(tool_results=tool_results, text=text)


def result_message(
    text: str = "",
    usage: Optional[Usage] = None,
    cost: Optional[float] = None,
) -> Message:
    return Message.result(text=text, usage=usage, cost=cost)


# Backwards-friendly aliases matching the architecture document.
SystemMessage = Message
AssistantMessage = Message
UserMessage = Message
ResultMessage = Message

__all__ = [
    "Message",
    "MessageType",
    "ToolCall",
    "ToolResult",
    "Usage",
    "SystemMessage",
    "AssistantMessage",
    "UserMessage",
    "ResultMessage",
    "system_message",
    "assistant_message",
    "user_message",
    "result_message",
]
