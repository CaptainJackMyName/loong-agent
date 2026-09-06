"""Hook event definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class HookEvent(str, Enum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SUBAGENT_START = "SubagentStart"
    SUBAGENT_STOP = "SubagentStop"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    IDLE = "Idle"


@dataclass
class HookContext:
    """Metadata passed to hook callbacks."""

    session_id: Optional[str] = None
    cwd: Optional[str] = None
    model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookOutput:
    """Return value of a hook callback.

    Attributes:
        permission_decision: ``"allow"``, ``"deny"`` or ``"ask"``.
        updated_input: Replacement input for the operation being hooked.
        injected_context: Additional context injected into the conversation.
        output: Optional message to surface to the caller.
        stop_reason: Optional reason string if the hook stops the session.
    """

    permission_decision: str = "allow"
    updated_input: Optional[Dict[str, Any]] = None
    injected_context: Optional[str] = None
    output: Optional[str] = None
    stop_reason: Optional[str] = None

    @classmethod
    def allow(cls, **kwargs: Any) -> "HookOutput":
        return cls(permission_decision="allow", **kwargs)

    @classmethod
    def deny(cls, reason: str = "", **kwargs: Any) -> "HookOutput":
        return cls(permission_decision="deny", output=reason or None, **kwargs)


__all__ = ["HookEvent", "HookContext", "HookOutput"]
