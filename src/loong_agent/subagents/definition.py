"""Subagent definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentDefinition:
    """Definition of a subagent.

    Attributes:
        description: Human-readable purpose, used by the parent agent to decide
            when to delegate to this subagent.
        system_prompt: Dedicated system prompt for the subagent.
        tools: Tool names the subagent is allowed to use (empty = inherit).
        model: Optional model override for the subagent.
        parallel: Whether this subagent supports parallel execution.
    """

    description: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    model: Optional[str] = None
    parallel: bool = False


__all__ = ["AgentDefinition"]
