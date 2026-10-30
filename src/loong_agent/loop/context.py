"""Session context management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from loong_agent.types.messages import Message, Usage
from loong_agent.types.options import LoongAgentOptions

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.llm.base import LLMProvider


@dataclass
class SessionContext:
    """Mutable state shared across the turns of a single agent run."""

    options: LoongAgentOptions
    llm: Optional["LLMProvider"] = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = field(default_factory=list)
    turn_count: int = 0
    total_usage: Usage = field(default_factory=Usage)

    @property
    def model_name(self) -> Optional[str]:
        llm = self.llm or self.options.llm
        return getattr(llm, "get_model_name", lambda: None)()

    def append(self, message: Message) -> None:
        self.messages.append(message)
        # Terminal result messages carry a *snapshot* of the aggregated usage;
        # merging them again would double-count tokens.
        if message.usage is not None and message.type != "result":
            self.total_usage = self.total_usage.merge(message.usage)

    def replace_system_prompt(self, text: str) -> None:
        """Update the first system message in place (used on handoff)."""
        for message in self.messages:
            if message.type == "system":
                message.text = text
                return

    def to_llm_messages(self) -> List[Message]:
        """Messages that should be forwarded to the LLM (excludes terminal results)."""
        return [m for m in self.messages if m.type != "result"]


__all__ = ["SessionContext"]
