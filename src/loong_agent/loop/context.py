"""Session context management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from loong_agent.types.messages import Message, Usage
from loong_agent.types.options import LoongAgentOptions


@dataclass
class SessionContext:
    """Mutable state shared across the turns of a single agent run."""

    options: LoongAgentOptions
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Message] = field(default_factory=list)
    turn_count: int = 0
    total_usage: Usage = field(default_factory=Usage)

    @property
    def model_name(self) -> Optional[str]:
        return getattr(self.options.llm, "get_model_name", lambda: None)()

    def append(self, message: Message) -> None:
        self.messages.append(message)
        if message.usage is not None:
            self.total_usage = self.total_usage.merge(message.usage)

    def to_llm_messages(self) -> List[Message]:
        """Messages that should be forwarded to the LLM (excludes terminal results)."""
        return [m for m in self.messages if m.type != "result"]


__all__ = ["SessionContext"]
