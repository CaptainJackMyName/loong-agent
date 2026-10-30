"""Assigns monotonic sequence numbers and run identity to stream events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai.types.responses import ResponseStreamEvent

from loong_agent.types.events import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunItemStreamEvent,
    RunItemStreamEventName,
)

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.agent import Agent


class EventSequencer:
    """Factory enriching events with ``run_id``/``sequence``/``turn_index``."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.turn_index: int = 0
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def raw(self, data: ResponseStreamEvent) -> RawResponsesStreamEvent:
        return RawResponsesStreamEvent(
            data=data,
            run_id=self.run_id,
            sequence=self._next_sequence(),
            turn_index=self.turn_index,
        )

    def item(
        self, name: RunItemStreamEventName, item: Any
    ) -> RunItemStreamEvent:
        return RunItemStreamEvent(
            name=name,
            item=item,
            run_id=self.run_id,
            sequence=self._next_sequence(),
            turn_index=self.turn_index,
        )

    def agent_updated(self, agent: "Agent") -> AgentUpdatedStreamEvent:
        return AgentUpdatedStreamEvent(
            new_agent=agent,
            run_id=self.run_id,
            sequence=self._next_sequence(),
            turn_index=self.turn_index,
        )


__all__ = ["EventSequencer"]
