"""Run results for streaming and non-streaming executions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

from loong_agent.llm.base import ChatResponse
from loong_agent.types.events import StreamEvent
from loong_agent.types.items import RunItem
from loong_agent.types.messages import Message, Usage

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.agent import Agent
    from loong_agent.loop.engine import AgentLoop


@dataclass
class RunResultBase:
    """Shared final-state fields for both run styles."""

    input: str = ""
    final_output: Optional[str] = None
    new_items: List[RunItem] = field(default_factory=list)
    # Final, user-facing assistant messages (one per ReAct-loop exit).
    # Intermediate assistant messages carrying tool calls are not included —
    # their text is only available via the raw delta stream.
    messages: List[Message] = field(default_factory=list)
    raw_responses: List[ChatResponse] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    last_agent: Optional["Agent"] = None
    # "completed" | "failed" | "cancelled" | "max_turns"
    status: str = "completed"
    error: Optional[BaseException] = None


@dataclass
class RunResult(RunResultBase):
    """Final result of ``await agent.run(...)``."""


class _QueueCompleteSentinel:
    pass


_QUEUE_COMPLETE = _QueueCompleteSentinel()


@dataclass
class RunResultStreaming(RunResultBase):
    """Mutable handle for a streaming run.

    The background run task fills this object while the caller consumes
    ``stream_events()``. A run is complete only when the event iterator
    finishes; afterwards the object holds the final snapshot.
    """

    current_agent: Optional["Agent"] = None
    current_turn: int = 0
    max_turns: int = 50
    is_complete: bool = False

    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)
    _consumed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _run_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _exception: Optional[BaseException] = field(default=None, repr=False)
    _loop: Optional["AgentLoop"] = field(default=None, repr=False)

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Consume the run's events until completion.

        Raises the background exception (if any) once the stream ends. If the
        consumer exits early, the background run is cancelled to avoid leaks.
        """
        exhausted = False
        try:
            while True:
                event = await self._event_queue.get()
                if event is _QUEUE_COMPLETE:
                    exhausted = True
                    break
                # Acknowledge consumption so the producer may continue; this
                # guarantees consumers observe each event in production order.
                self._consumed.set()
                yield event
            if self._exception is not None:
                raise self._exception
        finally:
            if not exhausted:
                self.cancel(mode="immediate")
            if self._run_task is not None and not self._run_task.done():
                try:
                    await self._run_task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 - surfaced via _exception/status
                    pass

    def cancel(self, mode: str = "immediate") -> None:
        """Cancel the run.

        - ``immediate``: cancel the background task now.
        - ``after_turn``: let the current turn finish, stop before the next one.
        """
        if mode == "after_turn":
            if self._loop is not None:
                self._loop.request_cancel()
            return
        if mode != "immediate":
            raise ValueError(f"Unknown cancel mode: {mode!r}")
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
        self.is_complete = True
        try:
            self._event_queue.put_nowait(_QUEUE_COMPLETE)
        except Exception:  # noqa: BLE001 - queue cannot practically reject
            pass


__all__ = ["RunResult", "RunResultStreaming"]
