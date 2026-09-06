"""Public client and convenience entrypoints."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator

from loong_agent.loop.engine import AgentLoop
from loong_agent.types.messages import Message
from loong_agent.types.options import LoongAgentOptions


async def query(
    prompt: str,
    options: LoongAgentOptions | None = None,
) -> AsyncIterator[Message]:
    """Stream the execution of an agent task, yielding each message in turn.

    Example::

        async for message in query("Summarize src/", options=options):
            print(message.type, message.text)
    """
    if options is None:
        options = LoongAgentOptions()
    loop = AgentLoop(options)
    try:
        async for message in loop.run(prompt):
            yield message
    finally:
        await loop.aclose()


class LoongAgentClient:
    """Stateful client supporting send/receive streaming and interrupts."""

    def __init__(self, options: LoongAgentOptions | None = None) -> None:
        self.options = options or LoongAgentOptions()
        self.loop = AgentLoop(self.options)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def send(self, prompt: str) -> None:
        """Start processing ``prompt``; results are read via :meth:`receive`."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("A request is already in progress.")
        self._task = asyncio.create_task(self._run(prompt))

    async def _run(self, prompt: str) -> None:
        try:
            async for message in self.loop.run(prompt):
                await self._queue.put(message)
        except Exception as exc:  # noqa: BLE001
            await self._queue.put(
                Message.result(text=f"Error: {exc}", is_error=True)
            )
        finally:
            await self._queue.put(_SENTINEL)

    async def receive(self) -> AsyncIterator[Message]:
        """Yield messages produced by the current request until completion."""
        while True:
            message = await self._queue.get()
            if message is _SENTINEL:
                break
            yield message

    async def interrupt(self) -> None:
        """Cancel the in-flight request."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._queue.put(_SENTINEL)

    async def close(self) -> None:
        await self.loop.aclose()


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


__all__ = ["query", "LoongAgentClient"]
