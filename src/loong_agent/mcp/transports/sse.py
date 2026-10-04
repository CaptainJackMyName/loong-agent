"""MCP SSE transport (HTTP + Server-Sent Events).

Implements the MCP "Streamable HTTP" transport: JSON-RPC requests are sent over
``POST`` and responses may arrive either as a JSON body or as an SSE stream. A
background ``GET`` SSE stream handles server-initiated messages (notifications).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from loong_agent.mcp.transports.base import Transport

logger = logging.getLogger(__name__)


def parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, str]]:
    """Parse a stream of lines into SSE events.

    Each yielded event is a dict of field name -> value (``data`` values are
    joined with newlines per the SSE specification).
    """

    async def _events() -> AsyncIterator[Dict[str, str]]:
        current: Dict[str, str] = {}
        data_lines: list[str] = []
        async for raw in lines:
            line = raw.rstrip("\n").rstrip("\r")
            if line == "":
                if current or data_lines:
                    if data_lines:
                        current["data"] = "\n".join(data_lines)
                    yield current
                    current = {}
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            value = value[1:] if sep and value.startswith(" ") else value
            if field == "data":
                data_lines.append(value)
            else:
                current[field] = value
        if current or data_lines:
            if data_lines:
                current["data"] = "\n".join(data_lines)
            yield current

    return _events()


class SseTransport(Transport):
    """JSON-RPC over HTTP with SSE response handling."""

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
        sse_url: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.sse_url = (sse_url or self.url).rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=timeout)
        )
        self._owns_client = client is None
        self._sse_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._started = False

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._sse_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        headers = {**self.headers, "Accept": "text/event-stream"}
        try:
            async with self._client.stream("GET", self.sse_url, headers=headers) as response:
                response.raise_for_status()
                async for event in parse_sse(response.aiter_lines()):
                    await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("SSE listener terminated: %s", exc)

    async def _handle_event(self, event: Dict[str, str]) -> None:
        if event.get("event") == "endpoint":
            endpoint = event.get("data", "").strip()
            if endpoint:
                self.url = endpoint
            return
        data = event.get("data", "").strip()
        if not data:
            return
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            return
        if "id" in message:
            future = self._pending.pop(message["id"], None)
            if future and not future.done():
                future.set_result(message)

    # -- Transport interface -------------------------------------------------
    async def request(
        self, method: str, params: Optional[Dict[str, Any]], request_id: int
    ) -> Dict[str, Any]:
        await self.start()
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        headers = {
            **self.headers,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        async with self._client.stream("POST", self.url, json=payload, headers=headers) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                async for event in parse_sse(response.aiter_lines()):
                    await self._handle_event(event)
                    data = event.get("data", "").strip()
                    if not data:
                        continue
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if message.get("id") == request_id:
                        if "error" in message:
                            raise RuntimeError(f"MCP error: {message['error']}")
                        return message.get("result", {})
                raise RuntimeError("SSE stream closed without a response.")

            data = response.json()
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error']}")
            return data.get("result", {})

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        await self.start()
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        headers = {**self.headers, "Content-Type": "application/json"}
        await self._client.post(self.url, json=payload, headers=headers)

    async def close(self) -> None:
        if self._sse_task is not None:
            self._sse_task.cancel()
            self._sse_task = None
        if self._owns_client:
            await self._client.aclose()
        self._started = False


__all__ = ["SseTransport", "parse_sse"]
