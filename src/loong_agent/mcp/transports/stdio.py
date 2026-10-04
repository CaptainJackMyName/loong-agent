"""MCP stdio transport (local subprocess, newline-delimited JSON-RPC)."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from loong_agent.mcp.transports.base import Transport


class StdioTransport(Transport):
    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        full_env = {**os.environ, **self.env}
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in message:
                future = self._pending.pop(message["id"], None)
                if future and not future.done():
                    future.set_result(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionError("MCP stdio server closed the connection"))
        self._pending.clear()

    async def request(
        self, method: str, params: Optional[Dict[str, Any]], request_id: int
    ) -> Dict[str, Any]:
        await self.start()
        assert self._proc and self._proc.stdin
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        await self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        response = await future
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response.get("result", {})

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        await self.start()
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, message: Dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
        if self._reader_task is not None:
            self._reader_task.cancel()
        self._started = False


__all__ = ["StdioTransport"]
