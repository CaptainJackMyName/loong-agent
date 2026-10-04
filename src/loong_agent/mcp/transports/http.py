"""MCP HTTP transport (JSON-RPC over HTTP POST)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from loong_agent.mcp.transports.base import Transport

logger = logging.getLogger(__name__)


class HttpTransport(Transport):
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.headers = headers or {}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=timeout),
            headers={**self.headers, "Content-Type": "application/json"},
        )

    async def request(
        self, method: str, params: Optional[Dict[str, Any]], request_id: int
    ) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        response = await self._client.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._client.post(self.url, json=payload)

    async def close(self) -> None:
        await self._client.aclose()


__all__ = ["HttpTransport"]
