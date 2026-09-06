"""MCP JSON-RPC client."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loong.mcp.transports.base import Transport

MCP_PROTOCOL_VERSION = "2024-11-05"


@dataclass
class MCPTool:
    """A tool exposed by an MCP server, callable from the agent."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    _client: Optional["MCPClient"] = None

    async def invoke(self, **arguments: Any) -> str:
        if self._client is None:
            return "Error: MCP tool is not bound to a client."
        return await self._client.call_tool(self.name, arguments)


class MCPClient:
    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self._request_id = 0
        self.server_info: Dict[str, Any] = {}

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def initialize(self) -> Dict[str, Any]:
        result = await self.transport.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "loong", "version": "0.1.0"},
            },
            self._next_id(),
        )
        self.server_info = result.get("serverInfo", {})
        await self.transport.notify("notifications/initialized", {})
        return result

    async def list_tools(self) -> List[MCPTool]:
        result = await self.transport.request("tools/list", {}, self._next_id())
        tools: List[MCPTool] = []
        for raw in result.get("tools", []):
            tools.append(
                MCPTool(
                    name=raw.get("name", ""),
                    description=raw.get("description", ""),
                    input_schema=raw.get("inputSchema", {}),
                    _client=self,
                )
            )
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        result = await self.transport.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            self._next_id(),
        )
        return _extract_content(result)

    async def close(self) -> None:
        await self.transport.close()


def _extract_content(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in result.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
    if result.get("isError"):
        return f"[MCP error]\n" + "\n".join(parts)
    return "\n".join(parts) if parts else json.dumps(result, ensure_ascii=False)


__all__ = ["MCPClient", "MCPTool", "MCP_PROTOCOL_VERSION"]
