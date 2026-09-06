"""MCP gateway — manages connections to multiple MCP servers."""

from __future__ import annotations

from typing import Any, Dict, List

from loong.mcp.client import MCPClient, MCPTool
from loong.mcp.transports.base import Transport
from loong.mcp.transports.http import HttpTransport
from loong.mcp.transports.sse import SseTransport
from loong.mcp.transports.stdio import StdioTransport
from loong.types.options import MCPServerConfig


class MCPGateway:
    def __init__(self) -> None:
        self._clients: Dict[str, MCPClient] = {}

    async def connect(self, name: str, config: MCPServerConfig) -> None:
        if name in self._clients:
            return
        transport = self._build_transport(config)
        client = MCPClient(transport)
        await client.initialize()
        self._clients[name] = client

    def _build_transport(self, config: MCPServerConfig) -> Transport:
        if config.type == "stdio":
            if not config.command:
                raise ValueError("stdio MCP servers require a 'command'.")
            return StdioTransport(config.command, config.args, config.env)
        if config.type == "http":
            if not config.url:
                raise ValueError("http MCP servers require a 'url'.")
            return HttpTransport(config.url, config.headers)
        if config.type == "sse":
            if not config.url:
                raise ValueError("sse MCP servers require a 'url'.")
            return SseTransport(config.url, config.headers)
        if config.type == "inprocess":
            return _InProcessTransport(config.server)
        raise ValueError(f"Unsupported MCP transport type: {config.type}")

    async def disconnect(self, name: str) -> None:
        client = self._clients.pop(name, None)
        if client is not None:
            await client.close()

    async def list_tools(self, server: str) -> List[MCPTool]:
        client = self._clients.get(server)
        if client is None:
            raise KeyError(f"Unknown MCP server: {server}")
        return await client.list_tools()

    async def call_tool(self, server: str, tool: str, args: Dict[str, Any]) -> Any:
        client = self._clients.get(server)
        if client is None:
            raise KeyError(f"Unknown MCP server: {server}")
        return await client.call_tool(tool, args)

    async def close_all(self) -> None:
        for name in list(self._clients.keys()):
            await self.disconnect(name)


class _InProcessTransport(Transport):
    """Runs an in-process MCP server object exposing the same call surface."""

    def __init__(self, server: Any) -> None:
        self.server = server

    async def request(self, method, params, request_id):
        if method == "initialize":
            return {}
        if method == "tools/list":
            tools = getattr(self.server, "list_tools", lambda: [])()
            return {"tools": tools}
        if method == "tools/call":
            handler = getattr(self.server, "call_tool", None)
            if handler is None:
                raise RuntimeError("In-process server does not implement call_tool")
            return await handler(params.get("name"), params.get("arguments", {}))
        raise RuntimeError(f"Unsupported method: {method}")

    async def notify(self, method, params):
        return None

    async def close(self):
        close = getattr(self.server, "close", None)
        if close is not None:
            result = close()
            import inspect

            if inspect.isawaitable(result):
                await result


__all__ = ["MCPGateway"]
