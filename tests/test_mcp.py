"""Tests for MCP client, gateway and SSE transport."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
import pytest

from loong.mcp.client import MCPClient
from loong.mcp.gateway import MCPGateway
from loong.mcp.transports.base import Transport
from loong.mcp.transports.sse import SseTransport, parse_sse
from loong.types.options import MCPServerConfig


class FakeTransport(Transport):
    def __init__(self) -> None:
        self.requests = []
        self.notifications = []

    async def request(self, method: str, params: Optional[Dict[str, Any]], request_id: int):
        self.requests.append((method, params))
        if method == "initialize":
            return {"serverInfo": {"name": "srv"}}
        if method == "tools/list":
            return {"tools": [{"name": "t1", "description": "d", "inputSchema": {"type": "object"}}]}
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "result-text"}], "isError": False}
        return {}

    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        self.notifications.append(method)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_mcp_client_initialize_list_call():
    transport = FakeTransport()
    client = MCPClient(transport)

    await client.initialize()
    assert client.server_info == {"name": "srv"}
    assert "initialize" in [m for m, _ in transport.requests]

    tools = await client.list_tools()
    assert tools[0].name == "t1"

    result = await client.call_tool("t1", {"a": 1})
    assert result == "result-text"


class InProcessServer:
    def list_tools(self):
        return [{"name": "t1", "description": "d", "inputSchema": {}}]

    async def call_tool(self, name, args):
        return {"content": [{"type": "text", "text": f"called {name}"}], "isError": False}


@pytest.mark.asyncio
async def test_gateway_inprocess():
    gateway = MCPGateway()
    await gateway.connect("srv", MCPServerConfig(type="inprocess", server=InProcessServer()))

    tools = await gateway.list_tools("srv")
    assert tools[0].name == "t1"

    result = await gateway.call_tool("srv", "t1", {})
    assert "called t1" in result

    await gateway.close_all()
    assert "srv" not in gateway._clients


@pytest.mark.asyncio
async def test_parse_sse():
    async def lines():
        yield "event: message"
        yield 'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
        yield ""
        yield "data: line1"
        yield "data: line2"
        yield ""

    events = [e async for e in parse_sse(lines())]
    assert len(events) == 2
    assert events[0]["event"] == "message"
    assert '"id":1' in events[0]["data"]
    assert events[1]["data"] == "line1\nline2"


@pytest.mark.asyncio
async def test_sse_transport_json_response():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = SseTransport("https://example.com/mcp", client=client)
    try:
        result = await transport.request("tools/list", {}, 1)
        assert result == {"ok": True}
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_sse_transport_streaming_response():
    sse_body = (
        "event: message\n"
        'data: {"jsonrpc":"2.0","id":7,"result":{"streamed":true}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200, content=sse_body.encode(), headers={"content-type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = SseTransport("https://example.com/mcp", client=client)
    try:
        result = await transport.request("tools/call", {}, 7)
        assert result == {"streamed": True}
    finally:
        await transport.close()
