"""Tests for LoongAgentOptions normalization."""

from __future__ import annotations

from loong_agent.types.options import LoongAgentOptions, MCPServerConfig, PluginRef


def test_normalize_mcp_servers_from_dicts():
    options = LoongAgentOptions(
        mcp_servers={"fs": {"type": "stdio", "command": "npx", "args": ["-y", "x"]}}
    )
    normalized = options.normalize_mcp_servers()
    assert isinstance(normalized["fs"], MCPServerConfig)
    assert normalized["fs"].command == "npx"
    assert normalized["fs"].args == ["-y", "x"]


def test_normalize_mcp_servers_passthrough():
    cfg = MCPServerConfig(type="http", url="https://example.com/mcp")
    options = LoongAgentOptions(mcp_servers={"remote": cfg})
    assert options.normalize_mcp_servers()["remote"] is cfg


def test_normalize_plugins():
    options = LoongAgentOptions(plugins=[{"type": "local", "path": "./my-plugin"}])
    refs = options.normalize_plugins()
    assert isinstance(refs[0], PluginRef)
    assert refs[0].path == "./my-plugin"


def test_default_allowed_tools_is_wildcard():
    options = LoongAgentOptions()
    assert options.allowed_tools == ["*"]
