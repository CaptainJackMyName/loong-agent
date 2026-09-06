# MCP 网关

MCP（Model Context Protocol）将 Agent 连接到外部工具和数据源。SDK 内置网关支持四种传输。

## 支持的传输

| 类型 | 说明 |
|------|------|
| `stdio` | 本地子进程（换行分隔 JSON-RPC） |
| `http` | 远程 HTTP（JSON-RPC POST） |
| `sse` | HTTP + Server-Sent Events（流式响应） |
| `inprocess` | 进程内对象直接调用 |

## 配置示例

```python
from loong_agent import LoongAgentOptions

options = LoongAgentOptions(
    llm=provider,
    mcp_servers={
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
        },
        "github": {
            "type": "http",
            "url": "https://api.github.com/mcp",
            "headers": {"Authorization": "Bearer ..."},
        },
        "remote": {
            "type": "sse",
            "url": "https://example.com/mcp/sse",
        },
    },
    allowed_tools=["mcp__filesystem__*", "mcp__github__*", "mcp__remote__*"],
)
```

MCP 工具以 `mcp__<server>__<tool>` 命名，通过 `allowed_tools` 过滤。

## 直接使用网关

```python
from loong_agent.mcp import MCPGateway
from loong_agent.types.options import MCPServerConfig

gateway = MCPGateway()
await gateway.connect("filesystem", MCPServerConfig(type="stdio", command="npx", args=["-y", "..."]))
tools = await gateway.list_tools("filesystem")
result = await gateway.call_tool("filesystem", "read_file", {"path": "/tmp/a.txt"})
await gateway.close_all()
```

## SSE 传输

SSE 传输实现了 MCP "Streamable HTTP" 规范：

- 请求通过 `POST` 发送，响应可以是 JSON 或 `text/event-stream`。
- 后台 `GET` SSE 流接收服务器主动推送的消息（通知）。
- 支持旧式 `endpoint` 事件动态切换消息端点。

```python
from loong_agent.mcp import SseTransport

transport = SseTransport(
    url="https://example.com/mcp",
    headers={"Authorization": "Bearer ..."},
)
```
