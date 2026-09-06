# API 参考

## 顶层入口

### `query`

```python
async def query(prompt: str, options: LoongAgentOptions = None) -> AsyncIterator[Message]
```

流式执行 Agent 任务，逐条返回消息。

### `LoongAgentClient`

```python
class LoongAgentClient:
    async def send(self, prompt: str) -> None
    async def receive(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def close(self) -> None
```

### `LoongAgentOptions`

核心配置数据类，字段见 [配置选项](configuration.md)。

## 类型（`loong_agent.types`）

| 类型 | 说明 |
|------|------|
| `Message` | 通用消息，`type` 区分 system/assistant/user/result |
| `ToolCall` | 工具调用请求（id / name / arguments） |
| `ToolResult` | 工具执行结果（tool_use_id / name / content / is_error） |
| `Usage` | token 用量与成本 |

## LLM（`loong_agent.llm`）

| 类型 | 说明 |
|------|------|
| `LLMProvider` | 抽象基类：`chat_completion()`、`get_model_name()` |
| `OpenAICompatibleProvider` | OpenAI-compatible 适配器 |
| `ChatResponse` | 归一化响应（text / tool_calls / usage） |
| `ToolDefinition` | 工具描述（name / description / input_schema） |

## 工具（`loong_agent.tools`）

| 类型 | 说明 |
|------|------|
| `Tool` | 工具实例，`execute(**kwargs) -> ToolResult` |
| `tool` | 装饰器，自动推导 JSON Schema |
| `ToolRegistry` | 注册表，支持通配符过滤 |

## 钩子（`loong_agent.hooks`）

| 类型 | 说明 |
|------|------|
| `HookEvent` | 事件枚举 |
| `HookMatcher` | 匹配器（event + 正则） |
| `HookRegistry` | 注册与分发 |
| `HookOutput` | 回调返回值 |

## 权限（`loong_agent.permissions`）

| 类型 | 说明 |
|------|------|
| `PermissionPolicy` | allow / deny / ask 通配符规则 |
| `PermissionManager` | `check(tool, context) -> PermissionResult` |
| `PermissionDecision` | ALLOW / DENY / ASK |

## MCP（`loong_agent.mcp`）

| 类型 | 说明 |
|------|------|
| `MCPGateway` | 多服务器管理 |
| `MCPClient` | JSON-RPC 客户端 |
| `StdioTransport` / `HttpTransport` / `SseTransport` | 传输实现 |

## Skills（`loong_agent.skills`）

| 类型 | 说明 |
|------|------|
| `Skill` | 解析后的技能 |
| `SkillLoader` | 发现与解析 SKILL.md |
| `SkillExecutor` | 执行技能 |

## 子智能体（`loong_agent.subagents`）

| 类型 | 说明 |
|------|------|
| `AgentDefinition` | 子智能体定义 |
| `SubagentRunner` | 独立上下文运行 |
| `SubagentOrchestrator` | 编排与 `Agent` 工具 |

## 插件（`loong_agent.plugins`）

| 类型 | 说明 |
|------|------|
| `PluginManifest` | 插件清单 |
| `PluginLoader` | 组件发现 |
| `PluginManager` | 生命周期与聚合 |

## 追踪（`loong_agent.tracing`）

| 类型 | 说明 |
|------|------|
| `LoongAgentTracer` | OTel 封装（可降级） |
| `configure_tracing` | 配置导出器 |
