# 配置选项

所有配置集中在 [`LoongAgentOptions`](api.md#LoongAgentoptions) 数据类中。

## 字段一览

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | `LLMProvider` | `None`（必填） | 支撑 Agent 循环的 LLM 提供方 |
| `allowed_tools` | `List[str]` | `["*"]` | 允许使用的工具名（支持通配符） |
| `system_prompt` | `str` | `None` | 覆盖默认系统提示 |
| `skills` | `str` / `List[str]` | `[]` | `"all"`、技能名列表，或 `[]` 禁用 |
| `agents` | `Dict[str, AgentDefinition]` | `{}` | 子智能体定义 |
| `hooks` | `Dict[str, List[Callable]]` | `{}` | 钩子回调，按事件名分组 |
| `mcp_servers` | `Dict[str, MCPServerConfig]` | `{}` | MCP 服务器配置 |
| `plugins` | `List[PluginRef]` | `[]` | 要加载的插件 |
| `permission_policy` | `PermissionPolicy` | `None` | 权限策略 |
| `max_turns` | `int` | `50` | 最大 ReAct 回合数 |
| `model` | `str` | `None` | 模型名（`llm` 已指定时忽略） |
| `temperature` | `float` | `None` | 采样温度 |

## LLM 提供方

```python
from loong_agent.llm import OpenAICompatibleProvider

options = LoongAgentOptions(
    llm=OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
        timeout=60.0,
        temperature=0.7,
    ),
)
```

## 允许的工具

```python
# 允许全部
options = LoongAgentOptions(allowed_tools=["*"])

# 仅允许部分内置工具 + 某个 MCP 工具族
options = LoongAgentOptions(allowed_tools=["Read", "Grep", "mcp__filesystem__*"])
```

## 权限策略

```python
from loong_agent.permissions import PermissionPolicy

options = LoongAgentOptions(
    permission_policy=PermissionPolicy(
        allow=["Read", "Write", "Grep", "Glob"],
        deny=["Bash"],
        ask=["mcp__github__*"],
    ),
)
```

## 自定义系统提示

```python
options = LoongAgentOptions(
    llm=provider,
    system_prompt="You are a senior Python engineer. Be concise.",
)
```
