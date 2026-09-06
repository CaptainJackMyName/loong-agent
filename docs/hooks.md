# 钩子与权限

## 钩子系统

钩子在关键事件触发时执行自定义回调，可用于拦截、修改或注入上下文。

### 事件类型

| 事件 | 触发时机 |
|------|----------|
| `PreToolUse` | 工具执行前 |
| `PostToolUse` | 工具执行后 |
| `SubagentStart` | 子智能体启动 |
| `SubagentStop` | 子智能体停止 |
| `SessionStart` | 会话开始 |
| `SessionEnd` | 会话结束 |
| `Idle` | Agent 空闲 |

### 定义钩子

钩子回调可以声明它需要的参数子集（`input_data`、`tool_use_id`、`context`）：

```python
from loong_agent.hooks import HookOutput

async def security_hook(input_data, tool_use_id, context):
    if input_data.get("tool_name") == "Bash":
        return HookOutput(permission_decision="deny", output="Bash is disabled")
    return HookOutput(permission_decision="allow")
```

### 注册钩子

```python
from loong_agent import LoongAgentOptions

options = LoongAgentOptions(
    llm=provider,
    hooks={
        "PreToolUse": [security_hook],
        "SessionStart": [my_start_hook],
    },
)
```

### HookOutput 字段

| 字段 | 说明 |
|------|------|
| `permission_decision` | `allow` / `deny` / `ask` |
| `updated_input` | 替换操作的输入 |
| `injected_context` | 注入对话的额外上下文 |
| `output` | 展示给调用方的消息 |
| `stop_reason` | 终止会话的原因 |

## 权限系统

### 权限策略

```python
from loong_agent.permissions import PermissionPolicy

policy = PermissionPolicy(
    allow=["Read", "Write", "mcp__filesystem__*"],
    deny=["Bash"],
    ask=["mcp__github__*"],
)
```

规则支持 `*` 通配符，`deny` 优先级最高，其次 `ask`，最后 `allow`。

### 决策类型

- `PermissionDecision.ALLOW` — 放行
- `PermissionDecision.DENY` — 拒绝
- `PermissionDecision.ASK` — 需要确认

工具执行时，循环引擎会先跑 `PreToolUse` 钩子，再查权限策略，两者任一拒绝都会返回错误结果给模型。
