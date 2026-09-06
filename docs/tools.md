# 工具系统

工具是 Agent 与外部环境交互的接口。每个工具包含 `name`、`description` 和 `input_schema`（JSON Schema），供 LLM 决定何时调用。

## 内置工具

| 工具名 | 功能 |
|--------|------|
| `Read` | 读取文件（支持 offset / limit） |
| `Write` | 写入文件（自动创建父目录） |
| `Edit` | 精确字符串替换 |
| `Bash` | 执行 shell 命令 |
| `Grep` | 正则搜索文件内容 |
| `Glob` | 文件模式匹配 |

## 自定义工具

使用 `@tool` 装饰器，参数类型注解会自动推导为 JSON Schema：

```python
from loong_agent.tools import tool

@tool("search_web", "Search the web for information")
async def search_web(query: str, max_results: int = 10) -> str:
    ...
    return "results..."
```

## 手动指定 Schema

```python
from loong_agent.tools import Tool

async def handler(a: int) -> str:
    return str(a)

custom = Tool(
    name="Square",
    description="Square a number",
    handler=handler,
    input_schema={
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
    },
)
```

## 工具执行结果

`Tool.execute(**kwargs)` 返回 `ToolResult`：

- 正常：`content` 为字符串结果，`is_error=False`。
- 异常：自动捕获异常，`content` 为错误信息，`is_error=True`。

## 注册与过滤

```python
from loong_agent.tools import ToolRegistry

registry = ToolRegistry()
registry.register(custom)
registry.filter(["mcp__filesystem__*"])   # 通配符过滤
```

内置工具通过 `LoongAgentOptions.allowed_tools` 控制是否注册到循环中。
