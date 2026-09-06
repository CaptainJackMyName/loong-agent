# loong

Agent framework for all —— 一个**完全开源、模型无关、纯代码实现**的自主 AI Agent SDK。

基于 [ReAct](https://arxiv.org/abs/2210.03629)（Reasoning + Acting）理论框架构建，参考 Claude Agent SDK 的设计理念，提供思考-行动-观察循环、工具系统、插件、MCP、Skills、子智能体、钩子、权限与链路追踪等能力。

## 核心特性

- **模型无关**：支持任意 OpenAI-compatible 接口（OpenAI、本地 LLaMA.cpp / Ollama / vLLM 等）。
- **纯代码实现**：全部逻辑使用 Python 源码，不依赖任何商业闭源二进制。
- **ReAct 循环**：`Think → Act → Observe` 的自主执行引擎。
- **丰富组件**：

| 组件 | 说明 |
|------|------|
| 工具系统 | 内置 Read / Write / Edit / Bash / Grep / Glob，支持 `@tool` 装饰器 |
| 钩子系统 | `PreToolUse` / `PostToolUse` 等事件，可 allow / deny / modify |
| 权限系统 | allow / deny / ask 通配符策略 |
| MCP 网关 | stdio / http / SSE / in-process 四种传输 |
| Skills | `SKILL.md` 文件发现与执行 |
| 子智能体 | 上下文隔离、并行执行 |
| 插件系统 | 聚合 skills / agents / hooks / MCP |
| 链路追踪 | OpenTelemetry Span / Metrics |

## 安装

```bash
pip install -e .
```

## 快速示例

```python
import asyncio

from loong_agent import LoongAgentOptions, query
from loong_agent.llm import OpenAICompatibleProvider

async def main():
    options = LoongAgentOptions(
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        ),
        allowed_tools=["Read", "Write", "Bash", "Grep", "Glob"],
    )

    async for message in query("Review the code in src/", options=options):
        print(message.type, message.text)

asyncio.run(main())
```

## 架构

```
Public API (query / LoongAgentClient)
            │
     Agent Loop Engine (Think → Act → Observe)
            │
  ┌─────────┼───────────┬───────────┬───────────┐
  │         │           │           │           │
 Tools     Hooks       MCP        Skills     Subagents
  │         │           │           │           │
  └─────────┴───────────┴───────────┴───────────┘
            │
  Plugin & Extension Layer
            │
  LLM Provider Layer (OpenAI-compatible)
            │
  Observability (OpenTelemetry)
```

## 文档导航

- [快速开始](quickstart.md)：三步跑通第一个 Agent
- [配置选项](configuration.md)：`LoongAgentOptions` 全部字段
- [工具系统](tools.md)：内置工具与自定义工具
- [钩子与权限](hooks.md)：事件拦截与权限决策
- [MCP 网关](mcp.md)：接入外部工具
- [Skills 与子智能体](skills-subagents.md)：技能与委托
- [插件系统](plugins.md)：打包与加载
- [链路追踪](tracing.md)：可观测性
- [API 参考](api.md)：公开接口清单
