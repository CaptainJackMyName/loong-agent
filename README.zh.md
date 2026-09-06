# 龙智能体

Agent framework for all — 一个完全开源、模型无关、纯代码实现的自主 AI Agent SDK。

基于 [ReAct](https://arxiv.org/abs/2210.03629)（Reasoning + Acting）理论框架构建，参考 Claude Agent SDK 的设计理念，提供思考-行动-观察循环、工具系统、插件、MCP、Skills、子智能体、钩子、权限与链路追踪等能力。

## 特性

- **模型无关**：支持任意 OpenAI-compatible 接口（OpenAI、本地 LLaMA.cpp / Ollama / vLLM 等）。
- **纯代码实现**：全部逻辑使用 Python 源码，不依赖商业闭源二进制。
- **ReAct 循环**：Think → Act → Observe 的自主执行引擎。
- **丰富组件**：内置工具、钩子系统、权限策略、MCP 网关、Skills、子智能体、插件、OpenTelemetry 追踪。

## 安装

```bash
pip install -e .
```

## 快速开始

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
        if message.type == "assistant":
            print(f"Agent: {message.text}")
        elif message.type == "result":
            print(f"Done! Usage: {message.usage}")

asyncio.run(main())
```

更多示例见 [`examples/`](examples/)。

## 项目结构

```
src/loong_agent/
├── client.py          # LoongAgentClient / query 入口
├── loop/              # Agent Loop 引擎（ReAct 循环）
├── llm/               # LLM 适配层（OpenAI-compatible）
├── tools/             # 工具系统 + 内置工具（Read/Write/Edit/Bash/Grep/Glob）
├── hooks/             # 钩子系统
├── permissions/       # 权限系统
├── mcp/               # MCP 网关（stdio / http / in-process）
├── skills/            # Skills 系统（SKILL.md）
├── subagents/         # 子智能体系统
├── plugins/           # 插件系统
├── tracing/           # OpenTelemetry 链路追踪
└── types/             # 核心类型（Message / Options）
```

## License

MIT
