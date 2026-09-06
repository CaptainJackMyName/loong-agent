# loong-agent

Agent framework for all — a fully open-source, model-agnostic, pure-code autonomous AI Agent SDK.

Built on the [ReAct](https://arxiv.org/abs/2210.03629) (Reasoning + Acting) framework and inspired by the design of the Claude Agent SDK, it provides a thought-action-observation loop, a tool system, plugins, MCP, Skills, subagents, hooks, permissions, and tracing.

## Features

- **Model-agnostic**: works with any OpenAI-compatible API (OpenAI, local LLaMA.cpp / Ollama / vLLM, etc.).
- **Pure code implementation**: all logic is written in Python source, with no dependency on commercial closed-source binaries.
- **ReAct loop**: an autonomous execution engine of Think → Act → Observe.
- **Rich components**: built-in tools, hook system, permission policies, MCP gateway, Skills, subagents, plugins, and OpenTelemetry tracing.

## Installation

```bash
pip install -e .
```

## Quick Start

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

See [`examples/`](examples/) for more examples.

## Project Structure

```
src/loong_agent/
├── client.py          # LoongAgentClient / query entry point
├── loop/              # Agent Loop engine (ReAct loop)
├── llm/               # LLM adapter layer (OpenAI-compatible)
├── tools/             # Tool system + built-in tools (Read/Write/Edit/Bash/Grep/Glob)
├── hooks/             # Hook system
├── permissions/       # Permission system
├── mcp/               # MCP gateway (stdio / http / in-process)
├── skills/            # Skills system (SKILL.md)
├── subagents/         # Subagent system
├── plugins/           # Plugin system
├── tracing/           # OpenTelemetry tracing
└── types/             # Core types (Message / Options)
```

## License

MIT
