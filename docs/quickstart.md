# 快速开始

## 1. 安装

```bash
pip install -e .
```

## 2. 编写第一个 Agent

```python
import asyncio

from loong_agent import LoongAgentOptions, query
from loong_agent.llm import OpenAICompatibleProvider

async def main():
    options = LoongAgentOptions(
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",        # 替换为你的密钥
            model="gpt-4o-mini",
        ),
        allowed_tools=["Read", "Write", "Bash", "Grep", "Glob"],
    )

    async for message in query("List the files in the current directory.", options=options):
        if message.type == "system":
            print(f"[system:{message.subtype}] {message.text}")
        elif message.type == "assistant":
            print(f"[assistant] {message.text}")
        elif message.type == "user":
            print(f"[user] {len(message.tool_results)} tool result(s)")
        elif message.type == "result":
            print(f"[result] {message.text}")

asyncio.run(main())
```

## 3. 使用本地模型

任意 OpenAI-compatible 接口均可接入：

```python
options = LoongAgentOptions(
    llm=OpenAICompatibleProvider(
        base_url="http://localhost:1234/v1",   # LM Studio / llama.cpp
        api_key="not-needed",
        model="llama-3.3-70b",
    ),
)
```

## 4. 流式交互（客户端模式）

```python
from loong_agent import LoongAgentClient

client = LoongAgentClient(options=options)
await client.send("Analyze this project structure")

async for message in client.receive():
    print(message.type, message.text)

await client.close()
```

## 消息类型

| 类型 | 说明 | 关键字段 |
|------|------|----------|
| `system` | 会话元数据（init） | `subtype`, `session_id`, `model` |
| `assistant` | 模型输出 | `text`, `tool_calls` |
| `user` | 工具执行结果 | `tool_results` |
| `result` | 最终结果 | `text`, `usage`, `cost` |
