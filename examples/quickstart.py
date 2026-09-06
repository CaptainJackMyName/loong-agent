"""Quick-start example for the loong-agent.

Run with::

    python examples/quickstart.py
"""

import asyncio
import os

from loong_agent import LoongAgentOptions, query
from loong_agent.llm import OpenAICompatibleProvider
from loong_agent.tools import tool

@tool("Add", "Add two integers together.")
async def add(a: int, b: int) -> str:
    return str(a + b)


async def main() -> None:
    base_url: str = os.environ.get("LOONG_AGENT_BASE_URL", "http://127.0.0.1:8000")
    api_key: str = os.environ.get("LOONG_AGENT_API_KEY", "EMPTY")
    model: str = os.environ.get("LOONG_AGENT_MODEL", "gpt-5.6-terra")
    options = LoongAgentOptions(
        llm=OpenAICompatibleProvider(
            base_url=base_url,
            api_key=api_key,  # replace with your key
            model=model,
        ),
        allowed_tools=["Read", "Write", "Bash", "Grep", "Glob"],
        max_turns=20,
    )

    async for message in query("Summarize the files in the current directory.", options=options):
        if message.type == "system":
            print(f"[system:{message.subtype}] {message.text[:60]}")
        elif message.type == "assistant":
            print(f"[assistant] {message.text[:120]}")
        elif message.type == "user":
            print(f"[user] {len(message.tool_results)} tool result(s)")
        elif message.type == "result":
            print(f"[result] {message.text}")


if __name__ == "__main__":
    asyncio.run(main())
