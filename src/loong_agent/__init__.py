"""loong — an open-source, model-agnostic autonomous Agent SDK.

Quick start::

    import asyncio

    from loong_agent import LoongAgentOptions, query
    from loong_agent.llm import OpenAICompatibleProvider

    options = LoongAgentOptions(
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        ),
        allowed_tools=["Read", "Write", "Bash"],
    )

    async def main() -> None:
        async for message in query("Review src/", options=options):
            print(message.type, message.text)

    asyncio.run(main())
"""

from __future__ import annotations

from loong_agent.client import LoongAgentClient, query
from loong_agent.loop.engine import AgentLoop
from loong_agent.subagents.definition import AgentDefinition
from loong_agent.tools import Tool, tool
from loong_agent.types import (
    AssistantMessage,
    LoongAgentOptions,
    Message,
    MessageType,
    ResultMessage,
    SystemMessage,
    ToolCall,
    ToolResult,
    Usage,
    UserMessage,
)

__version__ = "0.1.0"

__all__ = [
    "query",
    "LoongAgentClient",
    "LoongAgentOptions",
    "AgentLoop",
    "AgentDefinition",
    "Tool",
    "tool",
    "Message",
    "MessageType",
    "ToolCall",
    "ToolResult",
    "Usage",
    "SystemMessage",
    "AssistantMessage",
    "UserMessage",
    "ResultMessage",
    "__version__",
]
