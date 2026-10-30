"""loong — an open-source, model-agnostic autonomous Agent SDK.

Quick start (openai-agents style)::

    import asyncio

    from loong_agent import Agent, LoongAgentOptions
    from loong_agent.llm import OpenAICompatibleProvider

    agent = Agent(
        name="joker",
        instructions="You are a helpful assistant.",
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        ),
    )


    async def main() -> None:
        # Non-streaming
        result = await agent.run("Hello")
        print(result.final_output)

        # Streaming with fine-grained events
        streamed = agent.run_streamed("Hello")
        async for event in streamed.stream_events():
            if event.type == "raw_response_event":
                continue  # or print token deltas
            elif event.type == "agent_updated_stream_event":
                print("Agent:", event.new_agent.name)
            elif event.type == "run_item_stream_event":
                print("Item:", event.name)

    asyncio.run(main())
"""

from __future__ import annotations

from loong_agent.agent import Agent
from loong_agent.client import LoongAgentClient, query
from loong_agent.loop.engine import AgentLoop
from loong_agent.result import RunResult, RunResultStreaming
from loong_agent.runner import Runner
from loong_agent.subagents.definition import AgentDefinition
from loong_agent.tools import Tool, tool
from loong_agent.types import (
    AgentUpdatedStreamEvent,
    AssistantMessage,
    HandoffCallItem,
    HandoffOutputItem,
    LoongAgentOptions,
    Message,
    MessageOutputItem,
    MessageType,
    RawResponsesStreamEvent,
    ReasoningItem,
    ResultMessage,
    RunItem,
    RunItemStreamEvent,
    StreamEvent,
    SystemMessage,
    ToolCall,
    ToolCallItem,
    ToolCallOutputItem,
    ToolResult,
    Usage,
    UserMessage,
)

__version__ = "0.1.0"

__all__ = [
    # New primary API
    "Agent",
    "Runner",
    "RunResult",
    "RunResultStreaming",
    "StreamEvent",
    "RawResponsesStreamEvent",
    "RunItemStreamEvent",
    "AgentUpdatedStreamEvent",
    "RunItem",
    "MessageOutputItem",
    "ReasoningItem",
    "ToolCallItem",
    "ToolCallOutputItem",
    "HandoffCallItem",
    "HandoffOutputItem",
    # Legacy / compatibility API
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
