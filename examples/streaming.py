"""Streaming example mirroring the openai-agents style.

    python examples/streaming.py
"""

from __future__ import annotations

import asyncio

from loong_agent import Agent, LoongAgentOptions
from loong_agent.llm import OpenAICompatibleProvider


async def main() -> None:
    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant. Use tools when needed.",
        llm=OpenAICompatibleProvider(
            base_url="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        ),
        tools=[],
    )

    options = LoongAgentOptions(max_turns=10, allowed_tools=["*"])
    result = agent.run_streamed("List the Python files under src/", options=options)

    print("=== Run starting ===")
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            # Raw events carry Responses API stream semantics (same vocabulary
            # as the openai-agents SDK), e.g. token-level deltas.
            if event.data.type == "response.output_text.delta":
                print(event.data.delta, end="", flush=True)
            elif event.data.type == "response.reasoning_summary_text.delta":
                print(f"[reasoning] {event.data.delta}", end="", flush=True)
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            if event.item.type == "message_output_item":
                print(f"-- Message: {event.item.message.text!r}")
            elif event.item.type == "tool_call_item":
                print(f"-- Tool called: {event.item.tool_call.name}")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output.content[:80]!r}")
            elif event.item.type == "handoff_call_item":
                print("-- Handoff requested")
            elif event.item.type == "handoff_output_item":
                print("-- Handoff occurred")

    print("=== Run complete ===")
    print("Final output:", result.final_output)
    print("Usage:", result.usage)


if __name__ == "__main__":
    asyncio.run(main())
