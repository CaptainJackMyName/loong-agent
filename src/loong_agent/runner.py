"""Runner — shared execution implementation behind ``Agent.run``/``run_streamed``.

Both styles drive the exact same ``AgentLoop`` event kernel:

- ``run`` drains the event stream internally and returns a final ``RunResult``.
- ``run_streamed`` starts the loop as a background task, forwards every event
  into a queue and returns a mutable ``RunResultStreaming`` immediately.

Note: ``message_output_created`` (and therefore the ``messages`` field of the
results) only carries the final, user-facing assistant message emitted when
the agent exits the ReAct loop — final response, follow-up question or
refusal. Intermediate turns carrying tool calls emit no message item.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import List, Optional

from loong_agent.agent import Agent
from loong_agent.loop.engine import AgentLoop
from loong_agent.result import _QUEUE_COMPLETE, RunResult, RunResultStreaming
from loong_agent.types.events import RunItemStreamEvent
from loong_agent.types.items import MessageOutputItem
from loong_agent.types.options import LoongAgentOptions

_HANDOFF_TOOL_PREFIX = "transfer_to_"


class Runner:
    """Entry point used by :meth:`Agent.run` and :meth:`Agent.run_streamed`."""

    @classmethod
    def _build_options(
        cls, agent: Agent, options: Optional[LoongAgentOptions]
    ) -> LoongAgentOptions:
        base = options or LoongAgentOptions()
        if base.llm is None:
            base = replace(base, llm=agent.llm)
        if base.llm is None:
            raise ValueError(
                f"Agent '{agent.name}' needs an LLM provider (agent.llm or options.llm)."
            )
        # Ensure agent-owned tools and handoff tools pass the allowlist.
        allowed = list(base.allowed_tools)
        if "*" not in allowed:
            allowed.extend(tool.name for tool in agent.tools)
            allowed.extend(f"{_HANDOFF_TOOL_PREFIX}{target.name}" for target in agent.handoffs)
        return replace(base, allowed_tools=allowed)

    @classmethod
    async def run(
        cls,
        agent: Agent,
        input: str,
        *,
        options: Optional[LoongAgentOptions] = None,
    ) -> RunResult:
        opts = cls._build_options(agent, options)
        loop = AgentLoop(opts, agent)
        new_items: List = []
        messages: List = []
        try:
            async for event in loop.run_events(input):
                if isinstance(event, RunItemStreamEvent):
                    new_items.append(event.item)
                    if isinstance(event.item, MessageOutputItem):
                        messages.append(event.item.message)
        finally:
            await loop.aclose()

        if loop.status == "failed" and loop.error is not None:
            raise loop.error

        return RunResult(
            input=input,
            final_output=loop.final_output,
            new_items=new_items,
            messages=messages,
            raw_responses=list(loop.raw_responses),
            usage=loop.context.total_usage,
            last_agent=loop.current_agent,
            status=loop.status,
            error=loop.error,
        )

    @classmethod
    def run_streamed(
        cls,
        agent: Agent,
        input: str,
        *,
        options: Optional[LoongAgentOptions] = None,
    ) -> RunResultStreaming:
        """Start the run in the background (requires a running event loop)."""
        opts = cls._build_options(agent, options)
        loop = AgentLoop(opts, agent)
        result = RunResultStreaming(
            input=input,
            current_agent=agent,
            max_turns=opts.max_turns,
        )
        result._loop = loop
        result._run_task = asyncio.create_task(cls._run_streamed_task(loop, result))
        return result

    @staticmethod
    async def _run_streamed_task(loop: AgentLoop, result: RunResultStreaming) -> None:
        try:
            async for event in loop.run_events(result.input):
                result._consumed.clear()
                await result._event_queue.put(event)
                # Wait until the consumer has picked the event up so that
                # consumers (UIs, decision points) always observe an event —
                # e.g. the complete message — before the run continues.
                await result._consumed.wait()
                if isinstance(event, RunItemStreamEvent):
                    result.new_items.append(event.item)
                    if isinstance(event.item, MessageOutputItem):
                        result.messages.append(event.item.message)
        except Exception as exc:  # noqa: BLE001 - re-raised to the consumer
            result._exception = exc
        finally:
            result.final_output = loop.final_output
            result.raw_responses = list(loop.raw_responses)
            result.usage = loop.context.total_usage
            result.last_agent = loop.current_agent
            result.current_agent = loop.current_agent
            result.current_turn = loop.context.turn_count
            result.status = loop.status
            result.error = loop.error
            result.is_complete = True
            await result._event_queue.put(_QUEUE_COMPLETE)


__all__ = ["Runner"]
