"""The primary Agent object — what users create and run.

Usage mirrors the openai-agents SDK::

    agent = Agent(name="joker", instructions="...", llm=provider, tools=[...])

    result = await agent.run("Hello")            # non-streaming
    streamed = agent.run_streamed("Hello")       # streaming
    async for event in streamed.stream_events():
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, List, Optional

from loong_agent.llm.base import LLMProvider
from loong_agent.tools.base import Tool
from loong_agent.types.options import LoongAgentOptions

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.result import RunResult, RunResultStreaming


@dataclass
class Agent:
    """A runnable agent definition.

    Attributes:
        name: Unique identifier, also used for handoff transfer tools.
        instructions: System prompt for this agent.
        llm: The LLM provider backing this agent. May be omitted when the run
            ``options`` supply one.
        tools: Additional tools available to this agent.
        handoffs: Agents this agent may transfer control to.
    """

    name: str = "assistant"
    instructions: str = ""
    llm: Optional[LLMProvider] = None
    tools: List[Tool] = field(default_factory=list)
    handoffs: List["Agent"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Agent.name must not be empty.")
        names = [target.name for target in self.handoffs]
        if len(names) != len(set(names)):
            raise ValueError(f"Agent '{self.name}' has duplicate handoff targets.")
        if self.name in names:
            raise ValueError(f"Agent '{self.name}' cannot hand off to itself.")

    # -- entrypoints ----------------------------------------------------------
    def run(
        self,
        input: str,
        *,
        options: Optional[LoongAgentOptions] = None,
    ) -> Awaitable["RunResult"]:
        """Run the agent to completion and return the final result."""
        from loong_agent.runner import Runner

        return Runner.run(self, input, options=options)

    def run_streamed(
        self,
        input: str,
        *,
        options: Optional[LoongAgentOptions] = None,
    ) -> "RunResultStreaming":
        """Start the agent in the background and return a streaming handle.

        Must be called inside a running event loop. Consume events via
        ``result.stream_events()`` until the iterator finishes, then read
        ``result.final_output``.
        """
        from loong_agent.runner import Runner

        return Runner.run_streamed(self, input, options=options)


__all__ = ["Agent"]
