"""Agent Loop engine — the unified ReAct event kernel.

``run_events`` is the single source of truth for execution: it yields the
three-tier stream (raw model chunks, semantic run items, agent updates) and is
consumed by ``Runner.run``, ``Runner.run_streamed`` and the legacy message
adapter ``run``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import List, Optional

from loong_agent.agent import Agent
from loong_agent.hooks.events import HookContext, HookEvent
from loong_agent.hooks.registry import Hook, HookCallback, HookRegistry
from loong_agent.llm.base import ChatResponse
from loong_agent.loop.context import SessionContext
from loong_agent.loop.sequencer import EventSequencer
from loong_agent.loop.turn import Turn
from loong_agent.permissions.policy import PermissionManager
from loong_agent.tools.base import Tool
from loong_agent.tools.builtin import BUILTIN_TOOLS
from loong_agent.tools.registry import ToolRegistry
from loong_agent.tracing.tracer import get_tracer
from loong_agent.types.events import (
    RunItemStreamEvent,
    StreamEvent,
)
from loong_agent.types.items import HandoffOutputItem, ToolCallOutputItem
from loong_agent.types.messages import Message, ToolResult
from loong_agent.types.options import LoongAgentOptions

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a capable autonomous agent. Use the provided tools to accomplish "
    "the user's request, reasoning step by step and acting when needed."
)

HANDOFF_TOOL_PREFIX = "transfer_to_"


class AgentLoop:
    """The core ReAct execution loop for one run of an agent."""

    def __init__(self, options: LoongAgentOptions, agent: Optional[Agent] = None) -> None:
        if agent is None:
            agent = Agent(
                name="assistant",
                instructions=options.system_prompt or "",
                llm=options.llm,
            )
        if agent.llm is None:
            agent.llm = options.llm
        if agent.llm is None:
            raise ValueError(
                "An LLM provider is required to run an agent (agent.llm or options.llm)."
            )

        self.options = options
        self.agent = agent
        self.llm = agent.llm
        self.context = SessionContext(options, llm=self.llm)
        self.hooks = HookRegistry.from_config(_normalize_hooks(options.hooks))
        self.permissions = PermissionManager(options.permission_policy)
        self.tracer = get_tracer()

        self.run_id: str = uuid.uuid4().hex
        self.sequencer = EventSequencer(self.run_id)

        # Run state
        self.current_agent: Agent = agent
        self.final_output: Optional[str] = None
        self.status: str = "running"  # running|completed|failed|cancelled|max_turns
        self.error: Optional[BaseException] = None
        self.raw_responses: List[ChatResponse] = []

        self._shared_tools: List[Tool] = []
        self._mcp_gateway = None
        self._skill_tool = None
        self._agent_tool = None
        self._setup_done = False
        self._session_ended = False
        self._cancel_requested = False

    # -- cancellation ---------------------------------------------------------
    def request_cancel(self) -> None:
        """Stop before the next turn begins (after-turn cancellation)."""
        self._cancel_requested = True

    # -- lifecycle -----------------------------------------------------------
    async def setup(self) -> None:
        """Assemble shared tools (builtins + MCP + skills + subagents)."""
        if self._setup_done:
            return
        self._setup_done = True

        allowed = self.options.allowed_tools
        for builtin in BUILTIN_TOOLS:
            if _matches_any(builtin.name, allowed):
                self._shared_tools.append(builtin)

        if self.options.mcp_servers:
            await self._setup_mcp()

        if self.options.skills:
            self._setup_skill_tool()

        if self.options.agents:
            self._setup_agent_tool()

    async def aclose(self) -> None:
        if self._mcp_gateway is not None:
            await self._mcp_gateway.close_all()
            self._mcp_gateway = None

    # -- shared tool assembly -------------------------------------------------
    async def _setup_mcp(self) -> None:
        from loong_agent.mcp.gateway import MCPGateway

        self._mcp_gateway = MCPGateway()
        for name, config in self.options.normalize_mcp_servers().items():
            await self._mcp_gateway.connect(name, config)
            for tool in await self._mcp_gateway.list_tools(name):
                full_name = f"mcp__{name}__{tool.name}"
                if not _matches_any(full_name, self.options.allowed_tools):
                    continue
                self._shared_tools.append(
                    Tool(
                        name=full_name,
                        description=tool.description,
                        handler=tool.invoke,
                        input_schema=tool.input_schema,
                    )
                )

    def _setup_skill_tool(self) -> None:
        from loong_agent.skills.executor import SkillExecutor
        from loong_agent.skills.loader import SkillLoader

        loader = SkillLoader()
        executor = SkillExecutor(loader=loader, enabled=self.options.skills)
        self._skill_tool = executor.build_tool()
        self._shared_tools.append(self._skill_tool)

    def _setup_agent_tool(self) -> None:
        from loong_agent.subagents.orchestrator import SubagentOrchestrator

        orchestrator = SubagentOrchestrator(self.options, parent_llm=self.llm)
        self._agent_tool = orchestrator.build_tool()
        self._shared_tools.append(self._agent_tool)

    # -- per-agent tool registry ----------------------------------------------
    def _build_registry(self, agent: Agent) -> ToolRegistry:
        """Build the toolset for the current agent (shared + agent + handoffs)."""
        allowed = list(self.options.allowed_tools)
        allowed.extend(tool.name for tool in agent.tools)
        allowed.extend(f"{HANDOFF_TOOL_PREFIX}{target.name}" for target in agent.handoffs)

        registry = ToolRegistry()
        for tool in self._shared_tools:
            if _matches_any(tool.name, allowed):
                registry.register(tool)
        for tool in agent.tools:
            if registry.get(tool.name) is None and _matches_any(tool.name, allowed):
                registry.register(tool)
        for target in agent.handoffs:
            name = f"{HANDOFF_TOOL_PREFIX}{target.name}"
            if registry.get(name) is None:
                registry.register(self._handoff_tool(name, target))
        return registry

    @staticmethod
    def _handoff_tool(name: str, target: Agent) -> Tool:
        async def _transfer(**_kwargs: object) -> str:
            return (
                f"Transferred to agent '{target.name}'. "
                f"{target.instructions or DEFAULT_SYSTEM_PROMPT}"
            )

        return Tool(
            name=name,
            description=(
                f"Transfer the conversation to agent '{target.name}'. "
                "Use this tool when the task should be handled by that agent."
            ),
            handler=_transfer,
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )

    # -- unified event kernel --------------------------------------------------
    async def run_events(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Execute the run and yield fine-grained stream events."""
        await self.setup()
        self.status = "running"
        self.current_agent = self.agent
        try:
            self.context.append(
                Message.system(text=self.agent.instructions or DEFAULT_SYSTEM_PROMPT)
            )
            self.context.append(Message.user(text=prompt))

            await self.hooks.dispatch(
                HookEvent.SESSION_START,
                {"prompt": prompt},
                context=HookContext(
                    session_id=self.context.session_id,
                    model=self.context.model_name,
                ),
            )

            # The starting point of the first agent's work.
            yield self.sequencer.agent_updated(self.agent)

            llm_kwargs = dict(self.options.extra_llm_kwargs)
            if self.options.temperature is not None:
                llm_kwargs["temperature"] = self.options.temperature

            while self.context.turn_count < self.options.max_turns:
                if self._cancel_requested:
                    self.status = "cancelled"
                    return

                self.context.turn_count += 1
                self.sequencer.turn_index = self.context.turn_count

                turn = Turn(
                    llm=self.llm,
                    agent=self.current_agent,
                    tools=self._build_registry(self.current_agent),
                    hooks=self.hooks,
                    permissions=self.permissions,
                    tracer=self.tracer,
                    context=self.context,
                    sequencer=self.sequencer,
                    llm_kwargs=llm_kwargs,
                    is_handoff=lambda call: call.name.startswith(HANDOFF_TOOL_PREFIX),
                )
                async for event in turn.run_events():
                    yield event

                outcome = turn.outcome
                if outcome is None:  # pragma: no cover - generator exhausted early
                    break
                if outcome.response is not None:
                    self.raw_responses.append(outcome.response)

                self.context.append(outcome.assistant)

                if outcome.is_final:
                    result = Message.result(
                        text=outcome.assistant.text,
                        usage=self.context.total_usage,
                        cost=self.context.total_usage.cost,
                    )
                    self.context.append(result)
                    self.final_output = result.text
                    self.status = "completed"
                    return

                # Handoff bookkeeping: the turn already emitted
                # ``handoff_requested``; record the switch and notify consumers.
                transferred = [
                    call
                    for call in outcome.assistant.tool_calls
                    if call.name.startswith(HANDOFF_TOOL_PREFIX)
                ]
                if transferred:
                    call = transferred[0]
                    target_name = call.name[len(HANDOFF_TOOL_PREFIX):]
                    output = next(
                        (
                            r
                            for r in (outcome.user.tool_results if outcome.user else [])
                            if r.tool_use_id == call.id
                        ),
                        None,
                    )
                    yield self.sequencer.item(
                        "handoff_occurred",
                        HandoffOutputItem(
                            agent=self.current_agent, tool_call=call, output=output
                        ),
                    )
                    target = next(
                        (
                            a
                            for a in self.current_agent.handoffs
                            if a.name == target_name
                        ),
                        None,
                    )
                    if target is not None and target is not self.current_agent:
                        self.current_agent = target
                        self.context.replace_system_prompt(
                            target.instructions or DEFAULT_SYSTEM_PROMPT
                        )
                        # The new agent's work starts here.
                        yield self.sequencer.agent_updated(target)

                if outcome.user is not None:
                    self.context.append(outcome.user)

            # Reached max turns without a final answer.
            result = Message.result(
                text=f"Stopped after reaching max_turns={self.options.max_turns}",
                usage=self.context.total_usage,
                cost=self.context.total_usage.cost,
                is_error=True,
            )
            self.context.append(result)
            self.final_output = result.text
            self.status = "max_turns"
        except asyncio.CancelledError:
            self.status = "cancelled"
            raise
        except Exception as exc:
            self.status = "failed"
            self.error = exc
            raise
        finally:
            await self._session_end()

    # -- legacy message adapter -------------------------------------------------
    async def run(self, prompt: str) -> AsyncIterator[Message]:
        """Backward-compatible message stream (system/assistant/user/result).

        Note: ``message_output_created`` is only emitted when the agent exits
        the ReAct loop with a user-facing message (final response, follow-up
        question or refusal). Intermediate turns (assistant messages carrying
        tool calls) produce no message item here — their text remains
        available through the raw delta stream of ``run_events``.
        """
        yield Message.system(
            subtype="init",
            session_id=self.context.session_id,
            model=self.context.model_name or self.options.model,
            text=self.agent.instructions or DEFAULT_SYSTEM_PROMPT,
        )

        pending_results: List[ToolResult] = []
        async for event in self.run_events(prompt):
            if isinstance(event, RunItemStreamEvent):
                # Final, user-facing message: flush pending tool results first
                # so the stream keeps the natural user -> assistant ordering.
                if event.name == "message_output_created":
                    if pending_results:
                        yield Message.user(tool_results=pending_results)
                        pending_results = []
                    yield event.item.message
                elif event.name == "tool_output":
                    assert isinstance(event.item, ToolCallOutputItem)
                    pending_results.append(event.item.output)

        if pending_results:
            yield Message.user(tool_results=pending_results)

        yield Message.result(
            text=self.final_output or "",
            usage=self.context.total_usage,
            cost=self.context.total_usage.cost,
            is_error=self.status not in ("completed",),
        )

    async def _session_end(self) -> None:
        if self._session_ended:
            return
        self._session_ended = True
        try:
            await self.hooks.dispatch(
                HookEvent.SESSION_END,
                context=HookContext(
                    session_id=self.context.session_id,
                    model=self.context.model_name,
                ),
            )
        finally:
            await self.aclose()


def _normalize_hooks(
    config: Mapping[str, Sequence[Hook | HookCallback]],
) -> dict[str, list[HookCallback]]:
    """Flatten ``{event_name: [hooks]}`` into ``{event_name: [callbacks]}``."""
    return {
        event_name: [hook.callback if isinstance(hook, Hook) else hook for hook in hooks]
        for event_name, hooks in config.items()
    }


def _matches_any(name: str, patterns: list[str]) -> bool:
    if not patterns or "*" in patterns:
        return True
    import fnmatch

    return any(fnmatch.fnmatch(name, p) for p in patterns)


__all__ = ["AgentLoop", "DEFAULT_SYSTEM_PROMPT", "HANDOFF_TOOL_PREFIX"]
