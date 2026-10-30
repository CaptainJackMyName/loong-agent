"""Single ReAct turn execution (Think -> Act -> Observe) as an event stream.

A turn is an async generator of :class:`~loong_agent.types.events.StreamEvent`.
The complete outcome (assistant message, tool results, aggregated model
response) is stored on :attr:`Turn.outcome` once the generator is exhausted, so
the engine can continue the ReAct loop from the same unified pipeline used by
both streaming and non-streaming runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Callable, List, Optional

from loong_agent.hooks.events import HookContext, HookEvent, HookOutput
from loong_agent.hooks.registry import HookRegistry
from loong_agent.llm.base import ChatResponse, LLMProvider
from loong_agent.llm.responses_converter import ChatCompletionToResponsesConverter
from loong_agent.llm.streaming import ChatCompletionAggregator
from loong_agent.loop.context import SessionContext
from loong_agent.loop.sequencer import EventSequencer
from loong_agent.permissions.policy import PermissionManager
from loong_agent.tools.registry import ToolRegistry
from loong_agent.tracing.spans import TOOL_EXECUTE
from loong_agent.tracing.tracer import LoongAgentTracer
from loong_agent.types.events import RawResponsesStreamEvent, RunItemStreamEvent
from loong_agent.types.items import (
    HandoffCallItem,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from loong_agent.types.messages import Message, ToolCall, ToolResult

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.agent import Agent


@dataclass
class TurnOutcome:
    """Result of a single ReAct turn."""

    assistant: Message
    user: Optional[Message] = None
    is_final: bool = False
    response: Optional[ChatResponse] = None


class Turn:
    """Encapsulates one think/act/observe cycle as a fine-grained event stream."""

    def __init__(
        self,
        llm: LLMProvider,
        agent: "Agent",
        tools: ToolRegistry,
        hooks: HookRegistry,
        permissions: PermissionManager,
        tracer: LoongAgentTracer,
        context: SessionContext,
        sequencer: EventSequencer,
        llm_kwargs: Optional[dict] = None,
        is_handoff: Optional[Callable[[ToolCall], bool]] = None,
    ) -> None:
        self.llm = llm
        self.agent = agent
        self.tools = tools
        self.hooks = hooks
        self.permissions = permissions
        self.tracer = tracer
        self.context = context
        self.sequencer = sequencer
        self.llm_kwargs = llm_kwargs or {}
        self.is_handoff = is_handoff or (lambda call: False)
        self.outcome: Optional[TurnOutcome] = None

    async def run_events(self) -> AsyncIterator[RawResponsesStreamEvent | RunItemStreamEvent]:
        # 1. Think — forward raw deltas (converted to Responses API stream
        #    events, mirroring the openai-agents SDK), then aggregate.
        messages = self.context.to_llm_messages()
        definitions = self.tools.definitions()
        aggregator = ChatCompletionAggregator()
        converter = ChatCompletionToResponsesConverter()
        async for chunk in self.llm.chat_completion_stream(
            messages=messages,
            tools=definitions or None,
            **self.llm_kwargs,
        ):
            aggregator.add(chunk)
            for event in converter.add(chunk):
                yield self.sequencer.raw(event)
        for event in converter.finish():
            yield self.sequencer.raw(event)
        response = aggregator.finish()
        self.tracer.set_attribute("llm.model", response.model or self.llm.get_model_name())
        self.tracer.set_attribute("llm.tool_calls", len(response.tool_calls))

        # 2. Reasoning item (only when the model actually returned one).
        if response.reasoning:
            yield self.sequencer.item(
                "reasoning_item_created", ReasoningItem(agent=self.agent, text=response.reasoning)
            )

        # 3. No tool calls -> final, user-facing answer. This is the ONLY place
        #    ``message_output_created`` is emitted (final response, follow-up
        #    questions and refusals all end the loop through this branch).
        if not response.has_tool_calls():
            assistant = Message.assistant(
                text=response.text, model=response.model, usage=response.usage
            )
            yield self.sequencer.item(
                "message_output_created", MessageOutputItem(agent=self.agent, message=assistant)
            )
            self.outcome = TurnOutcome(assistant=assistant, is_final=True, response=response)
            return

        # 4. Act + Observe. The intermediate assistant message (text + tool
        #    calls) is deliberately NOT emitted as ``message_output_created``:
        #    that event is reserved for the final, user-facing message produced
        #    when the agent exits the ReAct loop (no-tool-calls branch above).
        #    The intermediate text is still available via the raw delta stream.
        assistant = Message.assistant(
            text=response.text,
            tool_calls=response.tool_calls,
            model=response.model,
            usage=response.usage,
        )

        results: List[ToolResult] = []
        async for event, result in self._execute_tools(response.tool_calls):
            yield event
            if result is not None:
                results.append(result)

        user = Message.user(tool_results=results)
        self.outcome = TurnOutcome(
            assistant=assistant, user=user, is_final=False, response=response
        )

    async def _execute_tools(
        self, tool_calls: List[ToolCall]
    ) -> AsyncIterator[tuple[RawResponsesStreamEvent | RunItemStreamEvent, Optional[ToolResult]]]:
        results: List[ToolResult] = []
        hook_ctx = HookContext(
            session_id=self.context.session_id,
            model=self.context.model_name,
        )
        for call in tool_calls:
            handoff = self.is_handoff(call)
            if not handoff:
                # Semantic "tool selected" event: emitted after the arguments
                # are fully parsed and immediately before execution.
                yield (
                    self.sequencer.item(
                        "tool_called", ToolCallItem(agent=self.agent, tool_call=call)
                    ),
                    None,
                )
            async with self.tracer.span(TOOL_EXECUTE) as span:
                span.set_attribute("tool.name", call.name)
                result = await self._execute_one(call, hook_ctx)
                span.set_attribute("tool.is_error", result.is_error)
            results.append(result)
            if handoff:
                yield (
                    self.sequencer.item(
                        "handoff_requested",
                        HandoffCallItem(agent=self.agent, tool_call=call),
                    ),
                    result,
                )
            else:
                yield (
                    self.sequencer.item(
                        "tool_output",
                        ToolCallOutputItem(agent=self.agent, tool_call=call, output=result),
                    ),
                    result,
                )

    async def _execute_one(self, call: ToolCall, hook_ctx: HookContext) -> ToolResult:
        input_data = {"tool_name": call.name, "tool_input": call.arguments}

        # PreToolUse hooks
        pre_outputs = await self.hooks.dispatch(
            HookEvent.PRE_TOOL_USE, input_data, call.id, hook_ctx
        )
        denied = _first_denial(pre_outputs)
        if denied is not None:
            return ToolResult(
                tool_use_id=call.id,
                name=call.name,
                content=f"Permission denied by hook: {denied.output or 'denied'}",
                is_error=True,
            )

        # Policy decision
        decision = await self.permissions.check(call.name, input_data)
        if not decision.allowed:
            return ToolResult(
                tool_use_id=call.id,
                name=call.name,
                content=(
                    f"Permission denied: "
                    f"{decision.reason or decision.message or 'not allowed'}"
                ),
                is_error=True,
            )

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_use_id=call.id,
                name=call.name,
                content=f"Error: unknown tool '{call.name}'",
                is_error=True,
            )

        result = await tool.execute(**call.arguments)
        result.tool_use_id = call.id

        # PostToolUse hooks
        await self.hooks.dispatch(
            HookEvent.POST_TOOL_USE,
            {"tool_name": call.name, "tool_response": result.content},
            call.id,
            hook_ctx,
        )
        return result


def _first_denial(outputs: List[HookOutput]) -> Optional[HookOutput]:
    for out in outputs:
        if out.permission_decision == "deny":
            return out
    return None


__all__ = ["Turn", "TurnOutcome"]
