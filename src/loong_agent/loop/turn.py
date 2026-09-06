"""Single ReAct turn execution (Think -> Act -> Observe)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loong_agent.hooks.events import HookContext, HookEvent, HookOutput
from loong_agent.hooks.registry import HookRegistry
from loong_agent.llm.base import ChatResponse, LLMProvider
from loong_agent.loop.context import SessionContext
from loong_agent.permissions.policy import PermissionManager
from loong_agent.tools.registry import ToolRegistry
from loong_agent.tracing.spans import LLM_REQUEST, TOOL_EXECUTE
from loong_agent.tracing.tracer import LoongAgentTracer
from loong_agent.types.messages import Message, ToolCall, ToolResult, Usage


@dataclass
class TurnOutcome:
    """Result of a single ReAct turn."""

    assistant: Message
    user: Optional[Message] = None
    result: Optional[Message] = None
    is_final: bool = False


class Turn:
    """Encapsulates one think/act/observe cycle."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        hooks: HookRegistry,
        permissions: PermissionManager,
        tracer: LoongAgentTracer,
        context: SessionContext,
        llm_kwargs: Optional[dict] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.hooks = hooks
        self.permissions = permissions
        self.tracer = tracer
        self.context = context
        self.llm_kwargs = llm_kwargs or {}

    async def run(self) -> TurnOutcome:
        # 1. Think
        async with self.tracer.span(LLM_REQUEST) as span:
            response = await self._think(span)

        # 2. No tool calls -> final answer
        if not response.has_tool_calls():
            assistant = Message.assistant(
                text=response.text, model=response.model, usage=response.usage
            )
            result = Message.result(
                text=response.text,
                usage=self.context.total_usage.merge(response.usage),
                cost=self.context.total_usage.merge(response.usage).cost,
            )
            return TurnOutcome(assistant=assistant, result=result, is_final=True)

        # 3. Act + Observe
        assistant = Message.assistant(
            text=response.text,
            tool_calls=response.tool_calls,
            model=response.model,
            usage=response.usage,
        )
        results = await self._execute_tools(response.tool_calls)
        user = Message.user(tool_results=results)
        return TurnOutcome(assistant=assistant, user=user, is_final=False)

    async def _think(self, span) -> ChatResponse:
        messages = self.context.to_llm_messages()
        definitions = self.tools.definitions()
        response = await self.llm.chat_completion(
            messages=messages,
            tools=definitions or None,
            **self.llm_kwargs,
        )
        span.set_attribute("llm.model", response.model or self.llm.get_model_name())
        span.set_attribute("llm.tool_calls", len(response.tool_calls))
        return response

    async def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        hook_ctx = HookContext(
            session_id=self.context.session_id,
            model=self.context.model_name,
        )
        for call in tool_calls:
            async with self.tracer.span(TOOL_EXECUTE) as span:
                span.set_attribute("tool.name", call.name)
                result = await self._execute_one(call, hook_ctx)
                span.set_attribute("tool.is_error", result.is_error)
            results.append(result)
        return results

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
                content=f"Permission denied: {decision.reason or decision.message or 'not allowed'}",
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
