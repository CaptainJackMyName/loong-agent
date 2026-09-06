"""Subagent runner — executes a subagent in an isolated context."""

from __future__ import annotations

from typing import Optional

from loong_agent.subagents.definition import AgentDefinition
from loong_agent.tracing.spans import SUBAGENT_RUN
from loong_agent.tracing.tracer import get_tracer
from loong_agent.types.options import LoongAgentOptions


class SubagentRunner:
    def __init__(self, parent_options: LoongAgentOptions) -> None:
        self.parent_options = parent_options
        self.tracer = get_tracer()

    def _options_for(self, definition: AgentDefinition) -> LoongAgentOptions:
        llm = self.parent_options.llm
        if definition.model and llm is not None:
            llm = llm.clone(model=definition.model)

        return LoongAgentOptions(
            llm=llm,
            allowed_tools=definition.tools or self.parent_options.allowed_tools,
            system_prompt=definition.system_prompt,
            max_turns=self.parent_options.max_turns,
            temperature=self.parent_options.temperature,
        )

    async def run(self, definition: AgentDefinition, prompt: str) -> str:
        # Lazy import avoids a circular dependency with loop.engine.
        from loong_agent.loop.engine import AgentLoop

        options = self._options_for(definition)
        loop = AgentLoop(options)
        final_text = ""
        async with self.tracer.span(SUBAGENT_RUN) as span:
            span.set_attribute("subagent.prompt", prompt)
            async for message in loop.run(prompt):
                if message.type == "result":
                    final_text = message.text
        await loop.aclose()
        return final_text


__all__ = ["SubagentRunner"]
