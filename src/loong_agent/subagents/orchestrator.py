"""Subagent orchestration and the ``Agent`` tool."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

from loong_agent.subagents.definition import AgentDefinition
from loong_agent.subagents.runner import SubagentRunner
from loong_agent.tools.base import Tool
from loong_agent.types.options import LoongAgentOptions


class SubagentOrchestrator:
    def __init__(
        self,
        parent_options: LoongAgentOptions,
        parent_llm=None,
        definitions: Optional[Dict[str, AgentDefinition]] = None,
    ) -> None:
        self.parent_options = parent_options
        self.definitions = definitions or parent_options.agents
        self.runner = SubagentRunner(parent_options)

    async def run(self, agent_name: str, prompt: str) -> str:
        definition = self.definitions.get(agent_name)
        if definition is None:
            return f"Error: unknown subagent '{agent_name}'. Available: {', '.join(self.definitions)}"
        return await self.runner.run(definition, prompt)

    async def run_parallel(self, tasks: List[Tuple[str, str]]) -> Dict[str, str]:
        """Run multiple subagents concurrently, keyed by agent name."""
        results = await asyncio.gather(
            *[self.run(name, prompt) for name, prompt in tasks],
            return_exceptions=True,
        )
        return {
            name: (str(res) if not isinstance(res, Exception) else f"Error: {res}")
            for (name, _), res in zip(tasks, results)
        }

    def build_tool(self) -> Tool:
        agent_names = list(self.definitions.keys())
        description = (
            "Delegate a task to a specialized subagent. Available subagents:\n"
            + "\n".join(
                f"- {name}: {self.definitions[name].description}" for name in agent_names
            )
        )
        schema = {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "enum": agent_names},
                "prompt": {"type": "string", "description": "Task description for the subagent."},
            },
            "required": ["agent_name", "prompt"],
            "additionalProperties": False,
        }
        return Tool(name="Agent", description=description, handler=self.run, input_schema=schema)


__all__ = ["SubagentOrchestrator"]
