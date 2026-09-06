"""Tests for the subagent system."""

from __future__ import annotations

import pytest

from loong_agent.subagents.definition import AgentDefinition
from loong_agent.subagents.orchestrator import SubagentOrchestrator
from loong_agent.subagents.runner import SubagentRunner
from loong_agent.types.options import LoongAgentOptions


def test_agent_definition_defaults():
    definition = AgentDefinition(description="d", system_prompt="sp")
    assert definition.tools == []
    assert definition.model is None
    assert definition.parallel is False


def test_orchestrator_build_tool(make_llm):
    options = LoongAgentOptions(
        llm=make_llm(),
        agents={"reviewer": AgentDefinition(description="reviews code", system_prompt="review")},
    )
    orchestrator = SubagentOrchestrator(options)
    tool = orchestrator.build_tool()
    assert tool.name == "Agent"
    assert "reviewer" in tool.description
    assert "reviewer" in tool.input_schema["properties"]["agent_name"]["enum"]


@pytest.mark.asyncio
async def test_orchestrator_unknown_agent(make_llm):
    options = LoongAgentOptions(llm=make_llm(), agents={})
    orchestrator = SubagentOrchestrator(options)
    result = await orchestrator.run("missing", "do it")
    assert "unknown subagent" in result


@pytest.mark.asyncio
async def test_subagent_runner_returns_final_text(make_llm):
    options = LoongAgentOptions(llm=make_llm(), max_turns=3)
    runner = SubagentRunner(options)
    result = await runner.run(
        AgentDefinition(description="d", system_prompt="sp"), "hello"
    )
    assert result == "done"
