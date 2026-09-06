"""Span name constants following the ``loong_agent.<layer>.<op>`` convention."""

from __future__ import annotations

LOOP_TURN = "loong_agent.loop.turn"
LLM_REQUEST = "loong_agent.llm.request"
TOOL_EXECUTE = "loong_agent.tool.execute"
HOOK_RUN = "loong_agent.hook.run"
MCP_CALL = "loong_agent.mcp.call"
SUBAGENT_RUN = "loong_agent.subagent.run"
SKILL_EXECUTE = "loong_agent.skill.execute"

__all__ = [
    "LOOP_TURN",
    "LLM_REQUEST",
    "TOOL_EXECUTE",
    "HOOK_RUN",
    "MCP_CALL",
    "SUBAGENT_RUN",
    "SKILL_EXECUTE",
]
