"""Tool registry."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from loong_agent.llm.base import ToolDefinition
from loong_agent.tools.base import Tool


class ToolRegistry:
    """A named collection of tools with wildcard filtering."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def definitions(self) -> List[ToolDefinition]:
        return [t.to_definition() for t in self._tools.values()]

    def filter(self, allowed: Optional[List[str]]) -> "ToolRegistry":
        """Return a new registry restricted to ``allowed`` patterns.

        Patterns support ``*`` wildcards, e.g. ``"mcp__filesystem__*"``.
        """
        import fnmatch

        if allowed is None or "*" in allowed:
            return self

        filtered = ToolRegistry()
        for name, tool in self._tools.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in allowed):
                filtered.register(tool)
        return filtered


__all__ = ["ToolRegistry"]
