"""Permission policy and decision management."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loong_agent.permissions.decision import PermissionDecision, PermissionResult


@dataclass
class PermissionPolicy:
    """Allow/deny/ask rules using wildcard patterns (``*``)."""

    allow: List[str] = field(default_factory=lambda: ["*"])
    deny: List[str] = field(default_factory=list)
    ask: List[str] = field(default_factory=list)

    def matches(self, pattern: str, target: str) -> bool:
        return fnmatch.fnmatch(target, pattern)

    def in_list(self, rules: List[str], target: str) -> bool:
        return any(self.matches(p, target) for p in rules)


class PermissionManager:
    def __init__(self, policy: Optional[PermissionPolicy] = None) -> None:
        self.policy = policy or PermissionPolicy()

    async def check(self, tool: str, context: Optional[Dict[str, Any]] = None) -> PermissionResult:
        context = context or {}
        # Explicit deny always wins.
        if self.policy.in_list(self.policy.deny, tool):
            return PermissionResult.deny(f"'{tool}' is denied by policy")
        if self.policy.in_list(self.policy.ask, tool):
            return PermissionResult.ask(f"'{tool}' requires confirmation")
        if self.policy.in_list(self.policy.allow, tool):
            return PermissionResult.allow()
        return PermissionResult.deny(f"'{tool}' is not allowed by policy")

    async def check_all(self, tool: str, context: Optional[Dict[str, Any]] = None) -> PermissionDecision:
        return (await self.check(tool, context)).decision


__all__ = ["PermissionPolicy", "PermissionManager"]
