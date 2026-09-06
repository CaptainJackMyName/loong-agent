"""Hook matching utilities."""

from __future__ import annotations

import re

from dataclasses import dataclass

from loong_agent.hooks.events import HookEvent


@dataclass
class HookMatcher:
    """Filters whether a hook should fire for a given event/input.

    ``matcher``, when provided, is a regular expression tested against the
    string form of the hook input (e.g. a tool name).
    """

    event: HookEvent
    matcher: str | None = None

    def matches(self, event: HookEvent, target: str | None = None) -> bool:
        if self.event != event:
            return False
        if self.matcher is None:
            return True
        if target is None:
            return False
        return re.search(self.matcher, target) is not None


__all__ = ["HookMatcher"]
