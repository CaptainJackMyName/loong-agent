"""Permission decision types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionResult:
    decision: PermissionDecision
    reason: str = ""
    message: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecision.ALLOW

    @classmethod
    def allow(cls, reason: str = "") -> "PermissionResult":
        return cls(decision=PermissionDecision.ALLOW, reason=reason)

    @classmethod
    def deny(cls, reason: str = "") -> "PermissionResult":
        return cls(decision=PermissionDecision.DENY, reason=reason)

    @classmethod
    def ask(cls, reason: str = "") -> "PermissionResult":
        return cls(decision=PermissionDecision.ASK, reason=reason)


__all__ = ["PermissionDecision", "PermissionResult"]
