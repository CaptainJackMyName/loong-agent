"""MCP transport abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Transport(ABC):
    """Request/response transport for the MCP JSON-RPC protocol."""

    @abstractmethod
    async def request(
        self, method: str, params: Optional[Dict[str, Any]], request_id: int
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the response object."""

    @abstractmethod
    async def notify(self, method: str, params: Optional[Dict[str, Any]]) -> None:
        """Send a JSON-RPC notification (no response expected)."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the transport."""


__all__ = ["Transport"]
