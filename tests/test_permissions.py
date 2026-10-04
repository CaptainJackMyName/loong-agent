"""Tests for the permission system."""

from __future__ import annotations

import pytest

from loong_agent.permissions import PermissionDecision, PermissionManager, PermissionPolicy


@pytest.mark.asyncio
async def test_allow_by_default():
    manager = PermissionManager()
    result = await manager.check("Read", {})
    assert result.decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_deny_wins_over_allow():
    policy = PermissionPolicy(allow=["*"], deny=["Bash"])
    manager = PermissionManager(policy)
    result = await manager.check("Bash", {})
    assert result.decision == PermissionDecision.DENY


@pytest.mark.asyncio
async def test_ask_rule():
    policy = PermissionPolicy(allow=["Read"], ask=["Write"])
    manager = PermissionManager(policy)
    assert (await manager.check("Write", {})).decision == PermissionDecision.ASK
    assert (await manager.check("Read", {})).decision == PermissionDecision.ALLOW


@pytest.mark.asyncio
async def test_wildcard_patterns():
    policy = PermissionPolicy(allow=["mcp__filesystem__*"])
    manager = PermissionManager(policy)
    assert (await manager.check("mcp__filesystem__read", {})).allowed
    assert not (await manager.check("mcp__github__read", {})).allowed
