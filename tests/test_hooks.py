"""Tests for the hook system."""

from __future__ import annotations

import pytest

from loong_agent.hooks import HookContext, HookEvent, HookMatcher, HookOutput, HookRegistry


def test_matcher_filtering():
    matcher = HookMatcher(event=HookEvent.PRE_TOOL_USE, matcher="Bash")
    assert matcher.matches(HookEvent.PRE_TOOL_USE, "Bash")
    assert not matcher.matches(HookEvent.PRE_TOOL_USE, "Read")
    assert not matcher.matches(HookEvent.POST_TOOL_USE, "Bash")


def test_matcher_without_pattern_matches_any_target():
    matcher = HookMatcher(event=HookEvent.IDLE)
    assert matcher.matches(HookEvent.IDLE, None)


@pytest.mark.asyncio
async def test_registry_dispatches_matching_hooks():
    calls = []

    async def hook(input_data, context):
        calls.append(input_data)
        return HookOutput(permission_decision="deny", output="blocked")

    registry = HookRegistry()
    registry.register_for_event(HookEvent.PRE_TOOL_USE, hook, matcher="Bash")

    outputs = await registry.dispatch(
        HookEvent.PRE_TOOL_USE, {"tool_name": "Bash"}, "tool-1", HookContext()
    )
    assert len(outputs) == 1
    assert outputs[0].permission_decision == "deny"

    no_outputs = await registry.dispatch(
        HookEvent.PRE_TOOL_USE, {"tool_name": "Read"}, "tool-1", HookContext()
    )
    assert no_outputs == []


@pytest.mark.asyncio
async def test_hook_can_declare_only_subset_of_args():
    async def minimal(input_data):
        return "allow"

    registry = HookRegistry()
    registry.register_for_event(HookEvent.PRE_TOOL_USE, minimal)
    outputs = await registry.dispatch(
        HookEvent.PRE_TOOL_USE, {"tool_name": "X"}, "tool-1", HookContext()
    )
    assert outputs[0].permission_decision == "allow"


@pytest.mark.asyncio
async def test_from_config_builds_registry():
    async def cb(input_data, context):
        return "deny"

    registry = HookRegistry.from_config({"PreToolUse": [cb]})
    outputs = await registry.dispatch(
        HookEvent.PRE_TOOL_USE, {"tool_name": "X"}, None, HookContext()
    )
    assert outputs[0].permission_decision == "deny"


def test_hook_output_helpers():
    assert HookOutput.allow().permission_decision == "allow"
    denied = HookOutput.deny(reason="nope")
    assert denied.permission_decision == "deny"
    assert denied.output == "nope"
