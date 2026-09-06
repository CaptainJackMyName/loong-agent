"""Hook registration and dispatch."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loong_agent.hooks.events import HookContext, HookEvent, HookOutput
from loong_agent.hooks.matcher import HookMatcher

HookCallback = Callable[..., Any]


@dataclass
class Hook:
    """A registered hook callback with an optional matcher."""

    callback: HookCallback
    matcher: Optional[HookMatcher] = None

    async def invoke(
        self,
        event: HookEvent,
        input_data: Optional[Dict[str, Any]] = None,
        tool_use_id: Optional[str] = None,
        context: Optional[HookContext] = None,
    ) -> Optional[HookOutput]:
        if self.matcher is not None and not self.matcher.matches(
            event, _target(input_data)
        ):
            return None

        signature = inspect.signature(self.callback)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        available: Dict[str, Any] = {
            "input_data": input_data or {},
            "tool_use_id": tool_use_id,
            "context": context or HookContext(),
        }
        kwargs: Dict[str, Any] = {
            name: value
            for name, value in available.items()
            if name in signature.parameters or accepts_var_kw
        }

        result = self.callback(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return None
        if isinstance(result, HookOutput):
            return result
        # Allow callbacks to return a plain permission string.
        if isinstance(result, str):
            return HookOutput(permission_decision=result)
        return HookOutput(**result) if isinstance(result, dict) else None


def _target(input_data: Optional[Dict[str, Any]]) -> Optional[str]:
    if not input_data:
        return None
    for key in ("tool_name", "name", "event"):
        if key in input_data:
            return str(input_data[key])
    return None


class HookRegistry:
    """Collects hooks and dispatches events to matching callbacks."""

    def __init__(self) -> None:
        self._hooks: List[Hook] = []

    def register(self, callback: HookCallback, matcher: Optional[HookMatcher] = None) -> None:
        self._hooks.append(Hook(callback=callback, matcher=matcher))

    def register_for_event(
        self,
        event: HookEvent,
        callback: HookCallback,
        matcher: Optional[str] = None,
    ) -> None:
        self._hooks.append(
            Hook(callback=callback, matcher=HookMatcher(event=event, matcher=matcher))
        )

    def clear(self) -> None:
        self._hooks.clear()

    async def dispatch(
        self,
        event: HookEvent,
        input_data: Optional[Dict[str, Any]] = None,
        tool_use_id: Optional[str] = None,
        context: Optional[HookContext] = None,
    ) -> List[HookOutput]:
        """Invoke every matching hook and return their outputs."""
        outputs: List[HookOutput] = []
        for hook in self._hooks:
            out = await hook.invoke(event, input_data, tool_use_id, context)
            if out is not None:
                outputs.append(out)
        return outputs

    @staticmethod
    def from_config(config: dict[str, list[HookCallback]]) -> "HookRegistry":
        """Build a registry from an ``{event_name: [callbacks]}`` mapping."""
        registry = HookRegistry()
        for event_name, callbacks in config.items():
            try:
                event = HookEvent(event_name)
            except ValueError:
                # Unknown events are registered with no matcher and dispatched
                # only when an explicit event filter matches the raw name.
                continue
            for callback in callbacks:
                registry.register_for_event(event, callback)
        return registry


__all__ = ["Hook", "HookRegistry", "HookCallback"]
