"""Model provider registry and routing."""

from __future__ import annotations

from typing import Dict, Optional, Type

from loong_agent.llm.base import LLMProvider

_providers: Dict[str, Type[LLMProvider]] = {}


def register_provider(name: str):
    """Decorator registering a provider class under ``name``.

    Example::

        @register_provider("openai")
        class MyProvider(LLMProvider):
            ...
    """

    def _decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        _providers[name] = cls
        return cls

    return _decorator


def get_provider(name: str) -> Optional[Type[LLMProvider]]:
    return _providers.get(name)


def available_providers() -> Dict[str, Type[LLMProvider]]:
    return dict(_providers)


# Register the built-in provider.
from loong_agent.llm.openai_compatible import OpenAICompatibleProvider  # noqa: E402

_providers["openai"] = OpenAICompatibleProvider
_providers["openai-compatible"] = OpenAICompatibleProvider

__all__ = ["register_provider", "get_provider", "available_providers"]
