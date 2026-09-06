"""Configuration options for the Free Agent SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from loong_agent.hooks.registry import Hook
    from loong_agent.llm.base import LLMProvider
    from loong_agent.permissions.policy import PermissionPolicy
    from loong_agent.subagents.definition import AgentDefinition


@dataclass
class MCPServerConfig:
    """Connection configuration for a single MCP server."""

    type: str = "stdio"  # "stdio" | "http" | "inprocess"
    # stdio
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    # http
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    # in-process
    server: Any = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MCPServerConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class PluginRef:
    """Reference to a plugin to load."""

    type: str = "local"  # "local"
    path: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PluginRef":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class LoongAgentOptions:
    """Top-level configuration for an agent run.

    Attributes:
        llm: The LLM provider backing the agent loop.
        allowed_tools: Tool names (or ``*``) the agent is permitted to use.
        system_prompt: Optional overriding system prompt.
        skills: ``"all"``, a list of skill names, or ``[]`` to disable skills.
        agents: Subagent definitions keyed by name.
        hooks: Hook callbacks keyed by event name.
        mcp_servers: MCP server configurations keyed by name.
        plugins: Plugins to load.
        permission_policy: Optional permission policy.
        max_turns: Maximum number of ReAct turns before giving up.
        model: Convenience model name (ignored if ``llm`` already sets one).
        temperature: Sampling temperature forwarded to the LLM.
    """

    llm: Optional["LLMProvider"] = None
    allowed_tools: List[str] = field(default_factory=lambda: ["*"])
    system_prompt: Optional[str] = None
    skills: Union[str, List[str]] = field(default_factory=list)
    agents: Dict[str, "AgentDefinition"] = field(default_factory=dict)
    hooks: dict[str, list["Hook"]] = field(default_factory=dict)
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    plugins: List[PluginRef] = field(default_factory=list)
    permission_policy: Optional["PermissionPolicy"] = None
    max_turns: int = 50
    model: Optional[str] = None
    temperature: Optional[float] = None
    extra_llm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def normalize_mcp_servers(self) -> Dict[str, MCPServerConfig]:
        """Coerce raw dict configs into :class:`MCPServerConfig` objects."""
        normalized: Dict[str, MCPServerConfig] = {}
        for name, cfg in self.mcp_servers.items():
            if isinstance(cfg, MCPServerConfig):
                normalized[name] = cfg
            elif isinstance(cfg, dict):
                normalized[name] = MCPServerConfig.from_dict(cfg)
            else:
                raise TypeError(f"Invalid MCP server config for '{name}'")
        return normalized

    def normalize_plugins(self) -> List[PluginRef]:
        out: List[PluginRef] = []
        for ref in self.plugins:
            if isinstance(ref, PluginRef):
                out.append(ref)
            elif isinstance(ref, dict):
                out.append(PluginRef.from_dict(ref))
            else:
                raise TypeError("Invalid plugin reference")
        return out


__all__ = ["LoongAgentOptions", "MCPServerConfig", "PluginRef"]
