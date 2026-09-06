"""Plugin lifecycle management and component aggregation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from loong.plugins.loader import LoadedPlugin, PluginLoader
from loong.skills.loader import SkillLoader
from loong.skills.loader import Skill
from loong.subagents.definition import AgentDefinition
from loong.types.options import MCPServerConfig, PluginRef

logger = logging.getLogger(__name__)


class PluginManager:
    """Loads plugins and aggregates their skills/agents/mcp/hooks."""

    def __init__(self) -> None:
        self.loader = PluginLoader()
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._skills: List[Skill] = []
        self._agents: Dict[str, AgentDefinition] = {}
        self._mcp_servers: Dict[str, MCPServerConfig] = {}

    def load_all(self, refs: List[PluginRef]) -> None:
        for ref in refs:
            if ref.type != "local":
                logger.warning("Unsupported plugin type: %s", ref.type)
                continue
            try:
                self.activate(self.loader.load(ref.path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load plugin %s: %s", ref.path, exc)

    def activate(self, plugin: LoadedPlugin) -> None:
        self._plugins[plugin.manifest.name] = plugin

        for skill_path in plugin.skill_paths:
            skill = SkillLoader.parse(skill_path)
            if skill is not None:
                self._skills.append(skill)

        self._agents.update(plugin.agents)
        self._mcp_servers.update(plugin.mcp_servers)

    def deactivate(self, name: str) -> None:
        self._plugins.pop(name, None)

    @property
    def skills(self) -> List[Skill]:
        return self._skills

    @property
    def agents(self) -> Dict[str, AgentDefinition]:
        return self._agents

    @property
    def mcp_servers(self) -> Dict[str, MCPServerConfig]:
        return self._mcp_servers

    def summary(self) -> Dict[str, object]:
        """Human-readable summary surfaced in the session system message."""
        return {
            "plugins": [
                {"name": p.manifest.name, "version": p.manifest.version}
                for p in self._plugins.values()
            ],
            "skills": [s.name for s in self._skills],
            "agents": list(self._agents.keys()),
            "mcp_servers": list(self._mcp_servers.keys()),
        }


__all__ = ["PluginManager"]
