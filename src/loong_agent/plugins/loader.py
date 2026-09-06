"""Plugin loader — discovers the components bundled in a plugin directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loong.plugins.manifest import PluginManifest
from loong.subagents.definition import AgentDefinition
from loong.types.options import MCPServerConfig


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    root: Path
    skill_paths: List[Path] = field(default_factory=list)
    agents: Dict[str, AgentDefinition] = field(default_factory=dict)
    mcp_servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    hook_paths: List[Path] = field(default_factory=list)


class PluginLoader:
    def load(self, path: str) -> LoadedPlugin:
        root = Path(path).expanduser().resolve()
        manifest = PluginManifest.load(root)
        plugin = LoadedPlugin(manifest=manifest, root=root)

        plugin.skill_paths = self._discover_skills(root)
        plugin.agents = self._discover_agents(root)
        plugin.mcp_servers = self._discover_mcp(root)
        plugin.hook_paths = self._discover_hooks(root)
        return plugin

    @staticmethod
    def _discover_skills(root: Path) -> List[Path]:
        skills_dir = root / "skills"
        if not skills_dir.exists():
            return []
        return sorted(skills_dir.rglob("SKILL.md"))

    @staticmethod
    def _discover_agents(root: Path) -> Dict[str, AgentDefinition]:
        agents_dir = root / "agents"
        if not agents_dir.exists():
            return {}
        agents: Dict[str, AgentDefinition] = {}
        for agent_file in sorted(agents_dir.glob("*.md")):
            definition = _parse_agent_markdown(agent_file)
            if definition is not None:
                agents[agent_file.stem] = definition
        return agents

    @staticmethod
    def _discover_mcp(root: Path) -> Dict[str, MCPServerConfig]:
        mcp_dir = root / "mcp"
        if not mcp_dir.exists():
            return {}
        servers: Dict[str, MCPServerConfig] = {}
        for mcp_file in sorted(mcp_dir.glob("*.json")):
            try:
                data = json.loads(mcp_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            servers[mcp_file.stem] = MCPServerConfig.from_dict(data)
        return servers

    @staticmethod
    def _discover_hooks(root: Path) -> List[Path]:
        hooks_dir = root / "hooks"
        if not hooks_dir.exists():
            return []
        return sorted(hooks_dir.glob("*.py"))


def _parse_agent_markdown(path: Path) -> Optional[AgentDefinition]:
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    metadata: Dict[str, Any] = {}
    body = text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if match:
        body = text[match.end() :]
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                metadata[key.strip()] = value

    description = metadata.get("description", path.stem)
    if isinstance(description, list):
        description = " ".join(description)
    return AgentDefinition(
        description=str(description),
        system_prompt=body.strip(),
        tools=list(metadata.get("tools", [])),
        model=metadata.get("model"),
    )


__all__ = ["PluginLoader", "LoadedPlugin"]
