"""Tests for the plugin system."""

from __future__ import annotations

import json

from loong.plugins.lifecycle import PluginManager
from loong.plugins.loader import PluginLoader
from loong.plugins.manifest import PluginManifest
from loong.types.options import PluginRef


def test_manifest_load_from_root(tmp_path):
    (tmp_path / "plugin.json").write_text(
        json.dumps({"name": "p", "version": "1.0.0", "description": "desc"}), encoding="utf-8"
    )
    manifest = PluginManifest.load(tmp_path)
    assert manifest.name == "p"
    assert manifest.version == "1.0.0"
    assert manifest.description == "desc"


def test_manifest_load_from_claude_plugin_dir(tmp_path):
    (tmp_path / ".claude-plugin").mkdir()
    (tmp_path / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "p2"}), encoding="utf-8"
    )
    assert PluginManifest.load(tmp_path).name == "p2"


def test_plugin_loader_discovers_components(tmp_path):
    root = tmp_path / "myplugin"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "p"}), encoding="utf-8")

    (root / "skills" / "s1").mkdir(parents=True)
    (root / "skills" / "s1" / "SKILL.md").write_text("---\nname: s1\n---\nbody", encoding="utf-8")

    (root / "agents").mkdir()
    (root / "agents" / "reviewer.md").write_text(
        "---\ndescription: reviews code\ntools: [Read, Grep]\n---\nYou review.", encoding="utf-8"
    )

    (root / "mcp").mkdir()
    (root / "mcp" / "fs.json").write_text(
        json.dumps({"type": "stdio", "command": "npx"}), encoding="utf-8"
    )

    plugin = PluginLoader().load(str(root))
    assert plugin.manifest.name == "p"
    assert len(plugin.skill_paths) == 1
    assert "reviewer" in plugin.agents
    assert plugin.agents["reviewer"].tools == ["Read", "Grep"]
    assert "fs" in plugin.mcp_servers
    assert plugin.mcp_servers["fs"].command == "npx"


def test_plugin_manager_aggregates(tmp_path):
    root = tmp_path / "myplugin"
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "p"}), encoding="utf-8")
    (root / "skills" / "s1").mkdir(parents=True)
    (root / "skills" / "s1" / "SKILL.md").write_text("---\nname: s1\n---\nbody", encoding="utf-8")

    manager = PluginManager()
    manager.load_all([PluginRef(type="local", path=str(root))])

    assert manager.summary()["skills"] == ["s1"]
    assert manager.summary()["plugins"][0]["name"] == "p"
