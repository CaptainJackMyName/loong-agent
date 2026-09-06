"""Plugin system."""

from loong.plugins.lifecycle import PluginManager
from loong.plugins.loader import LoadedPlugin, PluginLoader
from loong.plugins.manifest import PluginManifest

__all__ = ["PluginManifest", "PluginLoader", "LoadedPlugin", "PluginManager"]
