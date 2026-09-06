"""Plugin manifest parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PluginManifest:
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PluginManifest":
        if path.is_dir():
            candidates = [
                path / ".claude-plugin" / "plugin.json",
                path / "plugin.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break
            else:
                raise FileNotFoundError(f"No plugin.json found in {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", path.parent.name),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author"),
            raw=data,
        )


__all__ = ["PluginManifest"]
