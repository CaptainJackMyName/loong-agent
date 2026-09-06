"""Filesystem discovery of SKILL.md files."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable


def discover_skills(paths: Iterable[str] | None = None) -> list[Path]:
    """Find all ``SKILL.md`` files under the given directories.

    When ``paths`` is empty, searches the conventional skill roots:
    ``.claude/skills`` and ``.loong_agent/skills`` relative to the current
    working directory.
    """
    roots: list[Path] = []
    if paths:
        roots = [Path(p).expanduser() for p in paths]
    else:
        roots = [
            Path.cwd() / ".loong_agent" / "skills",
        ]

    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == "SKILL.md":
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(root.rglob("SKILL.md")))
    return found


__all__ = ["discover_skills"]
