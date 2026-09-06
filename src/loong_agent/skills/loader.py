"""SKILL.md parsing and loading."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loong.skills.discovery import discover_skills


@dataclass
class Skill:
    """A parsed skill."""

    name: str
    description: str = ""
    content: str = ""
    path: Optional[Path] = None
    metadata: Dict[str, str] = field(default_factory=dict)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillLoader:
    def __init__(self, roots: Optional[Iterable[str]] = None) -> None:
        self.roots = list(roots) if roots is not None else None

    def discover(self, paths: Optional[Iterable[str]] = None) -> List[Skill]:
        search_paths = paths if paths is not None else self.roots
        skills: List[Skill] = []
        for md_path in discover_skills(search_paths):
            skill = self.parse(md_path)
            if skill is not None:
                skills.append(skill)
        return skills

    def load(self, skill_name: str, paths: Optional[Iterable[str]] = None) -> Optional[Skill]:
        for skill in self.discover(paths):
            if skill.name == skill_name:
                return skill
        return None

    @staticmethod
    def parse(path: Path) -> Optional[Skill]:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata: Dict[str, str] = {}
        body = text
        match = _FRONTMATTER_RE.match(text)
        if match:
            body = text[match.end() :]
            for line in match.group(1).splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()

        name = metadata.get("name") or path.parent.name or path.stem
        description = metadata.get("description", "")
        return Skill(
            name=name,
            description=description,
            content=body.strip(),
            path=path,
            metadata=metadata,
        )


__all__ = ["Skill", "SkillLoader"]
