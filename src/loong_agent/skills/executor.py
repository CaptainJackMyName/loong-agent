"""Skill execution.

Skills are instruction bundles: invoking one injects its ``SKILL.md`` content
into the conversation so the model can follow the documented procedure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from loong_agent.skills.loader import Skill, SkillLoader
from loong_agent.tools.base import Tool, tool


class SkillExecutor:
    def __init__(
        self,
        loader: Optional[SkillLoader] = None,
        enabled: Union[str, List[str], None] = None,
    ) -> None:
        self.loader = loader or SkillLoader()
        self.enabled = enabled or []
        self._skills: Optional[List[Skill]] = None

    def _load(self) -> List[Skill]:
        if self._skills is None:
            self._skills = self.loader.discover()
        return self._skills

    def available_names(self) -> List[str]:
        return [s.name for s in self._load()]

    def _is_enabled(self, name: str) -> bool:
        if self.enabled == "all":
            return True
        if isinstance(self.enabled, list):
            return name in self.enabled
        return False

    async def execute(self, skill_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        skill = self.loader.load(skill_name)
        if skill is None:
            names = ", ".join(self.available_names()) or "(none)"
            return f"Error: unknown skill '{skill_name}'. Available skills: {names}"
        if not self._is_enabled(skill.name):
            return f"Error: skill '{skill_name}' is not enabled."
        header = f"# Skill: {skill.name}\n{skill.description}\n\n"
        return header + skill.content

    def build_tool(self) -> Tool:
        schema = {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to invoke.",
                    "enum": self.available_names(),
                }
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        }
        return Tool(
            name="Skill",
            description="Invoke a skill to load its instructions and procedures.",
            handler=self.execute,
            input_schema=schema,
        )


__all__ = ["SkillExecutor"]
