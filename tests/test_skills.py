"""Tests for the skills system."""

from __future__ import annotations

import pytest

from loong.skills.executor import SkillExecutor
from loong.skills.loader import SkillLoader


def test_skill_loader_parse_frontmatter(tmp_path):
    skill_dir = tmp_path / "pdf"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text("---\nname: pdf\ndescription: handle pdf\n---\n\nDo the pdf thing.", encoding="utf-8")

    skill = SkillLoader.parse(md)
    assert skill.name == "pdf"
    assert "handle pdf" in skill.description
    assert "Do the pdf thing" in skill.content


def test_skill_loader_discover(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text("---\nname: a\n---\nbody", encoding="utf-8")
    (tmp_path / "b" / "SKILL.md").write_text("---\nname: b\n---\nbody", encoding="utf-8")

    skills = SkillLoader().discover([str(tmp_path)])
    assert {s.name for s in skills} == {"a", "b"}


@pytest.mark.asyncio
async def test_skill_executor_enabled_all(tmp_path):
    (tmp_path / "pdf").mkdir(parents=True)
    (tmp_path / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf\ndescription: handle pdf\n---\n\nbody here", encoding="utf-8"
    )
    loader = SkillLoader(roots=[str(tmp_path)])
    executor = SkillExecutor(loader=loader, enabled="all")

    result = await executor.execute("pdf")
    assert "body here" in result

    tool = executor.build_tool()
    assert tool.name == "Skill"


@pytest.mark.asyncio
async def test_skill_executor_disabled(tmp_path):
    (tmp_path / "pdf").mkdir(parents=True)
    (tmp_path / "pdf" / "SKILL.md").write_text(
        "---\nname: pdf\n---\nbody", encoding="utf-8"
    )
    loader = SkillLoader(roots=[str(tmp_path)])
    executor = SkillExecutor(loader=loader, enabled=[])

    result = await executor.execute("pdf")
    assert "not enabled" in result


@pytest.mark.asyncio
async def test_skill_executor_unknown(tmp_path):
    loader = SkillLoader(roots=[str(tmp_path)])
    executor = SkillExecutor(loader=loader, enabled="all")
    result = await executor.execute("missing")
    assert "unknown skill" in result
