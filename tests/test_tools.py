"""Tests for the tool system and built-in tools."""

from __future__ import annotations

import pytest

from loong.tools import Tool, ToolRegistry, tool
from loong.tools.builtin.bash import bash
from loong.tools.builtin.files import (
    edit_file,
    glob_files,
    grep,
    read_file,
    write_file,
)


def test_tool_decorator_infers_schema():
    @tool("Add", "add two numbers")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    assert add.name == "Add"
    assert "a" in add.input_schema["properties"]
    assert add.input_schema["properties"]["a"] == {"type": "integer"}
    assert add.input_schema["required"] == ["a"]


@pytest.mark.asyncio
async def test_tool_execute():
    @tool("Add", "add two numbers")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    result = await add.execute(a=1, b=2)
    assert result.content == "3"
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_execute_captures_exception():
    @tool("Fail", "always fails")
    async def fail() -> str:
        raise ValueError("boom")

    result = await fail.execute()
    assert result.is_error
    assert "boom" in result.content


def test_registry_wildcard_filter():
    registry = ToolRegistry()
    registry.register(Tool("Read", "d", lambda: None))
    registry.register(Tool("mcp__fs__ls", "d", lambda: None))

    filtered = registry.filter(["mcp__fs__*"])
    assert filtered.names() == ["mcp__fs__ls"]


@pytest.mark.asyncio
async def test_file_tools_roundtrip(tmp_path):
    target = tmp_path / "a.txt"

    res = await write_file.execute(file_path=str(target), content="hello\nworld\n")
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"

    res = await read_file.execute(file_path=str(target))
    assert "hello" in res.content

    res = await edit_file.execute(file_path=str(target), old_string="hello", new_string="HELLO")
    assert target.read_text(encoding="utf-8").startswith("HELLO")

    res = await grep.execute(pattern="HELLO", path=str(tmp_path))
    assert "HELLO" in res.content

    res = await glob_files.execute(pattern="*.txt", path=str(tmp_path))
    assert "a.txt" in res.content


@pytest.mark.asyncio
async def test_read_missing_file_reports_error(tmp_path):
    res = await read_file.execute(file_path=str(tmp_path / "nope.txt"))
    assert "not found" in res.content


@pytest.mark.asyncio
async def test_edit_non_unique_string(tmp_path):
    target = tmp_path / "dup.txt"
    target.write_text("x x", encoding="utf-8")
    res = await edit_file.execute(file_path=str(target), old_string="x", new_string="y")
    assert "not unique" in res.content


@pytest.mark.asyncio
async def test_bash_echo():
    res = await bash.execute(command="echo hello")
    assert "hello" in res.content
