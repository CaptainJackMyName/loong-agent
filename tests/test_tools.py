"""Tests for the tool system and built-in tools."""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import Field

from loong_agent.tools import Tool, ToolRegistry, tool
from loong_agent.tools.builtin.bash import bash
from loong_agent.tools.builtin.files import (
    edit_file,
    glob_files,
    grep,
    read_file,
    write_file,
)


def test_tool_bare_decorator_uses_docstring():
    @tool
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"weather in {city}"

    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather for a city."
    assert get_weather.input_schema["properties"]["city"]["type"] == "string"


def test_tool_decorator_keyword_args():
    @tool(name="Add", description="add two numbers")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    assert add.name == "Add"
    assert add.description == "add two numbers"


def test_tool_decorator_infers_strict_schema():
    @tool(name="Add", description="add two numbers")
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    schema = add.input_schema
    assert schema["type"] == "object"
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "integer"
    # Strict mode: every declared property is required.
    assert schema["required"] == ["a", "b"]
    assert schema["additionalProperties"] is False


def test_tool_decorator_non_strict_schema():
    @tool(name="Add", description="add two numbers", strict_mode=False)
    async def add(a: int, b: int = 0) -> str:
        return str(a + b)

    schema = add.input_schema
    assert schema["required"] == ["a"]
    assert "additionalProperties" not in schema


def test_tool_decorator_annotated_descriptions():
    @tool
    async def get_weather(
        city: Annotated[str, "The city name"],
        unit: Annotated[str, Field(description="Temperature unit")] = "celsius",
    ) -> str:
        """Get the current weather for a city."""
        return city

    schema = get_weather.input_schema
    assert schema["properties"]["city"]["description"] == "The city name"
    assert schema["properties"]["unit"]["description"] == "Temperature unit"
    assert schema["properties"]["unit"]["default"] == "celsius"


def test_tool_docstring_param_descriptions():
    @tool
    def search(query: str, limit: int = 5) -> str:
        """Search the web.

        Args:
            query: The search query.
            limit: Maximum results to return.
        """
        return query

    schema = search.input_schema
    assert search.description == "Search the web."
    assert schema["properties"]["query"]["description"] == "The search query."
    assert schema["properties"]["limit"]["description"] == "Maximum results to return."


def test_tool_rejects_positional_name_description():
    with pytest.raises(TypeError, match="no longer accepts positional"):

        @tool("Read", "desc")
        async def read(path: str) -> str:  # pragma: no cover - never called
            return path


def test_tool_rejects_var_keyword_in_strict_mode():
    with pytest.raises(ValueError, match="strict"):

        @tool
        def flexible(**kwargs: Any) -> str:  # pragma: no cover - never called
            return "ok"


def test_tool_var_keyword_non_strict():
    @tool(strict_mode=False)
    def flexible(a: int, **extra: Any) -> str:
        return f"{a}"

    assert "extra" in flexible.input_schema["properties"]
    assert flexible.input_schema["properties"]["extra"]["type"] == "object"


def test_tool_explicit_input_schema_bypasses_inference():
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string", "description": "query"}},
        "required": ["q"],
        "additionalProperties": False,
    }

    @tool(name="Custom", description="custom schema", input_schema=schema)
    async def custom(q: str) -> str:  # pragma: no cover - signature unused
        return q

    assert custom.input_schema is schema
    assert custom.schema_info is None


def test_tool_sync_function_supported():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    assert multiply.input_schema["required"] == ["a", "b"]


@pytest.mark.asyncio
async def test_tool_execute_sync_function():
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    result = await multiply.execute(a=3, b=4)
    assert result.content == "12"
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_execute():
    @tool(name="Add", description="add two numbers")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    result = await add.execute(a=1, b=2)
    assert result.content == "3"
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_execute_applies_defaults():
    @tool(name="Add", description="add two numbers")
    async def add(a: int, b: int = 10) -> str:
        return str(a + b)

    result = await add.execute(a=1)
    assert result.content == "11"


@pytest.mark.asyncio
async def test_tool_execute_validates_arguments():
    @tool(name="Add", description="add two numbers")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    result = await add.execute(a="not-a-number", b=2)
    assert result.is_error
    assert "Invalid arguments for tool Add" in result.content


@pytest.mark.asyncio
async def test_tool_execute_captures_exception():
    @tool(name="Fail", description="always fails")
    async def fail() -> str:
        raise ValueError("boom")

    result = await fail.execute()
    assert result.is_error
    assert "ValueError: boom" in result.content


@pytest.mark.asyncio
async def test_tool_failure_error_function_custom():
    def custom_error(exc: Exception) -> str:
        return f"custom failure: {exc}"

    @tool(name="Fail", description="always fails", failure_error_function=custom_error)
    async def fail() -> str:
        raise RuntimeError("exploded")

    result = await fail.execute()
    assert result.is_error
    assert result.content == "custom failure: exploded"


@pytest.mark.asyncio
async def test_tool_failure_error_function_none_reraises():
    @tool(name="Fail", description="always fails", failure_error_function=None)
    async def fail() -> str:
        raise RuntimeError("propagated")

    with pytest.raises(RuntimeError, match="propagated"):
        await fail.execute()


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
