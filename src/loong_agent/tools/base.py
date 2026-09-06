"""Tool abstractions.

A :class:`Tool` wraps a callable and carries enough metadata (name,
description, JSON Schema) for the LLM to invoke it. Tools are defined either as
a class or via the :func:`tool` decorator.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from loong_agent.llm.base import ToolDefinition
from loong_agent.types.messages import ToolResult

ToolHandler = Callable[..., Union[Any, Awaitable[Any]]]


@dataclass
class Tool:
    """A concrete tool instance."""

    name: str
    description: str
    handler: ToolHandler
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = self.handler(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            content = result if isinstance(result, str) else _stringify(result)
            return ToolResult(tool_use_id="", name=self.name, content=content)
        except Exception as exc:  # noqa: BLE001 - surface any error to the model
            return ToolResult(
                tool_use_id="",
                name=self.name,
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    import json

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def tool(
    name: str,
    description: str,
    input_schema: Optional[Dict[str, Any]] = None,
) -> Callable[[ToolHandler], Tool]:
    """Decorator turning an async function into a :class:`Tool`.

    Example::

        @tool("search_web", "Search the web for information")
        async def search_web(query: str, max_results: int = 10) -> str:
            ...
    """

    def _decorator(handler: ToolHandler) -> Tool:
        schema = input_schema if input_schema is not None else _infer_schema(handler)
        return Tool(name=name, description=description, handler=handler, input_schema=schema)

    return _decorator


def _infer_schema(handler: ToolHandler) -> Dict[str, Any]:
    """Build a best-effort JSON Schema from the handler signature."""
    sig = inspect.signature(handler)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in sig.parameters.values():
        if param.name in {"self", "cls"}:
            continue
        typ = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        properties[param.name] = _annotation_to_schema(typ)
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _annotation_to_schema(annotation: Any) -> Dict[str, Any]:
    import json
    import typing

    origin = typing.get_origin(annotation)
    if origin is Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]  # noqa: E721
        if non_none:
            return _annotation_to_schema(non_none[0])
    if annotation in (str, "str"):
        return {"type": "string"}
    if annotation in (int, "int"):
        return {"type": "integer"}
    if annotation in (float, "float"):
        return {"type": "number"}
    if annotation in (bool, "bool"):
        return {"type": "boolean"}
    if origin is list:
        return {"type": "array", "items": {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    # Fall back to anything JSON-serializable.
    return {"type": "string"}


__all__ = ["Tool", "tool", "ToolHandler"]
