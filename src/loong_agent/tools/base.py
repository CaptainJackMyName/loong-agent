"""Tool abstractions.

A :class:`Tool` wraps a callable and carries enough metadata (name,
description, JSON Schema) for the LLM to invoke it. Tools are defined either as
a class or via the :func:`tool` decorator.

The :func:`tool` decorator is modeled after the ``@function_tool`` decorator of
the ``openai-agents`` SDK. By default it will:

1. Parse the function signature (via pydantic) to create a JSON Schema for the
   tool's parameters, including ``Annotated`` metadata and ``Field`` constraints.
2. Use the function's docstring to populate the tool's description.
3. Use the function's docstring to populate argument descriptions
   (google/numpy/sphinx styles are auto-detected).
4. Run synchronous handlers in a worker thread via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Literal,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from loong_agent.llm.base import ToolDefinition
from loong_agent.types.messages import ToolResult

ToolHandler = Callable[..., Any | Awaitable[Any]]
"""A tool handler: sync or async callable invoked with validated arguments."""

ToolErrorFunction = Callable[[Exception], str | Awaitable[str]]
"""Generates the model-visible error message for a failed tool invocation."""

DocstringStyle = Literal["google", "numpy", "sphinx"]

#: Sentinel: use the default failure behavior (``"{type}: {message}"`` result).
_UNSET_FAILURE_ERROR_FUNCTION = object()


# ---------------------------------------------------------------------------
# Strict JSON Schema
# ---------------------------------------------------------------------------

_EMPTY_SCHEMA: dict[str, Any] = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Mutate ``schema`` to conform to the ``strict`` standard the OpenAI API expects.

    Every object gets ``additionalProperties: false`` and a ``required`` list
    covering all declared properties; nested ``$defs``, ``properties``, ``items``
    and ``anyOf`` subschemas are processed recursively.
    """
    if not schema:
        return copy.deepcopy(_EMPTY_SCHEMA)
    _ensure_strict_json_schema(schema)
    return schema


def _ensure_strict_json_schema(json_schema: Any) -> None:
    if not isinstance(json_schema, dict):
        return

    defs = json_schema.get("$defs")
    if isinstance(defs, dict):
        for def_schema in defs.values():
            _ensure_strict_json_schema(def_schema)

    typ = json_schema.get("type")
    properties = json_schema.get("properties")
    if typ is None and isinstance(properties, dict):
        typ = json_schema["type"] = "object"

    if typ == "object":
        json_schema["additionalProperties"] = False

    if isinstance(properties, dict):
        json_schema["required"] = list(properties.keys())
        for prop_schema in properties.values():
            _ensure_strict_json_schema(prop_schema)

    items = json_schema.get("items")
    if isinstance(items, dict):
        _ensure_strict_json_schema(items)

    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        for variant in any_of:
            _ensure_strict_json_schema(variant)

    # `None` defaults are meaningless in strict mode; the schema stays nullable.
    if "default" in json_schema and json_schema["default"] is None:
        json_schema.pop("default")


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

@dataclass
class FuncDocumentation:
    """Metadata about a Python function, extracted from its docstring."""

    name: str
    """The name of the function, via ``__name__``."""

    description: str | None
    """The description of the function, derived from the docstring."""

    param_descriptions: dict[str, str] | None
    """The parameter descriptions of the function, derived from the docstring."""


# As of Feb 2025, the automatic style detection in griffe is an Insiders feature.
# This code approximates it.
def _detect_docstring_style(doc: str) -> DocstringStyle:
    scores: dict[str, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters' followed by a dashed underline.
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon.
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    # Priority order in case of tie: sphinx > numpy > google.
    for style in ("sphinx", "numpy", "google"):
        if scores[style] == max_score:
            return cast(DocstringStyle, style)
    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    # Suppresses warnings about missing annotations for params.
    logger = logging.getLogger("griffe")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


_GOOGLE_SECTION_HEADER_RE = re.compile(
    r"^(args|arguments|params|parameters):\s*$",
    re.IGNORECASE,
)


def _ensure_blank_line_before_google_sections(doc: str) -> str:
    """Insert a blank line before a Google-style ``Args:`` header that directly
    follows a non-blank line. griffe's Google parser silently skips such headers,
    which would leak the raw ``Args:`` block into the description.
    """
    lines = doc.splitlines()
    output: list[str] = []
    inserted = False
    for index, line in enumerate(lines):
        if (
            index > 0
            and _GOOGLE_SECTION_HEADER_RE.match(line)
            and output
            and output[-1].strip()
            and index + 1 < len(lines)
            and lines[index + 1].startswith((" ", "\t"))
        ):
            output.append("")
            inserted = True
        output.append(line)

    if not inserted:
        return doc
    return "\n".join(output)


def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """Extract metadata from a function docstring for use as an LLM tool.

    Args:
        func: The function to extract documentation from.
        style: The docstring style to parse with. If not provided, the style is
            auto-detected.

    Returns:
        A :class:`FuncDocumentation` with the function's name, description, and
        parameter descriptions.
    """
    name = func.__name__
    doc = inspect.getdoc(func)
    if not doc:
        return FuncDocumentation(name=name, description=None, param_descriptions=None)

    resolved_style = style or _detect_docstring_style(doc)
    if resolved_style == "google":
        doc = _ensure_blank_line_before_google_sections(doc)

    with _suppress_griffe_logging():
        # Griffe runs cleandoc again. A leading newline keeps the already-cleaned
        # first line in its indentation calculation, preserving a leading Args: block.
        docstring = Docstring("\n" + doc, lineno=1, parser=resolved_style)
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )

    param_descriptions: dict[str, str] = {
        # Variadic parameters are written with stars ("*args:", "**kwargs:") in
        # docstrings; griffe returns those names verbatim, so strip them.
        param.name.lstrip("*"): param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FuncDocumentation(
        name=name,
        description=description,
        param_descriptions=param_descriptions or None,
    )


# ---------------------------------------------------------------------------
# Function schema
# ---------------------------------------------------------------------------

@dataclass
class FuncSchema:
    """Captures the schema for a Python function, in preparation for sending it
    to an LLM as a tool."""

    name: str
    """The name of the function."""

    description: str | None
    """The description of the function."""

    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""

    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""

    signature: inspect.Signature
    """The signature of the function."""

    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode."""

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """Convert validated Pydantic data into ``(args, kwargs)`` suitable for
        calling the original function."""
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False
        # Read instance storage first so Pydantic properties such as ``model_extra``
        # do not shadow tool parameters of the same name. ``model_dump()`` is
        # unsuitable here because it converts nested models to dicts.
        instance_values = object.__getattribute__(data, "__dict__")

        for name, param in self.signature.parameters.items():
            value = instance_values[name] if name in instance_values else getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args and mark that *args is now seen.
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, use keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # KEYWORD_ONLY parameters always use keyword args.
                keyword_args[name] = value
        return positional_args, keyword_args


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the underlying annotation and any metadata from ``typing.Annotated``."""
    metadata: tuple[Any, ...] = ()
    ann = annotation

    while get_origin(ann) is Annotated:
        args = get_args(ann)
        if not args:
            break
        ann = args[0]
        metadata = (*metadata, *args[1:])

    return ann, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extract a human readable description from ``Annotated`` metadata if present."""
    for item in metadata:
        if isinstance(item, str):
            return item
    return None


def function_schema(
    func: Callable[..., Any],
    *,
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """Given a Python function, extract a :class:`FuncSchema` capturing the name,
    description, parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The docstring style to parse with. If not provided, the
            style is auto-detected.
        name_override: If provided, use this name instead of the function's ``__name__``.
        description_override: If provided, use this description instead of the one
            derived from the docstring.
        use_docstring_info: If True, use the docstring to generate the description
            and parameter descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode.

    Returns:
        A :class:`FuncSchema` object.
    """
    # 1. Grab docstring info.
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = dict(doc_info.param_descriptions or {})
    else:
        doc_info = None
        param_descs = {}

    try:
        type_hints_with_extras = get_type_hints(func, include_extras=True)
    except NameError as exc:
        raise ValueError(
            f"Failed to resolve type hints for {func.__name__}: {exc}. Make sure all "
            "annotations are importable from the function's module."
        ) from exc

    type_hints: dict[str, Any] = {}
    param_metadata: dict[str, tuple[Any, ...]] = {}

    for name, annotation in type_hints_with_extras.items():
        if name == "return":
            continue

        stripped_ann, metadata = _strip_annotated(annotation)
        type_hints[name] = stripped_ann
        param_metadata[name] = metadata

        description = _extract_description_from_metadata(metadata)
        if description is not None:
            param_descs.setdefault(name, description)

    # Ensure name_override takes precedence even if docstring info is disabled.
    func_name = name_override or (doc_info.name if doc_info else func.__name__)

    # 2. Inspect the signature and build pydantic field definitions.
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}
    model_config = ConfigDict()

    for name, param in sig.parameters.items():
        ann = type_hints.get(name, param.annotation)

        # If there's no type hint, assume `Any`.
        if ann is inspect._empty:
            ann = Any

        field_description = param_descs.get(name)

        # Let pydantic retain other Annotated metadata, including constrained
        # types and validators, in its original order.
        field_info_from_annotated = (
            FieldInfo.from_annotation(type_hints_with_extras[name])
            if param_metadata.get(name)
            else None
        )

        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: int -> List[int]; a homogeneous tuple describes each item.
            if get_origin(ann) is tuple:
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = args_of_tuple[0]
                elif hasattr(ann, "__args__"):
                    raise ValueError(
                        f"Variadic parameter `*{name}` in function {func.__name__} is annotated"
                        f" with the fixed-length tuple `{ann}`. A variadic annotation describes"
                        " each positional argument, so use tuple[T, ...] or list[T] instead."
                    )
            ann = list[ann]  # type: ignore[valid-type]
            fields[name] = (ann, Field(default_factory=list, description=field_description))

        elif param.kind == param.VAR_KEYWORD:
            if strict_json_schema:
                raise ValueError(
                    f"Variadic keyword parameter `**{name}` in function {func.__name__} cannot"
                    " use a strict schema. Set strict_mode=False on the tool (or"
                    " strict_json_schema=False on function_schema) and pass keyword arguments"
                    f" in the nested `{name}` object, or use explicit parameters for a strict"
                    " tool."
                )
            # Reject flat keyword arguments instead of silently dropping them.
            model_config["extra"] = "forbid"
            # **kwargs: X applies to each keyword value (PEP 484), so collect as dict[str, X].
            ann = dict[str, ann]  # type: ignore[valid-type]
            fields[name] = (ann, Field(default_factory=dict, description=field_description))

        else:
            # Normal parameter.
            if field_info_from_annotated is not None:
                merged = FieldInfo.merge_field_infos(
                    field_info_from_annotated,
                    description=field_description or field_info_from_annotated.description,
                )
                if param.default is not inspect._empty and not isinstance(
                    param.default, FieldInfo
                ):
                    merged = FieldInfo.merge_field_infos(merged, default=param.default)
                fields[name] = (ann, merged)
            elif param.default is inspect._empty:
                # Required field.
                fields[name] = (ann, Field(..., description=field_description))
            elif isinstance(param.default, FieldInfo):
                # Parameter with a default value that is a Field(...).
                fields[name] = (
                    ann,
                    FieldInfo.merge_field_infos(
                        param.default, description=field_description or param.default.description
                    ),
                )
            else:
                # Parameter with a default value.
                fields[name] = (ann, Field(default=param.default, description=field_description))

    # 3. Dynamically build a Pydantic model.
    dynamic_model = create_model(
        f"{func_name}_args", __base__=BaseModel, __config__=model_config, **fields
    )

    # 4. Build the JSON schema from that model.
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # 5. Return as a FuncSchema dataclass.
    return FuncSchema(
        name=func_name,
        # Ensure description_override takes precedence even if docstring info is disabled.
        description=description_override or (doc_info.description if doc_info else None),
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        strict_json_schema=strict_json_schema,
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A concrete tool instance."""

    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=dict)
    schema_info: FuncSchema | None = field(default=None, repr=False, compare=False)
    """Populated by :func:`tool`; enables argument validation and positional binding."""
    failure_error_function: Any = field(
        default=_UNSET_FAILURE_ERROR_FUNCTION, repr=False, compare=False
    )
    """Custom error handler. ``None`` re-raises, ``_UNSET`` uses the default message."""

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    def _bind_arguments(self, kwargs: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        """Validate model-supplied arguments and bind them to the handler signature."""
        schema = self.schema_info
        if schema is None:
            # Tool constructed directly (e.g. with an explicit input_schema):
            # pass arguments through without extra validation.
            return [], kwargs
        try:
            parsed = schema.params_pydantic_model(**kwargs)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"Invalid arguments for tool {self.name}: {exc}"
            ) from exc
        return schema.to_call_args(parsed)

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            try:
                args, call_kwargs = self._bind_arguments(kwargs)
            except ToolArgumentError as exc:
                return ToolResult(
                    tool_use_id="",
                    name=self.name,
                    content=str(exc),
                    is_error=True,
                )

            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(*args, **call_kwargs)
            else:
                # Run sync handlers in a worker thread so they don't block the loop.
                result = await asyncio.to_thread(self.handler, *args, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result

            content = result if isinstance(result, str) else _stringify(result)
            return ToolResult(tool_use_id="", name=self.name, content=content)
        except Exception as exc:  # noqa: BLE001 - surface any error to the model
            if self.failure_error_function is None:
                raise
            if self.failure_error_function is _UNSET_FAILURE_ERROR_FUNCTION:
                content = f"{type(exc).__name__}: {exc}"
            else:
                content = self.failure_error_function(exc)
                if inspect.isawaitable(content):
                    content = await content
            return ToolResult(
                tool_use_id="",
                name=self.name,
                content=content,
                is_error=True,
            )


class ToolArgumentError(ValueError):
    """Raised when model-supplied tool arguments fail validation."""


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


# ---------------------------------------------------------------------------
# tool decorator
# ---------------------------------------------------------------------------

@overload
def tool(
    func: ToolHandler,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    strict_mode: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Tool:
    """Overload for usage as @tool (no parentheses)."""
    ...


@overload
def tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    strict_mode: bool = True,
    failure_error_function: ToolErrorFunction | None = None,
    input_schema: dict[str, Any] | None = None,
) -> Callable[[ToolHandler], Tool]:
    """Overload for usage as @tool(...)."""
    ...


def tool(
    func: ToolHandler | None = None,
    /,
    *legacy_args: Any,
    name: str | None = None,
    description: str | None = None,
    docstring_style: DocstringStyle | None = None,
    use_docstring_info: bool = True,
    strict_mode: bool = True,
    failure_error_function: ToolErrorFunction | None | object = (
        _UNSET_FAILURE_ERROR_FUNCTION
    ),
    input_schema: dict[str, Any] | None = None,
) -> Tool | Callable[[ToolHandler], Tool]:
    """Decorator to create a :class:`Tool` from a function.

    Supports bare (``@tool``) and parameterized (``@tool(...)``) usage. By default:

    1. The function signature is parsed to create a JSON schema for the tool's
       parameters (via pydantic, honoring ``Annotated`` metadata and ``Field``).
    2. The function's docstring populates the tool's description.
    3. The function's docstring populates argument descriptions; the docstring
       style is auto-detected but can be overridden.

    Example::

        @tool
        async def get_weather(city: Annotated[str, "The city name"]) -> str:
            \"\"\"Get the current weather for a city.\"\"\"
            ...

        @tool(name="search_web", strict_mode=False)
        def search_web(query: str, max_results: int = 10, **filters: Any) -> str:
            \"\"\"Search the web for information.

            Args:
                query: The search query.
                max_results: Maximum number of results to return.
            \"\"\"
            ...

    Args:
        func: The function to wrap (only set when using bare ``@tool``).
        name: If provided, use this name for the tool instead of the function's name.
        description: If provided, use this description for the tool instead of the
            function's docstring.
        docstring_style: If provided, use this style for docstring parsing. If not
            provided, the style is auto-detected.
        use_docstring_info: If True, use the docstring to populate the tool's
            description and argument descriptions.
        strict_mode: Whether to enable strict mode for the tool's JSON schema. We
            strongly recommend setting this to True, as it increases the likelihood
            of correct JSON input. If False, non-strict JSON schemas are allowed
            (e.g. ``**kwargs`` parameters, defaults become optional).
        failure_error_function: If provided, use this function to generate an error
            message when the tool call fails; the message is sent to the LLM. Pass
            None to re-raise exceptions instead of returning an error result.
        input_schema: Escape hatch providing an explicit JSON schema, bypassing
            automatic schema generation (arguments are then passed through
            unvalidated).
    """
    if legacy_args:
        raise TypeError(
            "tool() no longer accepts positional arguments like @tool('name', 'desc'). "
            "Use @tool, @tool(name=..., description=...), or construct Tool(...) directly."
        )
    if func is not None:
        if not callable(func):
            raise TypeError(
                "tool() no longer accepts positional arguments like @tool('name', 'desc'). "
                "Use @tool, @tool(name=..., description=...), or construct Tool(...) directly."
            )
        return _create_tool(
            func,
            name=name,
            description=description,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_mode=strict_mode,
            failure_error_function=failure_error_function,
            input_schema=input_schema,
        )

    def _decorator(handler: ToolHandler) -> Tool:
        return _create_tool(
            handler,
            name=name,
            description=description,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_mode=strict_mode,
            failure_error_function=failure_error_function,
            input_schema=input_schema,
        )

    return _decorator


def _create_tool(
    handler: ToolHandler,
    *,
    name: str | None,
    description: str | None,
    docstring_style: DocstringStyle | None,
    use_docstring_info: bool,
    strict_mode: bool,
    failure_error_function: Any,
    input_schema: dict[str, Any] | None,
) -> Tool:
    fallback_name = getattr(handler, "__name__", None) or type(handler).__name__

    if input_schema is not None:
        # Explicit escape hatch: no automatic schema or validation.
        schema_info: FuncSchema | None = None
        final_schema = input_schema
        final_name = name or fallback_name
        doc_description = (
            generate_func_documentation(handler, docstring_style).description
            if use_docstring_info
            else None
        )
    else:
        schema_info = function_schema(
            handler,
            name_override=name,
            description_override=description,
            docstring_style=docstring_style,
            use_docstring_info=use_docstring_info,
            strict_json_schema=strict_mode,
        )
        final_schema = schema_info.params_json_schema
        final_name = schema_info.name
        doc_description = schema_info.description if use_docstring_info else None

    return Tool(
        name=final_name,
        description=description or doc_description or "",
        handler=handler,
        input_schema=final_schema,
        schema_info=schema_info,
        failure_error_function=failure_error_function,
    )


__all__ = [
    "DocstringStyle",
    "FuncDocumentation",
    "FuncSchema",
    "Tool",
    "ToolArgumentError",
    "ToolErrorFunction",
    "ToolHandler",
    "ensure_strict_json_schema",
    "function_schema",
    "generate_func_documentation",
    "tool",
]
