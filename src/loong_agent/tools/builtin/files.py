"""Filesystem tools: Read, Write, Edit, Grep, Glob."""

from __future__ import annotations

import re
from pathlib import Path

from loong_agent.tools.base import tool


@tool(
    name="Read",
    description="Read a file from the local filesystem with optional line offset/limit.",
)
async def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a file from the local filesystem.

    Args:
        file_path: Path of the file to read.
        offset: 1-based line number to start reading from.
        limit: Maximum number of lines to return.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"Error: file not found: {file_path}"
    if path.is_dir():
        return f"Error: path is a directory: {file_path}"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {file_path}: {exc}"

    lines = text.splitlines()
    start = (offset or 1) - 1
    start = max(0, start)
    if limit is not None:
        lines = lines[start : start + limit]
    else:
        lines = lines[start:]

    if len(lines) == 0 and limit:
        return "(empty)"
    return "\n".join(lines)


@tool(
    name="Write",
    description="Write a file to the local filesystem, creating parent directories as needed.",
)
async def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Args:
        file_path: Path of the file to write.
        content: Full text content to write to the file.
    """
    path = Path(file_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error writing {file_path}: {exc}"
    return f"Successfully wrote {len(content)} characters to {file_path}."


@tool(name="Edit", description="Perform an exact string replacement in an existing file.")
async def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace an exact string in an existing file.

    Args:
        file_path: Path of the file to edit.
        old_string: Exact text to replace; must appear exactly once.
        new_string: Replacement text.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"Error: file not found: {file_path}"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {file_path}: {exc}"

    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {file_path}."
    if count > 1:
        return f"Error: old_string is not unique ({count} occurrences). Provide more context."
    try:
        path.write_text(text.replace(old_string, new_string), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error writing {file_path}: {exc}"
    return f"Successfully edited {file_path}."


@tool(name="Grep", description="Search file contents for a regular expression pattern.")
async def grep(pattern: str, path: str = ".", glob: str | None = None) -> str:
    """Search file contents for a regular expression pattern.

    Args:
        pattern: Regular expression to search for.
        path: Directory or file to search in.
        glob: Optional glob pattern restricting which files are searched.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    root = Path(path).expanduser()
    matches: list[str] = []
    if root.is_file():
        files = [root]
    else:
        glob_pattern = glob or "*"
        files = [p for p in root.rglob(glob_pattern) if p.is_file()]

    for file_path in files:
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:  # noqa: BLE001
            continue
        for idx, line in enumerate(lines, start=1):
            if compiled.search(line):
                matches.append(f"{file_path}:{idx}:{line}")

    if not matches:
        return "(no matches)"
    return "\n".join(matches[:200])


@tool(name="Glob", description="Find files matching a glob pattern.")
async def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern such as ``*.py``.
        path: Directory to search in.
    """
    root = Path(path).expanduser()
    matches = sorted(str(p) for p in root.glob(pattern) if not p.name.startswith("."))
    if not matches:
        return "(no matches)"
    return "\n".join(matches[:200])


__all__ = ["read_file", "write_file", "edit_file", "grep", "glob_files"]
