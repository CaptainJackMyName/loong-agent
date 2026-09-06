"""Filesystem tools: Read, Write, Edit, Grep, Glob."""

from __future__ import annotations

import re

from pathlib import Path
from typing import List, Optional

from loong_agent.tools.base import tool


@tool("Read", "Read a file from the local filesystem with optional line offset/limit.")
async def read_file(
    file_path: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
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


@tool("Write", "Write a file to the local filesystem, creating parent directories as needed.")
async def write_file(file_path: str, content: str) -> str:
    path = Path(file_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Error writing {file_path}: {exc}"
    return f"Successfully wrote {len(content)} characters to {file_path}."


@tool("Edit", "Perform an exact string replacement in an existing file.")
async def edit_file(file_path: str, old_string: str, new_string: str) -> str:
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


@tool("Grep", "Search file contents for a regular expression pattern.")
async def grep(pattern: str, path: str = ".", glob: Optional[str] = None) -> str:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regex: {exc}"

    root = Path(path).expanduser()
    matches: List[str] = []
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


@tool("Glob", "Find files matching a glob pattern.")
async def glob_files(pattern: str, path: str = ".") -> str:
    root = Path(path).expanduser()
    matches = sorted(str(p) for p in root.glob(pattern) if not p.name.startswith("."))
    if not matches:
        return "(no matches)"
    return "\n".join(matches[:200])


__all__ = ["read_file", "write_file", "edit_file", "grep", "glob_files"]
