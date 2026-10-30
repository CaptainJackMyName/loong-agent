"""Bash tool backed by an async subprocess."""

from __future__ import annotations

import asyncio

from loong_agent.tools.base import tool


@tool(name="Bash", description="Execute a shell command and return stdout/stderr.")
async def bash(command: str, timeout: float = 120.0, cwd: str | None = None) -> str:
    """Execute a shell command and return its stdout/stderr.

    Args:
        command: The shell command to execute.
        timeout: Maximum seconds to wait for the command to finish.
        cwd: Optional working directory for the command.
    """
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Error: command timed out after {timeout}s"

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out.strip():
            parts.append(out.rstrip())
        if err.strip():
            parts.append(f"[stderr]\n{err.rstrip()}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
    except Exception as exc:  # noqa: BLE001
        return f"Error executing command: {exc}"


__all__ = ["bash"]
