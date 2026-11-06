"""Headless Colab runtime tools backed by the official Colab CLI."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

Accelerator = Literal["T4", "L4", "A100", "H100", "G4"]
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_REMOTE_CONTENT = re.compile(r"^/content(?:/[^/]+)*$")
_OUTPUT_LIMIT = 8_000


def _command_prefix() -> list[str]:
    return shlex.split(os.environ.get("COLAB_CLI_COMMAND", "colab"))


def _validate_session(session: str) -> str:
    if not _NAME.fullmatch(session):
        raise ValueError("session must be 1-64 letters, numbers, '_' or '-'")
    return session


def _validate_remote(path: str) -> str:
    if not _REMOTE_CONTENT.fullmatch(path):
        raise ValueError("remote paths must be absolute files below /content")
    return path


def _receipt(command: list[str], returncode: int, stdout: str, stderr: str) -> dict:
    def redact(value: str) -> str:
        value = re.sub(r"(?i)(token|authorization)=?[^\s&]+", r"\1=[redacted]", value)
        return value[-_OUTPUT_LIMIT:]

    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "command": command[1:],
        "stdout": redact(stdout),
        "stderr": redact(stderr),
    }


async def _run(*args: str) -> dict:
    command = [*_command_prefix(), *args]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return _receipt(
        command, process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
    )


def _shell_script(command: str) -> str:
    """Build a remote wrapper which preserves shell exit status and output."""
    return f"""import subprocess
result = subprocess.run(
    ['bash', '-lc', {command!r}], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
print(result.stdout, end='')
raise SystemExit(result.returncode)
"""


def register_headless_tools(mcp: FastMCP) -> None:
    """Register runtime tools that do not depend on the browser proxy."""

    @mcp.tool(name="colab_headless_start")
    async def start(
        session: str, accelerator: Accelerator = "L4", notebook_url: str | None = None
    ) -> dict:
        """Allocate a named runtime. notebook_url is metadata, not runtime adoption."""
        result = await _run("new", "--session", _validate_session(session), "--gpu", accelerator)
        result["notebook_url"] = notebook_url
        result["adopted_notebook_runtime"] = False
        return result

    @mcp.tool(name="colab_headless_status")
    async def status(session: str) -> dict:
        """Return a named runtime's current state."""
        return await _run("status", "--session", _validate_session(session))

    @mcp.tool(name="colab_headless_upload")
    async def upload(session: str, local_path: str, remote_path: str) -> dict:
        """Upload one local regular file to /content."""
        local = Path(local_path).expanduser().resolve()
        if not local.is_file():
            raise ValueError("local_path must name an existing regular file")
        return await _run("upload", "--session", _validate_session(session), str(local), _validate_remote(remote_path))

    @mcp.tool(name="colab_headless_execute")
    async def execute(session: str, script_path: str, timeout_seconds: int = 3600) -> dict:
        """Execute a local Python script in the named runtime."""
        script = Path(script_path).expanduser().resolve()
        if not script.is_file() or script.suffix != ".py":
            raise ValueError("script_path must name an existing .py file")
        if not 1 <= timeout_seconds <= 14_400:
            raise ValueError("timeout_seconds must be between 1 and 14400")
        return await _run("exec", "--session", _validate_session(session), "--file", str(script), "--timeout", str(timeout_seconds))

    @mcp.tool(name="colab_headless_shell")
    async def shell(session: str, command: str, timeout_seconds: int = 600) -> dict:
        """Run one bash command in the named runtime and return its receipt."""
        if not command.strip() or len(command) > 16_000:
            raise ValueError("command must be non-empty and at most 16000 characters")
        if not 1 <= timeout_seconds <= 14_400:
            raise ValueError("timeout_seconds must be between 1 and 14400")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="colab-mcp-shell-", delete=False
        ) as handle:
            script = Path(handle.name)
            handle.write(_shell_script(command))
        try:
            return await _run(
                "exec", "--session", _validate_session(session), "--file", str(script),
                "--timeout", str(timeout_seconds),
            )
        finally:
            script.unlink(missing_ok=True)

    @mcp.tool(name="colab_headless_download")
    async def download(session: str, remote_path: str, local_path: str) -> dict:
        """Download one /content file from the named runtime."""
        local = Path(local_path).expanduser().resolve()
        local.parent.mkdir(parents=True, exist_ok=True)
        return await _run("download", "--session", _validate_session(session), _validate_remote(remote_path), str(local))

    @mcp.tool(name="colab_headless_stop")
    async def stop(session: str) -> dict:
        """Release the named Colab runtime."""
        return await _run("stop", "--session", _validate_session(session))
