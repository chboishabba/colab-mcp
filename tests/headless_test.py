import asyncio

import pytest

from colab_mcp import headless


def test_validates_session_names():
    assert headless._validate_session("whisperx-l4_1") == "whisperx-l4_1"
    with pytest.raises(ValueError):
        headless._validate_session("../bad")


def test_restricts_remote_paths_to_content():
    assert headless._validate_remote("/content/file.json") == "/content/file.json"
    with pytest.raises(ValueError):
        headless._validate_remote("/etc/passwd")


def test_shell_script_preserves_command_and_exit_status():
    script = headless._shell_script("python -m compileall /content")
    assert "python -m compileall /content" in script
    assert "raise SystemExit(result.returncode)" in script


@pytest.mark.asyncio
async def test_run_returns_redacted_receipt(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self):
            return b"token=supersecret ok", b""

    async def create(*args, **kwargs):
        assert args == ("colab", "status", "--session", "test")
        return Process()

    monkeypatch.delenv("COLAB_CLI_COMMAND", raising=False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    result = await headless._run("status", "--session", "test")
    assert result["ok"] is True
    assert "supersecret" not in result["stdout"]
