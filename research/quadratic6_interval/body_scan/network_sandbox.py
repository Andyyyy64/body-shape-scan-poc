"""Fail-closed process network isolation for macOS and Linux/WSL2."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys

from .io import InputError


def sandbox_command(kind: str, command: list[str]) -> list[str]:
    if kind == "macos" and sys.platform == "darwin":
        executable = shutil.which("sandbox-exec")
        if executable:
            return [
                executable,
                "-p",
                "(version 1)(allow default)(deny network*)",
                *command,
            ]
    elif kind == "linux" and sys.platform == "linux":
        executable = shutil.which("unshare")
        if executable:
            # Kill the inference child when the web server terminates its supervisor.
            return [
                executable,
                "--user",
                "--map-root-user",
                "--net",
                "--fork",
                "--kill-child",
                "--",
                *command,
            ]
    raise InputError(
        "Configured process network isolation is unavailable; no fallback was used"
    )


def verify_network_sandbox(kind: str, python: str) -> None:
    # A listening host socket makes the check distinguish isolation from a closed port.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        probe = (
            "import socket,sys\n"
            f"targets = [('127.0.0.1', {port}), ('1.1.1.1', 443)]\n"
            "for target in targets:\n"
            "    try:\n"
            "        connection = socket.create_connection(target, timeout=1)\n"
            "    except OSError:\n"
            "        continue\n"
            "    connection.close()\n"
            "    sys.exit(1)\n"
        )
        try:
            result = subprocess.run(
                sandbox_command(kind, [python, "-c", probe]),
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InputError(
                "Process network isolation probe failed; model execution is disabled"
            ) from exc
        if result.returncode != 0:
            raise InputError(
                "Process network isolation probe failed; model execution is disabled"
            )
