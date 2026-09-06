"""Publication allowlist and obvious leak checks. Human review is still required."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath

ALLOWED_ROOT = {
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "LICENSE",
    "AGENTS.md",
}


def allowed(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        path in ALLOWED_ROOT
        or (
            len(p.parts) == 2
            and p.parts[0] in {"body_scan", "tests"}
            and p.suffix == ".py"
        )
        or (len(p.parts) == 2 and p.parts[0] == "docs" and p.suffix == ".md")
        or (
            len(p.parts) == 3
            and p.parts[:2] == (".github", "workflows")
            and p.suffix == ".yml"
        )
    )


def inspect_blob(path: str, data: bytes) -> list[str]:
    reasons = []
    if not allowed(path):
        reasons.append("file is outside publication allowlist")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return reasons + ["binary content"]
    if len(data) > 1_000_000:
        reasons.append("unexpectedly large public file")
    patterns = [
        r"/(?:Users|home)/[A-Za-z0-9_-]+/",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}",
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        r"(?i)(?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*[\"\'][^\"\']{12,}[\"\']",
    ]
    if any(re.search(pattern, text) for pattern in patterns):
        reasons.append("possible private path or credential")
    return reasons


def audit(staged: bool, history: bool) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args])

    files = git("ls-files", "-z").decode().split("\0")
    failures, checked = [], 0
    for path in filter(None, files):
        data = git("show", ":" + path) if staged else Path(path).read_bytes()
        errors = inspect_blob(path, data)
        if errors:
            failures.append({"file": path, "reasons": errors})
        checked += 1
    commits = 0
    if history:
        for commit in git("rev-list", "--all").decode().splitlines():
            commits += 1
            identity = (
                git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", commit)
                .decode()
                .strip()
                .split("\0")
            )
            if (
                identity
                != ["Research Contributors", "contributors@users.noreply.github.com"]
                * 2
            ):
                failures.append(
                    {"commit": commit, "reasons": ["non-anonymous commit identity"]}
                )
            for path in filter(
                None,
                git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0"),
            ):
                errors = inspect_blob(path, git("show", commit + ":" + path))
                if errors:
                    failures.append({"commit": commit, "file": path, "reasons": errors})
    return {
        "status": "fail" if failures else "pass",
        "checked_files": checked,
        "checked_commits": commits,
        "failures": failures,
        "limitation": "Allowlist and pattern checks do not prove anonymization; manually review public text and Git metadata.",
    }
