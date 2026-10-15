import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from body_scan.io import write_json
from body_scan.synthetic_lab import launch


class FrozenSourceLaunchTests(unittest.TestCase):
    def test_mutable_working_directory_cannot_shadow_frozen_worker_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed, mutable = root / "frozen", root / "mutable"
            for path, label in [(fixed, "FROZEN"), (mutable, "MUTABLE")]:
                package = path / "body_scan"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("")
                (package / "probe.py").write_text(
                    f"print({label!r})\nprint(__file__)\n"
                )
            config = root / "config.json"
            write_json(
                config,
                {
                    "python": sys.executable,
                    "network_sandbox": "linux",
                    "cpu_threads": 1,
                },
            )
            command = ["-m", "body_scan.probe"]
            legacy = subprocess.run(
                [sys.executable, *command],
                cwd=mutable,
                env={"PYTHONPATH": str(fixed)},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(legacy.stdout.startswith("MUTABLE\n"))
            previous = Path.cwd()
            try:
                os.chdir(mutable)
                with (
                    patch(
                        "body_scan.synthetic_lab.__file__",
                        str(fixed / "body_scan/synthetic_lab.py"),
                    ),
                    patch("body_scan.synthetic_lab.verify_network_sandbox"),
                    patch(
                        "body_scan.synthetic_lab.sandbox_command",
                        side_effect=lambda kind, command: command,
                    ),
                ):
                    code = launch(str(config), command, root / "worker.log")
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            lines = (root / "worker.log").read_text().splitlines()
            self.assertEqual(lines[0], "FROZEN")
            self.assertEqual(Path(lines[1]).resolve(), fixed / "body_scan/probe.py")
