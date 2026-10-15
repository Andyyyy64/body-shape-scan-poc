import os
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from body_scan.devices import configure_device
from body_scan.io import InputError
from body_scan.network_sandbox import sandbox_command, verify_network_sandbox


class DeviceTests(unittest.TestCase):
    def test_unavailable_cuda_does_not_fall_back(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(InputError, "no fallback"):
            configure_device(torch, {"encoder_device": "cuda"})
        torch.ones.assert_not_called()

    def test_cuda_kernel_failure_is_a_hard_error(self):
        torch = MagicMock()
        torch.ones.side_effect = RuntimeError("unsupported architecture")
        with self.assertRaisesRegex(InputError, "cannot execute"):
            configure_device(torch, {"encoder_device": "cuda"})

    def test_memory_limit_rejects_unbounded_configuration(self):
        for fraction in (0, -1, 1.1, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(InputError):
                configure_device(
                    MagicMock(),
                    {"encoder_device": "cuda", "cuda_memory_fraction": fraction},
                )

    def test_unknown_device_never_selects_cpu(self):
        with self.assertRaises(InputError):
            configure_device(MagicMock(), {"encoder_device": "auto"})


class SandboxTests(unittest.TestCase):
    def test_absent_sandbox_fails_closed(self):
        with patch("body_scan.network_sandbox.shutil.which", return_value=None):
            with self.assertRaises(InputError):
                sandbox_command("linux", [sys.executable, "-c", "pass"])

    def test_unknown_sandbox_fails_closed(self):
        with self.assertRaises(InputError):
            sandbox_command("none", [sys.executable])

    def test_failed_or_timed_out_probe_disables_execution(self):
        for outcome in (
            subprocess.CompletedProcess([], 1),
            subprocess.TimeoutExpired([], 10),
        ):
            with self.subTest(outcome=type(outcome).__name__):
                with patch(
                    "body_scan.network_sandbox.sandbox_command", return_value=[]
                ):
                    with patch("body_scan.network_sandbox.subprocess.run") as run:
                        if isinstance(outcome, Exception):
                            run.side_effect = outcome
                        else:
                            run.return_value = outcome
                        with self.assertRaisesRegex(
                            InputError, "execution is disabled"
                        ):
                            verify_network_sandbox("linux", sys.executable)

    @unittest.skipUnless(
        sys.platform == "linux"
        and os.environ.get("BODY_SCAN_TEST_LINUX_SANDBOX") == "1",
        "opt-in Linux host isolation integration test",
    )
    def test_real_namespace_blocks_host_loopback_and_outbound_network(self):
        # No skip on permission failure: an unusable namespace is a platform regression.
        verify_network_sandbox("linux", sys.executable)

    def test_unsandboxed_probe_is_detected(self):
        with patch(
            "body_scan.network_sandbox.sandbox_command",
            side_effect=lambda _, command: command,
        ):
            with self.assertRaises(InputError):
                verify_network_sandbox("linux", sys.executable)


if __name__ == "__main__":
    unittest.main()
