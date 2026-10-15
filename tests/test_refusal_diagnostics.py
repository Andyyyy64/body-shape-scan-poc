import tempfile
import unittest
from pathlib import Path

from body_scan.io import write_json
from body_scan.synthetic_evidence import refusal_kind


class RefusalDiagnosticsTests(unittest.TestCase):
    def test_new_blur_refusal_is_classified_but_a_crash_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "store" / "session" / "attempt_one"
            attempt.mkdir(parents=True)
            write_json(
                attempt / "qc.json",
                {
                    "usable_frames": 0,
                    "required_frames": 36,
                    "exclusion_reasons": {"insufficient_image_detail": 48},
                },
            )
            row = {
                "name": "stress_blur",
                "state": "failed",
                "session_id": "session",
                "reconstruction": {
                    "stage": "failed",
                    "error": "使用できるフレームが0/48枚で必要数に届きません",
                },
            }
            self.assertEqual(refusal_kind(root, row), "frame_quality")
            row["reconstruction"]["error"] = "Model inference failed unexpectedly"
            self.assertIsNone(refusal_kind(root, row))
