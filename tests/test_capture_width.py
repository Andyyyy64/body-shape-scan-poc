import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from body_scan.io import InputError


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class CaptureWidthTests(unittest.TestCase):
    def test_invalid_reference_camera_is_rejected_before_geometry_or_calibration(self):
        from body_scan.capture_width import prepare_reference

        original = {"camera_delta": [[0, 0, 0], [0, 0, 0]]}
        cases = [
            {"camera_delta": [[0.001, 0, 0], [0, 0, 0]]},
            {"camera_delta": [[0, 0, 0]]},
            {"camera_delta": [[np.nan, 0, 0], [0, 0, 0]]},
        ]
        with (
            patch(
                "body_scan.capture_width.reference_change",
                side_effect=AssertionError("geometry should not run"),
            ),
            patch(
                "body_scan.capture_width.calibrate",
                side_effect=AssertionError("calibration should not run"),
            ),
        ):
            for bad in cases:
                with self.subTest(camera=bad), self.assertRaises(InputError):
                    prepare_reference(
                        {"shared_source_order": np.arange(2)}, bad, original
                    )

    def test_query_geometry_and_camera_are_checked_before_optimization(self):
        from body_scan.capture_width import measure_capture

        reference = {
            "faces": np.array([[0, 1, 2]]),
            "section_y": np.array(1.0),
            "section_center": np.zeros(2),
        }
        prepared = {"reference": reference}
        with patch(
            "body_scan.capture_width.infer",
            side_effect=AssertionError("optimization should not run"),
        ):
            different = {**reference, "section_y": np.array(1.1)}
            with self.assertRaises(InputError):
                measure_capture(prepared, different, {}, {}, {})
            query = {**reference, "shared_camera": np.zeros((2, 3))}
            for camera in [np.zeros((1, 3)), np.full((2, 3), np.nan)]:
                with self.subTest(camera=camera), self.assertRaises(InputError):
                    measure_capture(prepared, query, {"camera_delta": camera}, {}, {})
