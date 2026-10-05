import unittest

import numpy as np

from body_scan.coverage_renderer import paint_person
from body_scan.io import InputError


class CoverageRendererTests(unittest.TestCase):
    def rectangle(self, width):
        background = np.full((40, 80, 3), 200, np.uint8)
        left, right = 25.1 - width / 2, 25.1 + width / 2
        projected = np.array([[left, 5], [right, 5], [right, 35], [left, 35]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        return paint_person(
            background, projected, faces, np.full((2, 3), 50), np.arange(2)
        )

    def test_subpixel_motion_survives_at_original_resolution(self):
        areas = []
        for width in (19.5, 20, 20.5):
            image, mask = self.rectangle(width)
            self.assertEqual(image.shape, (40, 80, 3))
            self.assertEqual(mask.shape, (40, 80))
            self.assertTrue(((mask > 0) & (mask < 255)).any())
            areas.append(float(np.sum(mask[20] / 255)))
        np.testing.assert_allclose(np.diff(areas), 0.5, atol=0.08)

    def test_invalid_face_order_and_unbounded_projection_are_rejected(self):
        background = np.zeros((20, 20, 3), np.uint8)
        points = np.array([[5, 5], [15, 5], [5, 15]], float)
        faces = np.array([[0, 1, 2]])
        colors = np.full((1, 3), 50)
        for order in (np.array([0.0]), np.array([1]), np.array([[0]])):
            with self.subTest(order=order), self.assertRaises(InputError):
                paint_person(background, points, faces, colors, order)
        with self.assertRaises(InputError):
            paint_person(background, points * 1e10, faces, colors, np.array([0]))

    def test_render_frame_dispatch_preserves_default_and_rejects_unknown(self):
        from unittest.mock import patch

        from body_scan.synthetic_lab import render_frame

        mesh = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.2, 0]], float)
        faces = np.array([[0, 1, 2]])
        body = {"hip_y": 0, "shoulder_y": 0.2}
        spec = {"seed": 17, "distance": 2}
        original = render_frame(mesh, faces, body, spec, 0, 48)
        explicit = render_frame(
            mesh, faces, body, {**spec, "renderer": "legacy_integer.v1"}, 0, 48
        )
        np.testing.assert_array_equal(original, explicit)
        with patch("body_scan.coverage_renderer.paint_person") as paint:
            paint.return_value = (original, np.zeros(original.shape[:2], np.uint8))
            render_frame(
                mesh,
                faces,
                body,
                {**spec, "renderer": "continuous_coverage8.v1"},
                0,
                48,
            )
            self.assertEqual(paint.call_args.kwargs["scale"], 8)
        with self.assertRaises(InputError):
            render_frame(mesh, faces, body, {**spec, "renderer": "typo"}, 0, 48)

    def test_cli_freezes_renderer_without_changing_measurement_thresholds(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from body_scan.io import read_json, write_json
        from body_scan.synthetic_lab import main, protocol

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "runtime.json"
            write_json(config, {"engine_id": "test"})
            output = root / "study"
            argv = [
                "synthetic_lab",
                "init",
                "--runtime-config",
                str(config),
                "--out",
                str(output),
                "--cases",
                "baseline_17",
                "--renderer",
                "continuous_coverage8.v1",
            ]
            with (
                patch("sys.argv", argv),
                patch("body_scan.synthetic_lab.launch", return_value=0),
            ):
                main()
            matrix = read_json(output / "matrix.json")
            self.assertEqual(len(matrix["cases"]), 1)
            self.assertEqual(
                matrix["cases"][0]["spec"]["renderer"], "continuous_coverage8.v1"
            )
            saved = read_json(output / "protocol.json")
            self.assertEqual(saved.pop("renderer"), "continuous_coverage8.v1")
            self.assertEqual(saved, protocol())
