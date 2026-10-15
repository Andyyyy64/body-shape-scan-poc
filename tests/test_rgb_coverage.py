"""Physical fractional-pixel controls for the research RGB width observer."""

import unittest

try:
    import cv2
    import numpy as np
    import torch

    from body_scan.contour_width import central_widths
    from body_scan.subpixel_edges import central_positions, refine_positions
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy, OpenCV and Torch vision dependencies are required")
class SubpixelEdgeTests(unittest.TestCase):
    def image(self, left, right, *, blur=0.0, contrast=1.0):
        pixel = np.arange(160)
        coverage = np.maximum(
            0, np.minimum(pixel + 0.5, right) - np.maximum(pixel - 0.5, left)
        )
        background = np.array([180.0, 170.0, 170.0])
        foreground = background + contrast * (
            np.array([30.0, 100.0, 190.0]) - background
        )
        rgb = (
            np.broadcast_to(
                background + coverage[:, None] * (foreground - background), (80, 160, 3)
            )
            .copy()
            .astype(np.float32)
        )
        if blur:
            rgb = cv2.GaussianBlur(rgb, (0, 0), blur)
        return np.round(rgb).astype(np.uint8)

    def test_subpixel_width_changes_and_blur(self):
        errors = []
        for blur in [0.0, 0.8, 1.2]:
            for shift in np.linspace(-0.4, 0.4, 9):
                left, right = 39.2 - shift, 99.35 + shift
                image = self.image(left, right, blur=blur)
                seed = np.array([[np.ceil(left), np.floor(right)]])
                result = refine_positions(image, np.array([35.25]), seed)
                self.assertTrue(result["refined"].all())
                errors.extend(result["positions"][0] - [left, right])
                self.assertLess(
                    abs(np.diff(result["positions"][0])[0] - (right - left)), 0.15
                )
        self.assertLess(abs(np.asarray(errors)).max(), 0.12)

    def test_weak_or_remote_edges_are_explicit_fallback_without_row_removal(self):
        seed = np.array([[40.0, 99.0]])
        weak = refine_positions(
            self.image(39.2, 99.35, contrast=0.01), np.array([35.0]), seed
        )
        self.assertFalse(weak["refined"].any())
        np.testing.assert_array_equal(weak["positions"], seed)
        self.assertTrue((weak["fallback_reason"] == 1).all())
        remote = refine_positions(self.image(20.2, 120.35), np.array([35.0]), seed)
        self.assertFalse(remote["refined"].any())
        self.assertEqual(remote["positions"].shape, seed.shape)

    def test_separate_arm_does_not_change_central_interval(self):
        polygon = np.array(
            [
                [0, 0],
                [10, 0],
                [10, 3],
                [2, 3],
                [2, 9],
                [8, 9],
                [8, 4],
                [10, 4],
                [10, 10],
                [0, 10],
            ]
        )
        rows = np.array([5.0, 6.25])
        centers = np.array([1.0, 1.0])
        positions = central_positions(polygon, rows, centers)
        widths = central_widths(
            torch.tensor(polygon[None], dtype=torch.float64),
            torch.tensor(rows[None]),
            torch.tensor(centers[None]),
        ).numpy()[0]
        np.testing.assert_allclose(
            positions[:, 1] - positions[:, 0], widths, atol=1e-12
        )
        np.testing.assert_allclose(widths, 2.0)

    def test_affine_shading_does_not_move_the_physical_edge(self):
        pixel = np.arange(160)
        left, right = 39.2, 99.35
        alpha = np.maximum(
            0, np.minimum(pixel + 0.5, right) - np.maximum(pixel - 0.5, left)
        )
        background = np.array([180.0, 170.0, 170.0]) + np.array([0.3, -0.2, 0.05]) * (
            pixel[:, None] - 70
        )
        foreground = np.array([30.0, 100.0, 190.0]) + np.array([-0.2, 0.1, 0.3]) * (
            pixel[:, None] - 70
        )
        image = np.round(
            np.broadcast_to(
                background + alpha[:, None] * (foreground - background), (80, 160, 3)
            )
        ).astype(np.uint8)
        result = refine_positions(image, np.array([35.25]), np.array([[40.0, 99.0]]))
        self.assertTrue(result["refined"].all())
        np.testing.assert_allclose(result["positions"][0], [left, right], atol=0.04)
