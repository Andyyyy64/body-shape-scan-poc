import importlib.util
import unittest

import numpy as np

from body_scan.io import InputError


@unittest.skipUnless(importlib.util.find_spec("cv2"), "requires vision runtime")
class BoundaryDetailTests(unittest.TestCase):
    def test_noise_does_not_make_blurred_subject_look_sharp(self):
        import cv2

        from body_scan.image_quality import boundary_detail
        from body_scan.quality import image_detail

        image = np.full((400, 240, 3), 170, np.uint8)
        mask = np.zeros((400, 240), np.uint8)
        mask[40:360, 80:160] = 1
        image[mask != 0] = [65, 90, 135]
        blur = cv2.GaussianBlur(image, (21, 21), 5)
        noise = np.random.default_rng(87).normal(0, 30, image.shape)
        noisy = np.clip(blur.astype(float) + noise, 0, 255).astype(np.uint8)
        # The old bbox energy ranks noise as detail: the actual failure seam.
        self.assertGreater(
            image_detail(noisy, mask)["edge_energy"],
            image_detail(blur, mask)["edge_energy"],
        )
        original = boundary_detail(image, mask)["boundary_detail"]
        noisy_score = boundary_detail(noisy, mask)["boundary_detail"]
        self.assertGreater(original, 10 * max(noisy_score, 0.001))
        self.assertTrue(np.isfinite(noisy_score))
        self.assertEqual(
            boundary_detail(np.full_like(image, 80), mask)["boundary_detail"], 0
        )
        with self.assertRaises(InputError):
            boundary_detail(image, np.zeros_like(mask))

    def test_sharp_background_does_not_dominate_person_only_blur(self):
        import cv2

        from body_scan.image_quality import boundary_detail

        image = np.full((400, 240, 3), 170, np.uint8)
        cv2.line(image, (100, 0), (100, 399), (70, 85, 100), 2)
        mask = np.zeros((400, 240), np.uint8)
        mask[40:360, 70:170] = 1
        image[mask != 0] = [65, 90, 135]
        blur = cv2.GaussianBlur(image, (21, 21), 5)
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 5)[..., None]
        local = np.clip(image * (1 - alpha) + blur * alpha, 0, 255).astype(np.uint8)
        sharp = boundary_detail(image, mask)["boundary_detail"]
        # The soft composite retains some original edge contrast; require at
        # least a twofold reduction rather than an entirely absent boundary.
        self.assertLess(boundary_detail(local, mask)["boundary_detail"], sharp * 0.5)
