import importlib.util
import unittest

import numpy as np

from body_scan.io import InputError


@unittest.skipUnless(importlib.util.find_spec("cv2"), "requires vision runtime")
class ImageDetailTests(unittest.TestCase):
    def test_blur_lowers_detail_without_changing_person_mask(self):
        import cv2

        from body_scan.quality import image_detail

        image = np.full((320, 240, 3), 170, np.uint8)
        mask = np.zeros((320, 240), bool)
        mask[30:290, 60:180] = True
        image[mask] = [50, 85, 120]
        cv2.rectangle(image, (85, 40), (120, 260), (125, 145, 160), -1)
        original = image_detail(image, mask)["edge_energy"]
        blurred = image_detail(cv2.GaussianBlur(image, (21, 21), 5), mask)[
            "edge_energy"
        ]
        self.assertGreater(original, 5 * blurred)
        self.assertLess(
            image_detail(np.full_like(image, 80), mask)["edge_energy"], 1e-6
        )
        with self.assertRaises(InputError):
            image_detail(image, np.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
