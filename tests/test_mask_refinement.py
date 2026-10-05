import importlib.util
import unittest

import numpy as np

from body_scan.io import InputError
from body_scan.quality import FrameRejected


@unittest.skipUnless(importlib.util.find_spec("cv2"), "requires vision runtime")
class MaskRefinementTests(unittest.TestCase):
    def test_observed_rgb_restores_boundary_without_renderer_geometry(self):
        from body_scan.mask_refinement import refine_person_mask

        image = np.full((160, 120, 3), 180, np.uint8)
        actual = np.zeros(image.shape[:2], bool)
        actual[20:145, 35:85] = True
        image[actual] = [125, 70, 40]
        estimated = np.zeros_like(actual)
        estimated[22:143, 38:82] = True
        refined, _ = refine_person_mask(image, estimated)
        self.assertLess((refined != actual).sum(), (estimated != actual).sum())
        repeated, _ = refine_person_mask(image, estimated)
        np.testing.assert_array_equal(refined, repeated)
        with self.assertRaises(FrameRejected):
            refine_person_mask(image, np.zeros_like(actual))
        with self.assertRaises(InputError):
            refine_person_mask(image, actual[:100])
