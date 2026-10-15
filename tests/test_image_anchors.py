import unittest

import numpy as np

from body_scan.image_anchors import select_anchors
from body_scan.quality import FrameRejected


class ImageAnchorTests(unittest.TestCase):
    def test_person_and_pair_association_preserve_observed_coordinates(self):
        mask = np.zeros((100, 100), bool)
        mask[10:80, 20:60] = True
        reference = np.array([[30, 20], [50, 20], [32, 60], [48, 60]], float)
        points = np.zeros((2, 17, 3))
        observed = reference + [1, 2]
        points[0, [5, 6, 11, 12], :2] = observed[[1, 0, 3, 2]]
        points[0, [5, 6, 11, 12], 2] = [0.8, 0.9, 0.7, 0.95]
        points[1, [5, 6, 11, 12], :2] = reference + [45, 0]
        points[1, [5, 6, 11, 12], 2] = 1
        target, confidence = select_anchors(
            points, [[20, 10, 60, 80], [70, 10, 99, 80]], mask, reference
        )
        np.testing.assert_array_equal(target, observed)
        np.testing.assert_allclose(confidence, [0.9, 0.8, 0.95, 0.7])

    def test_missing_or_wrong_person_cannot_be_an_anatomy_observation(self):
        mask = np.zeros((100, 100), bool)
        mask[10:80, 20:60] = True
        reference = np.array([[30, 20], [50, 20], [32, 60], [48, 60]], float)
        points = np.zeros((1, 17, 3))
        points[0, [5, 6, 11, 12], :2] = reference
        points[:, :, 2] = 0.2
        with self.assertRaises(FrameRejected):
            select_anchors(points, [[20, 10, 60, 80]], mask, reference)
        points[:, :, 2] = 1
        with self.assertRaises(FrameRejected):
            select_anchors(points, [[80, 10, 99, 80]], mask, reference)
