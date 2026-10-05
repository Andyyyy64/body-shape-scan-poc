import unittest

import numpy as np

from body_scan.io import InputError
from body_scan.sampling import angular_indices


class AngularSamplingTests(unittest.TestCase):
    def test_slow_front_turn_does_not_dominate_selected_views(self):
        yaw = np.r_[np.linspace(0, 0.2, 40), np.linspace(0.3, 2 * np.pi - 0.2, 24)]
        picked = angular_indices(yaw, np.ones(len(yaw)), 16)
        self.assertEqual(len(set(picked)), 16)
        coverage = {int(yaw[i] % (2 * np.pi) / (2 * np.pi) * 8) for i in picked}
        self.assertEqual(len(coverage), 8)
        self.assertLess(sum(i < 40 for i in picked), 5)
        self.assertTrue(
            np.array_equal(picked, angular_indices(yaw, np.ones(len(yaw)), 16))
        )

    def test_partial_rotation_stays_partial_and_invalid_scores_are_rejected(self):
        yaw = np.linspace(0, np.pi / 2, 48)
        picked = angular_indices(yaw, np.ones(48), 16)
        self.assertLessEqual(yaw[picked].max() - yaw[picked].min(), np.pi / 2)
        self.assertTrue(
            np.array_equal(angular_indices(yaw, np.ones(48), 48), np.arange(48))
        )
        with self.assertRaises(InputError):
            angular_indices(yaw, np.full(48, np.nan), 16)
