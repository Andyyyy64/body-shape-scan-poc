import importlib.util
import unittest

from body_scan.io import InputError


@unittest.skipUnless(importlib.util.find_spec("roma"), "requires model runtime")
class PoseConstraintTests(unittest.TestCase):
    def test_rotation_matches_native_mhr_global_joint_contract(self):
        import numpy as np

        from body_scan.pose_constraints import root_rotation

        # Independently recorded MHR global joint 1 matrices from synthetic
        # inference; these expected values are not computed by Roma in this test.
        fixtures = [
            (
                [-0.014160050079226494, 0.11388838291168213, -0.001687709242105484],
                [
                    [0.9935202598571777, 0.00007841410115361214, 0.11365468055009842],
                    [-0.0016767749330028892, 0.9999009966850281, 0.013967780396342278],
                    [-0.11364234238862991, -0.014067847281694412, 0.9934220910072327],
                ],
            ),
            (
                [3.140125036239624, 0.17564046382904053, -3.1339128017425537],
                [
                    [-0.984585702419281, -0.007936210371553898, 0.17472216486930847],
                    [-0.007561621721833944, 0.9999673962593079, 0.0028095266316086054],
                    [-0.17473876476287842, 0.001445037079975009, -0.9846137166023254],
                ],
            ),
        ]
        for angles, expected in fixtures:
            params = np.zeros(204, np.float32)
            params[3:6] = angles
            np.testing.assert_allclose(root_rotation(params), expected, atol=1e-6)

    def test_all_headings_keep_rotation_when_tilt_is_constant(self):
        import numpy as np
        import roma
        import torch

        from body_scan.pose_constraints import constrain_root_tilt, root_tilt_reference

        tilt = roma.euler_to_rotmat("xyz", torch.tensor([0.03, 0, -0.04])).numpy()
        rows, matrices = [], []
        for angle in np.linspace(-np.pi, np.pi, 17):
            c, s = np.cos(angle), np.sin(angle)
            matrix = tilt @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
            params = np.zeros(204, np.float32)
            params[3:6] = roma.rotmat_to_euler("xyz", torch.from_numpy(matrix)).numpy()
            rows.append(params)
            matrices.append(matrix)
        reference = root_tilt_reference(rows)
        for params, matrix in zip(rows, matrices):
            adjusted = constrain_root_tilt(params, reference)
            result = roma.euler_to_rotmat(
                "xyz", torch.from_numpy(adjusted[3:6])
            ).numpy()
            np.testing.assert_allclose(result, matrix, atol=1e-5)
            np.testing.assert_array_equal(adjusted[6:], params[6:])

    def test_tilt_jitter_is_removed_without_losing_heading(self):
        import numpy as np
        import roma
        import torch

        from body_scan.pose_constraints import constrain_root_tilt

        for heading in (-3.0, -1.0, 0.0, 1.0, 3.0):
            for pitch in (-0.05, 0.05):
                # Yaw followed by body-axis tilt creates frame-specific up vectors.
                yaw = roma.euler_to_rotmat("xyz", torch.tensor([0.0, heading, 0.0]))
                tilt = roma.euler_to_rotmat("xyz", torch.tensor([0.0, 0.0, pitch]))
                params = np.zeros(204, np.float32)
                params[3:6] = roma.rotmat_to_euler("xyz", yaw @ tilt).numpy()
                adjusted = constrain_root_tilt(params, np.eye(3))
                matrix = roma.euler_to_rotmat(
                    "xyz", torch.from_numpy(adjusted[3:6])
                ).numpy()
                np.testing.assert_allclose(matrix[:, 1], [0, 1, 0], atol=1e-5)
                np.testing.assert_allclose(matrix, yaw.numpy(), atol=1e-5)

    def test_invalid_reference_is_rejected(self):
        import numpy as np

        from body_scan.pose_constraints import constrain_root_tilt

        with self.assertRaises(InputError):
            constrain_root_tilt(np.zeros(204), np.ones((3, 3)))

    def test_smoothing_preserves_uniform_turns_across_wrap_and_uneven_frames(self):
        import numpy as np

        from body_scan.pose_constraints import smooth_headings

        times = np.array([0, 1, 2, 4, 5, 8, 9, 10, 12, 15, 16, 18, 19, 21], float)
        expected = 2.8 + 0.25 * times
        wrapped = (expected + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(
            smooth_headings(wrapped, 10000, times), expected, atol=1e-9
        )

    def test_smoothing_reduces_jitter_without_imposing_uniform_velocity(self):
        import numpy as np

        from body_scan.pose_constraints import smooth_headings

        t = np.arange(48)
        true = 0.1 * t + 0.2 * np.sin(t / 12)
        noisy = true + 0.05 * np.cos(t * 2)
        result = smooth_headings(noisy, 4)
        self.assertLess(float(np.sqrt(np.mean((result - true) ** 2))), 0.02)
        self.assertGreater(float(np.ptp(np.diff(result))), 0.02)

    def test_smoothing_rejects_nonmonotonic_times(self):
        from body_scan.pose_constraints import smooth_headings

        with self.assertRaises(InputError):
            smooth_headings([0, 0.1, 0.2], 4, [0, 2, 1])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires model runtime")
    def test_constant_rate_is_estimated_separately_from_periodic_heading_bias(self):
        import numpy as np

        from body_scan.pose_constraints import harmonic_bias_headings

        time = np.arange(72)
        for rate in (-0.11, 0.11):
            true = 0.4 + rate * time
            corrupted = true + 0.05 * np.sin(2 * true) + 0.02 * np.cos(true)
            wrapped = (corrupted + np.pi) % (2 * np.pi) - np.pi
            result = harmonic_bias_headings(wrapped, time)
            self.assertLess(float(np.sqrt(np.mean((result - true) ** 2))), 0.001)
            estimated_rate = np.polyfit(time, result, 1)[0]
            self.assertAlmostEqual(float(estimated_rate), rate, places=4)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "requires model runtime")
    def test_constant_motion_model_rejects_insufficient_turn_observation(self):
        from body_scan.pose_constraints import harmonic_bias_headings

        with self.assertRaises(InputError):
            harmonic_bias_headings([i * 0.1 for i in range(15)])
