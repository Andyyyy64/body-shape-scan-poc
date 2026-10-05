"""Research measurements must separate standing-phase errors from shape change."""

import unittest

try:
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.optimize import least_squares

    from body_scan.framewise_width import fit_curve, pose_sigma
    from body_scan.io import InputError
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy and SciPy research dependencies are required")
class FramewiseWidthTests(unittest.TestCase):
    def fixture(self):
        phase = np.arange(48) * 2 * np.pi / 48
        height = np.linspace(0, 1, 41)
        width = (
            60
            + 8 * np.cos(2 * phase[:, None])
            + 12 * np.sin(phase[:, None]) * height[None, :] ** 2
        )
        jacobian = np.stack(
            [
                100
                * (1 + 0.3 * np.cos(phase[:, None]))
                * np.exp(-(((height[None, :] - 0.3) / 0.15) ** 2)),
                80
                * (1 + 0.4 * np.sin(phase[:, None]))
                * np.exp(-(((height[None, :] - 0.6) / 0.15) ** 2)),
            ],
            -1,
        )
        calibration = {
            "heading": phase,
            "width": width,
            "jacobian": jacobian,
            "row_mode": "fitted_camera",
        }
        knots = np.r_[phase, 2 * np.pi]
        curve = CubicSpline(
            knots, np.concatenate([width, width[:1]]), axis=0, bc_type="periodic"
        )
        jcurve = CubicSpline(
            knots, np.concatenate([jacobian, jacobian[:1]]), axis=0, bc_type="periodic"
        )
        return calibration, phase, curve, jcurve

    def test_framewise_jitter_does_not_become_local_shape_change(self):
        c, phase, curve, jcurve = self.fixture()
        true = np.array([0.02, -0.01])
        jitter = 0.035 * np.sin(3 * phase) + 0.015 * np.cos(phase)
        observed = curve(phase + jitter) + jcurve(phase + jitter) @ true
        global_only = least_squares(
            lambda x: (
                curve(phase + x[2]) + jcurve(phase + x[2]) @ x[:2] - observed
            ).ravel(),
            np.zeros(3),
        )
        fit = fit_curve(
            c,
            phase,
            observed,
            np.full(48, 0.06),
            {
                "delta_coefficients": global_only.x[:2],
                "phase_refinement_deg": np.rad2deg(global_only.x[2]),
            },
        )
        self.assertIn(
            fit["solver_status"],
            [
                "converged_step",
                "converged_objective",
                "converged_gradient",
                "converged_objective_and_step",
            ],
        )
        self.assertLess(
            np.linalg.norm(fit["delta_coefficients"] - true),
            np.linalg.norm(global_only.x[:2] - true) * 0.2,
        )
        self.assertLess(np.linalg.norm(fit["delta_coefficients"] - true), 0.0002)
        # Near-stationary width views cannot identify phase; assess observed phase information.
        error = np.deg2rad(fit["total_frame_phase_deg"]) - jitter
        derivative = curve(phase + jitter, 1) + jcurve(phase + jitter, 1) @ true
        information = np.sum(derivative**2, axis=1)
        self.assertLess(
            np.sqrt(np.sum(information * error**2) / np.sum(information)), 0.003
        )
        self.assertGreater(np.sqrt(np.mean(error**2)), 0.003)
        self.assertEqual(fit["frame_count"] * fit["row_count"], observed.size)

    def test_coherent_shape_and_shared_phase_are_not_shrunk(self):
        c, phase, curve, jcurve = self.fixture()
        for true in [np.array([0.03, -0.02]), np.array([-0.03, 0.02]), np.zeros(2)]:
            shared_phase = 0.025
            observed = curve(phase + shared_phase) + jcurve(phase + shared_phase) @ true
            fit = fit_curve(
                c,
                phase,
                observed,
                np.full(48, 0.01),
                {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0},
            )
            np.testing.assert_allclose(fit["delta_coefficients"], true, atol=1e-7)
            self.assertLess(
                abs(np.deg2rad(fit["phase_refinement_deg"]) - shared_phase), 1e-7
            )
            self.assertLess(
                abs(np.asarray(fit["frame_phase_deviations_deg"])).max(), 1e-5
            )

    def test_invalid_uncertainty_is_rejected_and_rule_is_common(self):
        for value in [np.full(48, -1.0), np.full(48, np.nan), np.full(48, np.inf)]:
            with self.assertRaises(InputError):
                pose_sigma(value, np.ones(48))
        np.testing.assert_allclose(
            pose_sigma(np.zeros(48), np.zeros(48)), np.sqrt(2) * np.deg2rad(0.25)
        )
        np.testing.assert_allclose(
            pose_sigma(np.ones(48), np.ones(48)), np.sqrt(2) * np.deg2rad(5)
        )

    def test_diagonal_covariance_preserves_independent_fit(self):
        c, phase, curve, jcurve = self.fixture()
        true = np.array([0.02, -0.01])
        jitter = 0.035 * np.sin(3 * phase) + 0.015 * np.cos(phase)
        observed = curve(phase + jitter) + jcurve(phase + jitter) @ true
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        sigma = np.full(48, 0.06)
        original = fit_curve(c, phase, observed, sigma, initial)
        diagonal = fit_curve(
            c, phase, observed, sigma, initial, phase_covariance=np.diag(sigma**2)
        )
        np.testing.assert_allclose(
            diagonal["delta_coefficients"], original["delta_coefficients"], atol=1e-9
        )
        np.testing.assert_allclose(
            diagonal["total_frame_phase_deg"],
            original["total_frame_phase_deg"],
            atol=1e-7,
        )
        self.assertAlmostEqual(diagonal["objective"], original["objective"], places=9)

    def test_correlated_prior_preserves_coherent_changes_and_rejects_invalid_covariance(
        self,
    ):
        c, phase, curve, jcurve = self.fixture()
        index = np.arange(48)
        covariance = 0.01**2 * 0.6 ** abs(index[:, None] - index[None, :])
        true = np.array([-0.03, 0.02])
        observed = curve(phase + 0.025) + jcurve(phase + 0.025) @ true
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        fit = fit_curve(
            c, phase, observed, np.full(48, 0.01), initial, phase_covariance=covariance
        )
        np.testing.assert_allclose(fit["delta_coefficients"], true, atol=1e-7)
        self.assertLess(abs(np.deg2rad(fit["phase_refinement_deg"]) - 0.025), 1e-7)
        for invalid in [-covariance, covariance + np.triu(np.ones((48, 48))) * 0.01]:
            with self.assertRaises(InputError):
                fit_curve(
                    c,
                    phase,
                    observed,
                    np.full(48, 0.01),
                    initial,
                    phase_covariance=invalid,
                )
