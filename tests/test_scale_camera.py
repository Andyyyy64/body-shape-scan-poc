import unittest
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline

from body_scan.calibrated_width import observe, project
from body_scan.camera_prior_uncertainty import head_camera_prior_sigma
from body_scan.camera_width_response import width_camera_derivative
from body_scan.framewise_camera import fit_curve_camera
from body_scan.framewise_width import fit_curve
from body_scan.head_anatomy_scale import scan_head_anatomy
from body_scan.head_scale import representative_width, scale_from_reference, scan_head
from body_scan.io import InputError
from body_scan.leg_scale import calf_widths, scan_leg_anatomy
from body_scan.subpixel_edges import central_positions


class ObservationNoiseTests(unittest.TestCase):
    def fixture(self):
        phase = np.arange(48) * 2 * np.pi / 48
        rows = np.linspace(-1, 1, 6)
        width = 60 + 8 * np.cos(2 * phase[:, None]) + rows[None, :]
        jacobian = np.stack(
            [
                np.broadcast_to(100 * np.ones_like(phase)[:, None], width.shape),
                100 * (1 + 0.2 * np.cos(phase[:, None])) * rows[None, :],
            ],
            axis=-1,
        )
        calibration = {
            "heading": phase,
            "width": width,
            "jacobian": jacobian,
            "row_mode": "fitted_camera",
        }
        return (calibration, phase)

    def run_fit(self, calibration, phase, observed):
        return fit_curve(
            calibration,
            phase,
            observed,
            np.full(48, 0.01),
            {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0},
        )

    def test_noisy_rows_have_declared_uncertainty_without_dropping_rows(self):
        c, phase = self.fixture()
        true = np.array([0.02, -0.01])
        observed = c["width"] + c["jacobian"] @ true
        observed[:, -2:] += 4
        plain = self.run_fit(c, phase, observed)
        weighted = self.run_fit(
            {**c, "noise_sigma": np.array([1.0, 1.0, 1.0, 1.0, 100.0, 100.0])},
            phase,
            observed,
        )
        self.assertGreater(np.linalg.norm(plain["delta_coefficients"] - true), 0.003)
        self.assertLess(np.linalg.norm(weighted["delta_coefficients"] - true), 2e-05)
        self.assertLess(
            np.linalg.norm(weighted["delta_coefficients"] - true),
            np.linalg.norm(plain["delta_coefficients"] - true) * 0.01,
        )
        self.assertEqual(weighted["frame_count"] * weighted["row_count"], observed.size)
        self.assertTrue(weighted["all_observation_rows_retained"])

    def test_unit_noise_preserves_default_fit(self):
        c, phase = self.fixture()
        observed = c["width"] + c["jacobian"] @ np.array([0.02, -0.01])
        plain = self.run_fit(c, phase, observed)
        explicit = self.run_fit({**c, "noise_sigma": np.ones(6)}, phase, observed)
        np.testing.assert_array_equal(
            plain["delta_coefficients"], explicit["delta_coefficients"]
        )
        self.assertEqual(plain["objective"], explicit["objective"])

    def test_invalid_noise_is_rejected(self):
        c, phase = self.fixture()
        for invalid in [
            np.ones(5),
            np.zeros(6),
            np.full(6, -1),
            np.full(6, np.nan),
            np.full(6, np.inf),
        ]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(InputError):
                    self.run_fit({**c, "noise_sigma": invalid}, phase, c["width"])


fixtures = SimpleNamespace(ObservationNoiseTests=ObservationNoiseTests)


class AngularNoiseTests(unittest.TestCase):
    def test_local_bad_view_uncertainty_does_not_become_shape(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        true = np.array([0.02, -0.01])
        observed = c["width"] + c["jacobian"] @ true
        observed[12:24, :] += 4
        variance = np.ones_like(observed)
        variance[12:24, :] = 10000
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        plain = fit_curve(c, phase, observed, np.full(48, 0.01), initial)
        weighted = fit_curve(
            {**c, "noise_variance_curve": variance},
            phase,
            observed,
            np.full(48, 0.01),
            initial,
        )
        self.assertGreater(np.linalg.norm(plain["delta_coefficients"] - true), 0.0001)
        self.assertLess(np.linalg.norm(weighted["delta_coefficients"] - true), 1e-05)
        self.assertLess(
            np.linalg.norm(weighted["delta_coefficients"] - true),
            np.linalg.norm(plain["delta_coefficients"] - true) * 0.02,
        )
        self.assertEqual(weighted["frame_count"] * weighted["row_count"], observed.size)
        self.assertTrue(weighted["all_observation_rows_retained"])

    def test_unit_angular_variance_preserves_fit_exactly(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        observed = c["width"] + c["jacobian"] @ np.array([0.02, -0.01])
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        a = fit_curve(c, phase, observed, np.full(48, 0.01), initial)
        b = fit_curve(
            {**c, "noise_variance_curve": np.ones_like(observed)},
            phase,
            observed,
            np.full(48, 0.01),
            initial,
        )
        np.testing.assert_array_equal(a["delta_coefficients"], b["delta_coefficients"])
        self.assertEqual(a["objective"], b["objective"])

    def test_invalid_angular_variance_is_rejected(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        for invalid in [
            np.ones((48, 5)),
            np.zeros((48, 6)),
            np.full((48, 6), np.nan),
            np.full((48, 6), np.inf),
        ]:
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(InputError):
                    fit_curve(
                        {**c, "noise_variance_curve": invalid},
                        phase,
                        c["width"],
                        np.full(48, 0.01),
                        {
                            "delta_coefficients": np.zeros(2),
                            "phase_refinement_deg": 0.0,
                        },
                    )


class SymmetricNoiseTests(unittest.TestCase):
    def test_query_variance_matches_explicit_total_variance(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        observed = c["width"] + c["jacobian"] @ np.array([0.02, -0.01])
        observed[:, -2:] += 4
        query = np.zeros_like(observed)
        query[:, -2:] = 10000
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        total = fit_curve(
            {**c, "noise_variance_curve": 1 + query},
            phase,
            observed,
            np.full(48, 0.01),
            initial,
        )
        split = fit_curve(
            {
                **c,
                "noise_variance_curve": np.ones_like(query),
                "query_noise_variance": query,
            },
            phase,
            observed,
            np.full(48, 0.01),
            initial,
        )
        np.testing.assert_array_equal(
            split["delta_coefficients"], total["delta_coefficients"]
        )
        self.assertEqual(split["objective"], total["objective"])
        self.assertEqual(split["frame_count"] * split["row_count"], observed.size)

    def test_zero_query_variance_preserves_reference_only_result(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        observed = c["width"] + c["jacobian"] @ np.array([0.02, -0.01])
        c = {**c, "noise_variance_curve": np.ones_like(observed)}
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        a = fit_curve(c, phase, observed, np.full(48, 0.01), initial)
        b = fit_curve(
            {**c, "query_noise_variance": np.zeros_like(observed)},
            phase,
            observed,
            np.full(48, 0.01),
            initial,
        )
        np.testing.assert_array_equal(a["delta_coefficients"], b["delta_coefficients"])
        self.assertEqual(a["objective"], b["objective"])

    def test_invalid_query_variance_is_rejected(self):
        c, phase = fixtures.ObservationNoiseTests().fixture()
        c = {**c, "noise_variance_curve": np.ones_like(c["width"])}
        for invalid in [
            np.ones((48, 5)),
            -np.ones((48, 6)),
            np.full((48, 6), np.nan),
            np.full((48, 6), np.inf),
        ]:
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(InputError):
                    fit_curve(
                        {**c, "query_noise_variance": invalid},
                        phase,
                        c["width"],
                        np.full(48, 0.01),
                        {
                            "delta_coefficients": np.zeros(2),
                            "phase_refinement_deg": 0.0,
                        },
                    )


class HeadScaleTests(unittest.TestCase):
    def test_slanted_head_scans_actual_interval_instead_of_roi_bbox_center(self):
        polygon = np.array([[10, 0], [20, 0], [100, 20], [90, 20]])
        rows, positions, counts = scan_head(polygon, 100)
        centers = positions.mean(1)
        np.testing.assert_allclose(
            positions, central_positions(polygon, rows, centers), atol=1e-10
        )
        self.assertTrue((counts == 1).all())
        self.assertGreater(np.ptp(centers), 40)

    def test_same_pixel_shape_zoom_changes_width_by_exact_factor(self):
        polygon = np.array([[10, 0], [20, 0], [100, 20], [90, 20]])
        _, p, _ = scan_head(polygon, 100)
        _, q, _ = scan_head(polygon * 1.5 + np.array([3, 5]), 150)
        a = representative_width(np.tile(p[None], (48, 1, 1)))
        b = representative_width(np.tile(q[None], (48, 1, 1)))
        np.testing.assert_allclose(b / a, 1.5, atol=1e-10)

    def test_depth_bias_corrected_without_using_known_camera_depth(self):
        heading = np.arange(48) * 2 * np.pi / 48
        width = 0.2 + 0.02 * np.cos(2 * heading)
        result = scale_from_reference(heading, width, heading, width * 1.025)
        self.assertAlmostEqual(result["median_depth_scale"], 1 / 1.025, places=12)
        self.assertTrue(result["all48_head_observations_retained"])

    def test_missing_head_view_fails_instead_of_removing_case(self):
        heading = np.arange(48) * 2 * np.pi / 48
        width = np.ones(48)
        width[3] = 0
        with self.assertRaises(InputError):
            scale_from_reference(heading, np.ones(48), heading, width)

    def test_all64_rows_are_required(self):
        with self.assertRaises(InputError):
            representative_width(np.zeros((48, 63, 2)))


class HeadAnatomyScaleTests(unittest.TestCase):
    def fixture(self):
        polygon = np.array([[10, 0], [20, 0], [20, 200], [10, 200]])
        points = np.zeros((17, 3))
        points[:, 2] = 0.9
        points[[5, 6], 1] = 80
        points[[11, 12], 1] = 160
        return (polygon, points)

    def test_foot_extent_does_not_change_head_roi(self):
        poly, points = self.fixture()
        a = scan_head_anatomy(poly, points)
        poly[poly[:, 1] == 200, 1] = 300
        b = scan_head_anatomy(poly, points)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_zoom_changes_head_width_by_correct_scale(self):
        poly, points = self.fixture()
        _, p, _ = scan_head_anatomy(poly, points)
        points[:, :2] *= 1.5
        _, q, _ = scan_head_anatomy(poly * 1.5, points)
        np.testing.assert_allclose((q[:, 1] - q[:, 0]) / (p[:, 1] - p[:, 0]), 1.5)

    def test_independent_anatomy_confidence_required_for_every_view(self):
        poly, points = self.fixture()
        points[5, 2] = 0.4
        with self.assertRaises(InputError):
            scan_head_anatomy(poly, points)


class CalfScaleTests(unittest.TestCase):
    def fixture(self):
        polygon = np.array(
            [
                [0, 0],
                [30, 0],
                [30, 100],
                [22, 100],
                [22, 20],
                [8, 20],
                [8, 100],
                [0, 100],
            ],
            float,
        )
        kp = np.zeros((17, 3))
        kp[:, 2] = 1
        kp[[13, 14], 1] = 30
        kp[[15, 16], 1] = 90
        return (polygon, kp)

    def test_both_separate_legs_are_counted_without_background_gap(self):
        p, k = self.fixture()
        rows, spans = scan_leg_anatomy(p, k)
        np.testing.assert_array_equal(calf_widths(spans), np.full(32, 16.0))
        self.assertTrue(all((len(s) == 2 for s in spans)))
        self.assertEqual(len(rows), 32)

    def test_zoom_transports_all_calf_rows_and_widths(self):
        p, k = self.fixture()
        r, s = scan_leg_anatomy(p, k)
        kk = k.copy()
        kk[:, :2] *= 1.5
        rr, ss = scan_leg_anatomy(p * 1.5, kk)
        np.testing.assert_allclose(rr, r * 1.5)
        np.testing.assert_allclose(calf_widths(ss), calf_widths(s) * 1.5)

    def test_low_confidence_or_missing_rows_are_refused(self):
        p, k = self.fixture()
        k[15, 2] = 0.49
        with self.assertRaises(InputError):
            scan_leg_anatomy(p, k)
        with self.assertRaises(InputError):
            calf_widths([np.array([[0, 2.0]])] * 31)


class CameraWidthResponseTests(unittest.TestCase):
    def fixture(self):
        vertices = np.tile(
            [[[-0.3, -0.5, 0], [0.3, -0.5, 0], [0.3, 0.5, 0], [-0.3, 0.5, 0]]],
            (48, 1, 1),
        )
        anchors = np.tile(
            [[[-0.1, -0.3, 0], [0.1, -0.3, 0], [-0.1, 0.3, 0], [0.1, 0.3, 0]]],
            (48, 1, 1),
        )
        t = np.arange(48) * 2 * np.pi / 48
        rotations = np.array(
            [
                [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
                for a in t
            ]
        )
        camera = np.tile([[0.05, 0.08, 2.0]], (48, 1))
        focal = np.full(48, 430.0)
        center = np.tile([[320.0, 320.0]], (48, 1))
        data = {
            "initial": np.zeros(1),
            "shared_base": vertices,
            "shared_basis": np.zeros((48, 4, 3, 1)),
            "shared_canonical_local_independent": np.zeros((48, 4, 3, 2)),
            "shared_anchor_base": anchors,
            "shared_anchor_basis": np.zeros((48, 4, 3, 1)),
            "normalization_camera": camera,
            "shared_camera": camera.copy(),
            "shared_focal": focal,
            "shared_center": center,
            "shared_mask_height": np.full(48, 215.0),
            "frame_rotations": rotations,
            "root_tilt_reference": np.eye(3),
            "shared_source_order": np.arange(48),
        }
        data["shared_target"] = project(vertices, camera, focal, center).numpy()
        data["width_target"] = data["shared_target"]
        c = {
            "initial": np.zeros(1),
            "global_coefficients": np.zeros(1),
            "baseline_coefficients": np.zeros(2),
            "reference_height_model_m": 1.8,
        }
        _, width, rows, centers = observe(data, c["initial"], row_mode="fitted_camera")
        data["observed_width_px"] = width * (215 / 256) + 0.3
        data["width_observation_rows"] = rows.numpy()
        data["width_observation_centers"] = centers.numpy()
        return (data, c)

    def test_planar_constant_width_has_analytic_camera_response_including_rgb_offset(
        self,
    ):
        data, c = self.fixture()
        derivative = width_camera_derivative(data, c, 2.0)
        expected = data["observed_width_px"] * 256 / (430 * 1.8)
        np.testing.assert_allclose(derivative, expected, rtol=0, atol=1e-08)

    def test_response_step_agrees_and_original_pixels_and_camera_are_preserved(self):
        data, c = self.fixture()
        camera = data["normalization_camera"].copy()
        pixels = data["observed_width_px"].copy()
        a = width_camera_derivative(data, c, 2.0, step=0.0001)
        b = width_camera_derivative(data, c, 2.0, step=0.0005)
        np.testing.assert_allclose(a, b, atol=1e-08)
        np.testing.assert_array_equal(data["normalization_camera"], camera)
        np.testing.assert_array_equal(data["observed_width_px"], pixels)

    def test_invalid_camera_step_is_rejected(self):
        data, c = self.fixture()
        with self.assertRaises(InputError):
            width_camera_derivative(data, c, 2.0, step=0)


class CorrelatedHeadPriorTests(unittest.TestCase):
    def test_constant_measurements_keep_floor_without_discarding_views(self):
        r = head_camera_prior_sigma(np.ones(48), 2.0)
        self.assertEqual(r["sigma_model_m"], 0.005)
        self.assertEqual(r["effective_head_views"], 48)
        self.assertTrue(r["all48_views_retained"])

    def test_smooth_angular_bias_does_not_gain_false_48_independent_views(self):
        r = head_camera_prior_sigma(
            1 + 0.03 * np.cos(np.arange(48) * 2 * np.pi / 48), 2.0
        )
        self.assertLess(r["effective_head_views"], 6)
        self.assertGreater(r["sigma_model_m"], r["independent48_sigma_model_m"] * 2)

    def test_sigma_is_common_rule_and_invariant_to_circular_view_order(self):
        scales = 1 + 0.03 * np.cos(np.arange(48) * 2 * np.pi / 48)
        a = head_camera_prior_sigma(scales, 2.0)
        b = head_camera_prior_sigma(np.roll(scales, 9), 2.0)
        self.assertAlmostEqual(a["sigma_model_m"], b["sigma_model_m"], places=12)
        self.assertAlmostEqual(
            a["effective_head_views"], b["effective_head_views"], places=12
        )

    def test_missing_or_nonpositive_scale_fails(self):
        with self.assertRaises(InputError):
            head_camera_prior_sigma(np.ones(47), 2.0)
        v = np.ones(48)
        v[0] = 0
        with self.assertRaises(InputError):
            head_camera_prior_sigma(v, 2.0)


class CurveFixture:
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
        return (calibration, phase, curve, jcurve)


class JointCameraTests(unittest.TestCase):
    def test_camera_bias_is_not_mistaken_for_local_shape(self):
        c, phase, curve, jcurve = CurveFixture().fixture()
        true = np.array([0.02, -0.01])
        camera_true = 0.025
        column = c["width"] / 2
        observed = curve(phase) + jcurve(phase) @ true - column * camera_true
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        old = fit_curve(c, phase, observed, np.full(48, 0.01), initial)
        fit = fit_curve_camera(
            c,
            phase,
            observed,
            np.full(48, 0.01),
            initial,
            camera_derivative=column,
            camera_prior_mean=0.03,
            camera_prior_sigma=0.02,
            camera_bound=0.1,
        )
        self.assertGreater(np.linalg.norm(old["delta_coefficients"] - true), 0.002)
        self.assertLess(np.linalg.norm(fit["delta_coefficients"] - true), 0.0001)
        self.assertLess(abs(fit["camera_depth_offset_model_m"] - camera_true), 0.0002)
        self.assertTrue(fit["camera_information_profiled_in_shape"])
        self.assertEqual(fit["frame_count"] * fit["row_count"], 48 * 41)

    def test_zero_camera_response_preserves_old_shape_and_phase(self):
        c, phase, curve, jcurve = CurveFixture().fixture()
        true = np.array([0.02, -0.01])
        observed = curve(phase) + jcurve(phase) @ true
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        old = fit_curve(c, phase, observed, np.full(48, 0.01), initial)
        fit = fit_curve_camera(
            c,
            phase,
            observed,
            np.full(48, 0.01),
            initial,
            camera_derivative=np.zeros_like(observed),
            camera_prior_mean=0.03,
            camera_prior_sigma=0.02,
            camera_bound=0.1,
        )
        np.testing.assert_allclose(
            fit["delta_coefficients"], old["delta_coefficients"], atol=1e-09
        )
        self.assertAlmostEqual(
            fit["phase_refinement_deg"], old["phase_refinement_deg"], places=7
        )
        self.assertAlmostEqual(fit["camera_depth_offset_model_m"], 0.03, places=8)

    def test_missing_camera_row_or_invalid_prior_is_rejected(self):
        c, phase, curve, _ = CurveFixture().fixture()
        observed = curve(phase)
        initial = {"delta_coefficients": np.zeros(2), "phase_refinement_deg": 0.0}
        for column, sigma in [
            (np.zeros((48, 40)), 0.01),
            (np.zeros_like(observed), 0.0),
        ]:
            with self.assertRaises(InputError):
                fit_curve_camera(
                    c,
                    phase,
                    observed,
                    np.full(48, 0.01),
                    initial,
                    camera_derivative=column,
                    camera_prior_mean=0.0,
                    camera_prior_sigma=sigma,
                    camera_bound=0.1,
                )


class ScaleCueIdentityTests(unittest.TestCase):
    def test_changed_frames_missing_cues_or_nonfinite_widths_are_rejected(self):
        from body_scan.capture_scale_width import _cue_widths

        data = {"frames": np.arange(48), "shared_source_order": np.arange(48)}
        cues = {
            "observed_frames": np.arange(48),
            "rgb_head_width_px": np.full(48, 40.0),
            "rgb_leg_width_px": np.full(48, 30.0),
        }
        np.testing.assert_array_equal(
            _cue_widths(cues, data), np.stack([np.full(48, 40.0), np.full(48, 30.0)])
        )
        for change in [
            {"observed_frames": np.arange(48)[::-1]},
            {"rgb_leg_width_px": np.full(47, 30.0)},
            {"rgb_head_width_px": np.full(48, np.nan)},
            {"rgb_leg_width_px": np.zeros(48)},
        ]:
            with self.subTest(change=list(change)), self.assertRaises(InputError):
                _cue_widths({**cues, **change}, data)
