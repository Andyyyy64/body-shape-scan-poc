import importlib.util
import unittest

import numpy as np

from body_scan.io import InputError
from body_scan.research import (
    fit_region,
    method_id,
    options,
    relative_fit_error,
    torso_mask_support,
)


class ResearchTests(unittest.TestCase):
    def test_common_measurement_bridge_is_explicit_and_rejects_discarded_trials(self):
        common = {
            "distance_mode": "segments",
            "contour_primitive": "continuous_visible_edge64_viewport",
            "contour_samples": 256,
            "optimizer_mode": "adam_lbfgs64",
        }
        self.assertEqual(options({"fit": common})["fit"], common)
        self.assertNotEqual(method_id({"fit": common}), method_id({}))
        for invalid in (
            {**common, "distance_mode": "direct"},
            {**common, "contour_samples": 257},
            {**common, "optimizer_mode": "unsupported"},
            {**common, "contour_primitive": "edge"},
            {**common, "distance_mode": "indexed_segments"},
            {**common, "width_band_weight": 1},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(InputError):
                options({"fit": invalid})

    def test_nuisance_options_do_not_allow_unconstrained_pose_or_unknown_flags(self):
        for invalid in (
            {"pose_limit_deg": 5},
            {"shared_skeleton": 1},
            {"anchor_weight": float("nan")},
            {"refresh_contour": 0.5},
            {"normalize_observation": "yes"},
            {"pose_limit_deg": 11},
            {"oracle_yaw": 0},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(InputError):
                options({"fit": invalid})
        self.assertEqual(
            options({"fit": {"anchor_weight": 1, "pose_limit_deg": 5}})["fit"][
                "pose_limit_deg"
            ],
            5,
        )

    def test_intrinsics_reject_unsupported_and_nonfinite_values(self):
        camera = dict(
            width=640,
            height=640,
            fx=430,
            fy=430,
            cx=320,
            cy=320,
            provenance="synthetic_renderer",
        )
        self.assertEqual(options({"camera": camera})["camera"], camera)
        for key, value in (
            ("fx", float("nan")),
            ("width", True),
            ("width", 640.0),
            ("fy", 431),
            ("cx", 319),
            ("provenance", ""),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(InputError):
                options({"camera": {**camera, key: value}})
        with self.assertRaises(InputError):
            options({"qc": {"max_relative_fit_error": 0}})
        with self.assertRaises(InputError):
            options({"region_weight": float("inf")})

    def test_method_identity_is_stable_but_distinguishes_variants(self):
        self.assertEqual(
            method_id({"region": "abdomen", "region_weight": 8}),
            method_id({"region_weight": 8, "region": "abdomen"}),
        )
        self.assertNotEqual(
            method_id({}), method_id({"qc": {"require_rotation": True}})
        )

    def test_imputed_core_inside_image_cannot_replace_mask_support(self):
        points = np.full((70, 2), 50.0)
        points[[5, 6]] = [[40, 30], [60, 30]]
        points[[9, 10]] = [[43, 70], [57, 70]]
        mask = np.zeros((100, 100), bool)
        mask[20:80, 35:65] = True
        self.assertTrue(torso_mask_support(points, mask)["supported"])
        mask[45:56, 35:65] = False
        result = torso_mask_support(points, mask)
        self.assertFalse(result["supported"])
        self.assertEqual(result["coverage"][1], 0)
        # A mask-filled coat still passes: this check does not prove skin visibility.
        self.assertTrue(torso_mask_support(points, np.ones_like(mask))["supported"])

    def test_fit_error_is_normalized_and_cannot_accept_empty_observations(self):
        self.assertAlmostEqual(relative_fit_error(6, [290, 310]), 0.02)
        self.assertAlmostEqual(relative_fit_error(12, [580, 620]), 0.02)
        for heights in ([], [0], [float("nan")]):
            with self.assertRaises(InputError):
                relative_fit_error(6, heights)

    def test_core_outside_each_image_edge_is_unsupported_without_crashing(self):
        mask = np.ones((100, 100), bool)
        for x, y in ((-300, 50), (300, 50), (50, -300), (50, 300)):
            with self.subTest(x=x, y=y):
                points = np.full((70, 2), [x, y], dtype=float)
                points[[5, 6], 1] -= 20
                points[[9, 10], 1] += 20
                result = torso_mask_support(points, mask)
                self.assertFalse(result["supported"])
                self.assertEqual(result["coverage"], [0, 0, 0])

    def test_region_is_defined_by_anatomy_and_excludes_arms(self):
        joints = np.zeros((70, 3))
        joints[[5, 6]] = [[-0.25, 1, 0], [0.25, 1, 0]]
        points = np.array(
            [
                [x, y, 0]
                for x in np.linspace(-0.5, 0.5, 30)
                for y in np.linspace(0, 1, 30)
            ]
        )
        indices = fit_region(points, joints, "abdomen")
        selected = points[indices]
        self.assertGreaterEqual(selected[:, 1].min(), 0.1)
        self.assertLessEqual(selected[:, 1].max(), 0.65)
        self.assertLessEqual(abs(selected[:, 0]).max(), 0.25)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class RegionObjectiveTests(unittest.TestCase):
    def test_weighting_resolves_conflicting_local_and_global_observations(self):
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        base = np.zeros((1, 8, 3), np.float32)
        base[0, :4, 1] = -0.4
        base[0, 4:, 1] = 0.4
        basis = np.zeros((*base.shape, 1), np.float32)
        basis[..., 0, 0] = 1
        camera = np.array([[0, 0, 3]], np.float32)
        target = base[..., :2].copy()
        target[0, :4, 0], target[0, 4:, 0] = 0.04, -0.04
        target = target * 300 + [320, 320]
        args = (base, basis, target, camera, [900], [[320, 320]], [0.1])
        plain = fit_observations(*args, optimize_camera=False, shape_weight=0)
        weights = np.array([[8] * 4 + [1] * 4], np.float32)
        local = fit_observations(
            *args,
            optimize_camera=False,
            shape_weight=0,
            base_weights=weights,
            target_weights=weights,
        )
        self.assertLess(abs(float(plain["shape_delta"][0])), 0.001)
        self.assertGreater(float(local["shape_delta"][0]), 0.025)
        self.assertGreater(local["final_unweighted_loss_px"], local["final_loss_px"])
        for invalid in (
            np.zeros((1, 8)),
            np.ones((1, 7)),
            np.full((1, 8), float("nan")),
        ):
            with self.assertRaises(InputError):
                fit_observations(*args, base_weights=invalid)


if __name__ == "__main__":
    unittest.main()
