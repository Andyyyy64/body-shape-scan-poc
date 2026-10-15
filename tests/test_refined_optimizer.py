"""Known signed changes should survive optional double precision refinement."""

import importlib.util
import unittest

import numpy as np

from body_scan.shared_shape import fit_observations


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class RefinedOptimizerTests(unittest.TestCase):
    def observations(self, delta):
        angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        base = np.stack([0.3 * np.cos(angle), 0.5 * np.sin(angle), np.zeros(48)], -1)[
            None
        ]
        basis = np.zeros((*base.shape, 1))
        basis[0, :, 0, 0] = np.cos(angle)
        target = (base + basis[..., 0] * delta)[:, :, :2] / 2 * 430 + 320
        return [base, basis, target, [[0, 0, 2]], [430], [[320, 320]], [0.1]]

    def test_signed_exact_model_recovery_and_stationarity(self):
        import torch

        torch.set_num_threads(1)
        for delta in [-0.01, 0.01]:
            fit = fit_observations(
                *self.observations(delta),
                steps=30,
                shape_weight=0,
                camera_mode="fixed",
                distance_mode="segments",
                optimizer_mode="adam_lbfgs64",
            )
            self.assertAlmostEqual(float(fit["shape_delta"][0]), delta, places=7)
            self.assertLess(fit["optimizer_refinement"]["gradient_inf_norm"], 1e-06)
            self.assertLessEqual(
                fit["optimizer_refinement"]["objective_after"],
                fit["optimizer_refinement"]["objective_before"],
            )
