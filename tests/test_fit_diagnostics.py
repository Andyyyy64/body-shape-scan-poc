import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class FitDiagnosticsTests(unittest.TestCase):
    def test_attribution_does_not_change_fit_or_accumulate_gradients(self):
        import numpy as np
        import torch

        from body_scan.fit_diagnostics import objective_snapshot
        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        base = np.stack([0.2 * np.cos(angles), 0.3 * np.sin(angles), angles * 0], -1)
        base = base[None].astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[0, :, 0, 0] = np.cos(angles)
        camera = np.array([[0, 0, 2]], "float32")
        moved = base + basis[..., 0] * 0.01 + camera[:, None]
        target = moved[..., :2] / moved[..., 2:] * 430 + [320, 320]
        args = (base, basis, target, camera, [430], [[320, 320]], [0.1])
        original = fit_observations(*args, optimize_camera=False, steps=100)
        diagnosed = fit_observations(
            *args, optimize_camera=False, steps=100, diagnostics=True
        )
        np.testing.assert_array_equal(original["shape_delta"], diagnosed["shape_delta"])
        self.assertEqual(original["final_loss_px"], diagnosed["final_loss_px"])
        d = diagnosed["objective_diagnostics"]["final"]
        self.assertAlmostEqual(
            d["total"]["value"],
            sum(v["value"] for k, v in d.items() if k != "total"),
            places=5,
        )
        self.assertGreater(d["shape_prior"]["gradient_norm"]["shape"], 0)
        x = torch.tensor([2.0], requires_grad=True)
        snapshot = objective_snapshot({"loss": x.square().sum()}, {"shape": x})
        self.assertEqual(snapshot["loss"]["gradient_norm"]["shape"], 4)
        self.assertIsNone(x.grad)
