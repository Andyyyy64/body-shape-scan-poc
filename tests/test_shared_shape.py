import importlib.util
import unittest

from body_scan.io import InputError
from body_scan.mesh import compare_meshes, synthetic_mesh


class MethodTests(unittest.TestCase):
    def test_mixed_methods_are_rejected(self):
        a, b = synthetic_mesh(), synthetic_mesh()
        a["method"], b["method"] = "initial", "refined"
        with self.assertRaises(InputError):
            compare_meshes(a, b)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class SharedShapeTests(unittest.TestCase):
    def test_camera_motion_does_not_erase_known_shape_change(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], axis=-1)
        base = np.repeat(ring[None], 3, axis=0).astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[:, :, 0, 0] = np.cos(t)
        camera = np.array([[0, 0, 3], [0.15, -0.1, 2.8], [-0.1, 0.2, 3.2]], "float32")
        focal = np.full(3, 700, "float32")
        center = np.repeat([[320, 240]], 3, axis=0).astype("float32")

        def solve(change, order):
            points = base + basis[:, :, :, 0] * change + camera[:, None, :]
            target = (
                points[:, :, :2] / points[:, :, 2:] * focal[:, None, None]
                + center[:, None, :]
            )
            return fit_observations(
                base[order],
                basis[order],
                target[order],
                camera[order],
                focal[order],
                center[order],
                [0.1],
                optimize_camera=True,
                shape_weight=0,
                steps=250,
            )["shape_delta"][0]

        unchanged = solve(0, [0, 1, 2])
        changed = solve(0.02, [0, 1, 2])
        reordered = solve(0.02, [2, 0, 1])
        self.assertLess(abs(unchanged), 0.001)
        self.assertLess(abs(changed - 0.02), 0.001)
        self.assertLess(abs(changed - reordered), 0.0001)
        with self.assertRaises(InputError):
            fit_observations(
                base, basis, np.zeros((3, 48, 2)), camera, focal, center, [float("nan")]
            )
