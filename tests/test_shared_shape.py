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
    def test_zero_confidence_landmark_cannot_move_fitted_shape(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        base = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], -1)[
            None
        ].astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[0, :, 0, 0] = np.cos(t)
        anchors = np.array(
            [[[-0.2, -0.3, 0], [0.2, -0.3, 0], [-0.16, 0.3, 0], [0.16, 0.3, 0]]],
            "float32",
        )

        def project(points):
            p = points + [0, 0, 2]
            return p[:, :, :2] / p[:, :, 2:] * 700 + 320

        target = project(base + basis[..., 0] * 0.01)
        correct = project(anchors)
        bad = correct.copy()
        bad[0, 3] += [1000, -1000]
        kwargs = dict(
            anchor_base=anchors,
            anchor_basis=np.zeros((*anchors.shape, 1), "float32"),
            anchor_weight=4,
            anchor_weights=[[1, 1, 1, 0]],
            shape_weight=0,
            steps=300,
            distance_mode="direct",
        )
        a = fit_observations(
            base,
            basis,
            target,
            [[0, 0, 2]],
            [700],
            [[320, 320]],
            [0.1],
            anchor_target=correct,
            **kwargs,
        )
        b = fit_observations(
            base,
            basis,
            target,
            [[0, 0, 2]],
            [700],
            [[320, 320]],
            [0.1],
            anchor_target=bad,
            **kwargs,
        )
        np.testing.assert_array_equal(a["shape_delta"], b["shape_delta"])
        np.testing.assert_array_equal(a["camera_delta"], b["camera_delta"])
        self.assertEqual(a["anchor_rms_px"], b["anchor_rms_px"])
        self.assertLess(abs(float(b["shape_delta"][0]) - 0.01), 0.002)

    def test_shared_depth_recovers_change_with_frame_depth_jitter(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], -1)
        base = np.repeat(ring[None], 3, 0).astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[:, :, 0, 0] = np.cos(t)
        anchors = np.repeat(
            np.array(
                [[-0.2, -0.3, 0], [0.2, -0.3, 0], [-0.16, 0.3, 0], [0.16, 0.3, 0]],
                "float32",
            )[None],
            3,
            0,
        )
        camera = np.array([[0, 0, 1.98], [0, 0, 2], [0, 0, 2.02]], "float32")
        focal, center = np.full(3, 700, "float32"), np.full((3, 2), 320, "float32")

        def project(points):
            points = points + [0, 0, 2]
            return points[:, :, :2] / points[:, :, 2:] * 700 + 320

        result = fit_observations(
            base,
            basis,
            project(base + basis[..., 0] * 0.01),
            camera,
            focal,
            center,
            [0.1],
            anchor_base=anchors,
            anchor_basis=np.zeros((*anchors.shape, 1), "float32"),
            anchor_target=project(anchors),
            anchor_weight=4,
            shape_weight=0,
            camera_mode="shared_depth",
            distance_mode="direct",
            contour_softening=0.05,
            steps=600,
        )
        depths = camera[:, 2] + result["camera_delta"][:, 2]
        self.assertLess(np.ptp(depths), 1e-6)
        self.assertLess(abs(float(result["shape_delta"][0]) - 0.01), 0.001)
        self.assertLess(abs(float(depths[0]) - 2), 0.002)

    def test_shared_camera_recovers_signed_change_with_xyz_estimate_jitter(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        angle = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        ring = np.stack([0.3 * np.cos(angle), 0.5 * np.sin(angle), angle * 0], -1)
        base = np.tile(ring[None], (3, 1, 1)).astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[:, :, 0, 0] = np.cos(angle)
        anchors = np.tile(
            np.array(
                [[-0.2, -0.3, 0], [0.2, -0.3, 0], [-0.16, 0.3, 0], [0.16, 0.3, 0]],
                "float32",
            )[None],
            (3, 1, 1),
        )
        for depth, change in [(1.5, -0.01), (2, 0), (2.5, 0.01)]:
            with self.subTest(depth=depth, change=change):
                truth_camera = np.array([0, 0, depth], "float32")
                camera = truth_camera + np.array(
                    [[-0.02, 0.015, -0.02], [0, 0, 0], [0.025, -0.02, 0.02]], "float32"
                )

                def project(points):
                    position = points + truth_camera
                    return position[..., :2] / position[..., 2:] * 700 + 320

                result = fit_observations(
                    base,
                    basis,
                    project(base + basis[..., 0] * change),
                    camera,
                    np.full(3, 700),
                    np.full((3, 2), 320),
                    [0.1],
                    anchor_base=anchors,
                    anchor_basis=np.zeros((*anchors.shape, 1), "float32"),
                    anchor_target=project(anchors),
                    anchor_weight=4,
                    shape_weight=0,
                    camera_mode="shared",
                    distance_mode="direct",
                    contour_softening=0.05,
                    steps=600,
                )
                fitted_camera = camera + result["camera_delta"]
                self.assertLess(np.max(np.ptp(fitted_camera, axis=0)), 1e-6)
                self.assertLess(abs(float(result["shape_delta"][0]) - change), 0.001)
                self.assertLess(np.max(abs(fitted_camera - truth_camera)), 0.003)

    def test_local_residual_cannot_escape_anatomical_bound(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        base = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], -1)[
            None
        ].astype("float32")
        basis = np.zeros((*base.shape, 2), "float32")
        basis[0, :, 0, 1] = base[0, :, 0]
        p = base + basis[..., 1] * 0.5 + [0, 0, 2]
        target = p[:, :, :2] / p[:, :, 2:] * 700 + 320
        result = fit_observations(
            base,
            basis,
            target,
            [[0, 0, 2]],
            [700],
            [[320, 320]],
            [0.1, 0.1],
            optimize_camera=False,
            local_count=1,
            global_shape=False,
            local_weight=0,
            distance_mode="direct",
            steps=400,
        )
        self.assertEqual(float(result["shape_delta"][0]), 0)
        self.assertGreater(float(result["shape_delta"][1]), 0.09)
        self.assertLessEqual(float(result["shape_delta"][1]), 0.100001)

    def test_landmark_constrained_yaw_does_not_become_shape_change(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        ring = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), 0.2 * np.sin(t)], -1)
        direction = np.stack([np.cos(t), np.zeros_like(t), np.zeros_like(t)], -1)
        joints = np.array(
            [[-0.2, -0.3, 0], [0.2, -0.3, 0], [-0.16, 0.3, 0], [0.16, 0.3, 0]]
        )

        def rotate(points, yaw):
            c, s = np.cos(yaw), np.sin(yaw)
            return points @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])

        angles = [0, 0.6, 1.2]
        base = np.stack([rotate(ring, a) for a in angles]).astype("float32")
        basis = np.stack([rotate(direction, a) for a in angles])[..., None].astype(
            "float32"
        )
        anchors = np.stack([rotate(joints, a) for a in angles]).astype("float32")
        camera = np.array([[0, 0, 2], [0, 0, 2.5], [0, 0, 3]], "float32")
        focal = np.array([700] * 3, "float32")
        center = np.array([[320, 320]] * 3, "float32")

        def project(points):
            points = points + camera[:, None, :]
            return (
                points[:, :, :2] / points[:, :, 2:] * focal[:, None, None]
                + center[:, None, :]
            )

        yaw, change = np.deg2rad(3), 0.02
        target = project(
            np.stack([rotate(x, yaw) for x in base + basis[..., 0] * change])
        )
        anchor_target = project(np.stack([rotate(x, yaw) for x in anchors]))
        kwargs = dict(shape_weight=0, camera_weight=0.1, steps=600)
        original = fit_observations(
            base, basis, target, camera, focal, center, [0.1], **kwargs
        )
        refined = fit_observations(
            base,
            basis,
            target,
            camera,
            focal,
            center,
            [0.1],
            anchor_base=anchors,
            anchor_basis=np.zeros((*anchors.shape, 1), "float32"),
            anchor_target=anchor_target,
            anchor_weight=4,
            pose_limit_deg=5,
            pose_weight=0.01,
            **kwargs,
        )
        self.assertGreater(abs(float(original["shape_delta"][0]) - change), 0.005)
        self.assertLess(abs(float(refined["shape_delta"][0]) - change), 0.001)
        self.assertLess(refined["anchor_rms_px"], 0.1)
        self.assertTrue(np.all(abs(refined["yaw_delta_deg"] - 3) < 0.1))

    def test_normalization_removes_distance_dependent_prior_strength(self):
        import numpy as np
        import torch

        from body_scan.shared_shape import fit_observations

        torch.set_num_threads(2)
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        base = np.stack([0.3 * np.cos(t), 0.5 * np.sin(t), np.zeros_like(t)], -1)[
            None
        ].astype("float32")
        basis = np.zeros((*base.shape, 1), "float32")
        basis[:, :, 0, 0] = np.cos(t)
        results = {False: [], True: []}
        for distance in (1.5, 2.5):
            camera = np.array([[0, 0, distance]], "float32")
            focal, center = (
                np.array([700], "float32"),
                np.array([[320, 240]], "float32"),
            )
            points = base + basis[:, :, :, 0] * 0.02 + camera[:, None, :]
            target = (
                points[:, :, :2] / points[:, :, 2:] * focal[:, None, None]
                + center[:, None, :]
            )
            for normalized in (False, True):
                fit = fit_observations(
                    base,
                    basis,
                    target,
                    camera,
                    focal,
                    center,
                    [0.1],
                    steps=250,
                    optimize_camera=False,
                    shape_weight=10,
                    observation_scale=[700 / distance / 256] if normalized else None,
                )
                results[normalized].append(float(fit["shape_delta"][0]))
                self.assertGreater(fit["final_loss_px"], 0)
        self.assertGreater(abs(results[False][0] - results[False][1]), 0.001)
        self.assertLess(abs(results[True][0] - results[True][1]), 0.0001)

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
