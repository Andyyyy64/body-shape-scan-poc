import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("cv2"), "requires vision runtime")
class LocalContourRefreshTests(unittest.TestCase):
    def test_refresh_projects_local_fit_without_passing_residuals_to_mhr(self):
        import cv2
        import numpy as np

        from body_scan.contours import contour_observation
        from body_scan.fit_strategy import build_observations
        from body_scan.mesh import synthetic_mesh

        mesh = synthetic_mesh()
        vertices = np.asarray(mesh["vertices"], np.float32)
        vertices[:, 1] -= np.mean([vertices[:, 1].min(), vertices[:, 1].max()])
        vertices[:, 1] /= np.ptp(vertices[:, 1])
        vertices[:, 0] *= 0.6
        vertices[:, 2] *= 0.6
        faces = np.asarray(mesh["faces"])
        joints = np.zeros((11, 3), np.float32)
        joints[[5, 6, 9, 10]] = [
            [-0.15, 0.35, 0],
            [0.15, 0.35, 0],
            [-0.1, -0.35, 0],
            [0.1, -0.35, 0],
        ]
        camera = np.array([0, 0, 2], np.float32)
        sign = np.array([1, -1, -1])
        position = vertices * sign + camera
        projected = position[:, :2] / position[:, 2:] * 430 + 320
        mask = np.zeros((640, 640), np.uint8)
        for face in faces:
            cv2.fillConvexPoly(mask, projected[face].astype("int32"), 1)
        target = joints * sign + camera
        target = target[:, :2] / target[:, 2:] * 430 + 320
        value = {
            "mhr_model_params": np.zeros(204, np.float32),
            "pred_cam_t": camera,
            "focal_length": 430.0,
            "pred_keypoints_2d": target,
        }
        initial = np.zeros(45, np.float32)
        seen = []

        def generate(shape, params):
            self.assertEqual(len(shape), 45)
            return vertices + np.asarray(shape[:3]), joints + np.asarray(shape[:3])

        def observe(*args, **kwargs):
            seen.append(kwargs)
            return contour_observation(*args, **kwargs)

        fit = {
            "shape_delta": np.r_[np.zeros(45), 0.1, -0.1],
            "camera_delta": np.array([[0.01, -0.02, 0.03]]),
        }
        research = {
            "fit": {
                "local_mode": "independent",
                "local_transport": "canonical",
                "shared_skeleton": True,
                "anchor_weight": 1,
                "refresh_contour": 1,
            }
        }
        with patch("body_scan.fit_strategy.contour_observation", side_effect=observe):
            rows = build_observations(
                [(0, value, mask)],
                initial,
                np.zeros(68),
                faces,
                generate,
                np.arange(len(vertices)),
                research,
                selection_shape=initial,
                selection_fit=fit,
            )
        self.assertEqual(len(rows), 1)
        updated = seen[0]["selection_vertices"]
        base = vertices * sign
        belly = abs(vertices[:, 1] - (-0.35 + 0.32 * 0.7)) < 0.04
        self.assertTrue(belly.any())
        self.assertGreater(
            float(np.sum(abs(updated[belly, 0]))), float(np.sum(abs(base[belly, 0])))
        )
        self.assertLess(
            float(np.sum(abs(updated[belly, 2]))), float(np.sum(abs(base[belly, 2])))
        )
        np.testing.assert_allclose(
            seen[0]["selection_camera"], camera + fit["camera_delta"][0]
        )
