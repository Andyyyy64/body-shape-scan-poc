import importlib.util
import unittest

VISION = importlib.util.find_spec("cv2") is not None
TORCH = importlib.util.find_spec("torch") is not None


def ellipsoid_mesh():
    """Closed convex mesh (m) whose only shape dimension widens it along x."""
    import numpy as np

    rows, cols = 24, 48
    lat = np.linspace(-np.pi / 2, np.pi / 2, rows)
    lon = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    vertices = np.array(
        [
            [
                0.18 * np.cos(a) * np.cos(b),
                0.45 * np.sin(a),
                0.12 * np.cos(a) * np.sin(b),
            ]
            for a in lat
            for b in lon
        ],
        np.float32,
    )
    faces = []
    for i in range(rows - 1):
        for j in range(cols):
            a, b = i * cols + j, i * cols + (j + 1) % cols
            c, d = a + cols, b + cols
            faces += [[a, b, c], [b, d, c]]

    def generate(shape):
        out = vertices.copy()
        out[:, 0] *= 1 + float(shape[0])
        return out

    return generate, np.asarray(faces)


def render_mask(generate, faces, shape, camera, focal, size=(480, 640)):
    import cv2
    import numpy as np

    height, width = size
    position = generate(shape) + np.asarray(camera, np.float32)
    projected = position[:, :2] / position[:, 2:] * focal + [width / 2, height / 2]
    raster = np.zeros((height, width), np.uint8)
    for face in faces:
        cv2.fillConvexPoly(raster, projected[face].astype("int32"), 1)
    return raster


@unittest.skipUnless(VISION, "optional vision extra not installed")
class ContourObservationTests(unittest.TestCase):
    def test_observation_shapes_and_linearity(self):
        import numpy as np

        from body_scan.contours import CONTOUR_SAMPLES, contour_observation
        from body_scan.io import InputError

        generate, faces = ellipsoid_mesh()
        camera, focal = [0.0, 0.0, 2.0], 700.0
        mask = render_mask(generate, faces, [0.0], camera, focal)
        obs = contour_observation(mask, faces, generate, [0.0], camera, focal)
        self.assertEqual(obs["base"].shape, (CONTOUR_SAMPLES, 3))
        self.assertEqual(obs["basis"].shape, (CONTOUR_SAMPLES, 3, 1))
        self.assertEqual(obs["target"].shape, (CONTOUR_SAMPLES, 2))
        self.assertTrue((obs["basis"][:, 1:, 0] == 0).all())
        self.assertTrue(np.allclose(obs["basis"][:, 0, 0], obs["base"][:, 0]))
        with self.assertRaises(InputError):
            contour_observation(mask, faces, generate, [0.0], [0, 0, -2.0], focal)
        with self.assertRaises(InputError):
            contour_observation(
                np.zeros((480, 640), np.uint8), faces, generate, [0.0], camera, focal
            )

        def nonlinear(shape):
            out = generate(shape)
            out[:, 0] *= 1 + 50 * float(shape[0]) ** 2
            return out

        with self.assertRaises(InputError):
            contour_observation(mask, faces, nonlinear, [0.0], camera, focal)

    @unittest.skipUnless(TORCH, "requires model runtime")
    def test_fit_recovers_width_change_from_silhouettes(self):
        import numpy as np

        from body_scan.contours import contour_observation
        from body_scan.shared_shape import fit_observations

        generate, faces = ellipsoid_mesh()
        cameras = np.array(
            [[0, 0, 2.0], [0.1, -0.05, 2.2], [-0.1, 0.05, 1.9]], "float32"
        )
        focal = 700.0

        def solve(true_width):
            observations = [
                contour_observation(
                    render_mask(generate, faces, [true_width], cam, focal),
                    faces,
                    generate,
                    [0.0],
                    cam,
                    focal,
                )
                for cam in cameras
            ]
            return fit_observations(
                np.array([o["base"] for o in observations]),
                np.array([o["basis"] for o in observations]),
                np.array([o["target"] for o in observations]),
                np.array([o["camera"] for o in observations]),
                np.array([o["focal"] for o in observations]),
                np.array([o["center"] for o in observations]),
                [0.1],
                shape_weight=0,
                optimize_camera=False,
            )

        unchanged = solve(0.0)
        # Rasterised boundaries and fixed contour correspondences leave a
        # constant offset even for identical geometry, so the method is only
        # differential: changes are recovered relative to the unchanged fit.
        offset = float(unchanged["shape_delta"][0])
        for true_width in (0.08, -0.05):
            changed = solve(true_width)
            recovered = float(changed["shape_delta"][0]) - offset
            self.assertLess(abs(recovered - true_width), 0.01)
            self.assertLess(changed["final_loss_px"], changed["initial_loss_px"])
