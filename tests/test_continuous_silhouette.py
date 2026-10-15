"""Physical visibility, perspective and clipping invariants for research contours."""

import importlib.util
import unittest

try:
    import numpy as np

    from body_scan.silhouette_edges import (
        VisibilityGrid,
        continuous_outer_contour,
        guided_primitives,
        interpolate,
    )
except ImportError:
    np = None


@unittest.skipIf(
    np is None or importlib.util.find_spec("shapely") is None,
    "NumPy and Shapely research dependencies are required",
)
class VisibleEdgeGeometryTests(unittest.TestCase):
    def test_perspective_midpoint_is_not_3d_midpoint(self):
        position = np.array([[-0.2, 0, 1], [0.4, 0, 4], [0, 0.5, 2]])
        projected = position[:, :2] / position[:, 2:] * 100 + 320
        edge, weight, error = guided_primitives(
            np.array([[315.0, 320.0]]),
            position,
            projected,
            np.array([[0, 1, 2]]),
            visibility=True,
        )
        point = interpolate(position, edge, weight)[0]
        np.testing.assert_allclose(point, [-0.08, 0, 1.6], atol=1e-12)
        np.testing.assert_allclose(point[:2] / point[2], [-0.05, 0], atol=1e-12)
        self.assertLess(float(error.max()), 1e-12)

    def test_external_union_and_occlusion_are_both_required(self):
        position = np.array([[-0.2, 0, 1], [0.4, 0, 4], [0, 0.5, 2], [-0.5, 0, 10]])
        projected = position[:, :2] / position[:, 2:] * 100 + 320
        faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
        grid = VisibilityGrid(position, projected, faces, (640, 640))
        self.assertTrue(grid.external_visible(np.array([315.0, 320.0]), 1.6))
        self.assertFalse(grid.external_visible(np.array([315.0, 320.0]), 10))
        self.assertFalse(grid.external_visible(np.array([320.0, 330.0]), 1.5))
        edge, weight, _ = guided_primitives(
            np.array([[315.0, 320.0]]), position, projected, faces, visibility=True
        )
        self.assertLess(float(interpolate(position, edge, weight)[0, 2]), 2)

    def test_subdivision_does_not_move_continuous_boundary(self):
        position = np.array([[-0.2, 0, 1], [0.4, 0, 4], [0, 0.5, 2]])
        divided = np.concatenate([position, ((position[0] + position[1]) / 2)[None]])
        points = np.array([[-0.17, 0], [-0.05, 0], [0.06, 0]]) * 100 + 320
        outputs = []
        for v, faces in [
            (position, np.array([[0, 1, 2]])),
            (divided, np.array([[0, 3, 2], [3, 1, 2]])),
        ]:
            edge, weight, _ = guided_primitives(
                points, v, v[:, :2] / v[:, 2:] * 100 + 320, faces, visibility=True
            )
            outputs.append(interpolate(v, edge, weight))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)


@unittest.skipIf(
    np is None or importlib.util.find_spec("shapely") is None,
    "NumPy and Shapely research dependencies are required",
)
class ContinuousUnionTests(unittest.TestCase):
    def test_overlapping_triangles_have_no_raster_slit_or_hidden_outer_sample(self):
        uv = np.array(
            [
                [300.1, 310.2],
                [340.3, 310.2],
                [320.2, 350.1],
                [310.1, 310.1],
                [350.3, 310.1],
                [330.2, 350.2],
            ]
        )
        z = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        pos = np.column_stack([(uv - 320) * z[:, None] / 100, z])
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        points = continuous_outer_contour(uv, faces, 640, 640)
        edges, weights, error = guided_primitives(
            points, pos, uv, faces, visibility=True
        )
        self.assertLess(error.max(), 1e-09)
        estimated = interpolate(pos, edges, weights)
        grid = VisibilityGrid(pos, uv, faces, (640, 640))
        self.assertTrue(
            all(
                (
                    grid.external_visible(p[:2] / p[2] * 100 + 320, p[2])
                    for p in estimated
                )
            )
        )

    def test_triangle_subdivision_preserves_continuous_external_shape(self):
        uv = np.array([[300.0, 300], [340.0, 300], [320.0, 350]])
        divided = np.r_[uv, [(uv[0] + uv[1]) / 2]]
        a = continuous_outer_contour(uv, np.array([[0, 1, 2]]), 640, 640)
        b = continuous_outer_contour(
            divided, np.array([[0, 3, 2], [3, 1, 2]]), 640, 640
        )
        import shapely

        self.assertLess(shapely.Polygon(a).hausdorff_distance(shapely.Polygon(b)), 0.01)


@unittest.skipIf(
    np is None or importlib.util.find_spec("shapely") is None,
    "NumPy and Shapely research dependencies are required",
)
class ViewportContourTests(unittest.TestCase):
    def test_predicted_boundary_outside_camera_does_not_discard_visible_points(self):
        uv = np.array([[1.0, 300], [50.0, 300], [25.0, 350]])
        positions = np.column_stack([(uv - 320) / 100, np.ones(3)])
        faces = np.array([[0, 1, 2]])
        points = continuous_outer_contour(uv, faces, 640, 640)
        self.assertTrue((points > 2).all())
        self.assertTrue((points < [637, 637]).all())
        edge, weights, error = guided_primitives(
            points, positions, uv, faces, visibility=True
        )
        self.assertLess(error.max(), 1e-10)
        np.testing.assert_allclose(
            interpolate(positions, edge, weights)[:, :2] * 100 + 320, points, atol=1e-10
        )
