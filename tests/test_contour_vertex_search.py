import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("scipy"), "requires model runtime")
class BoundarySearchTests(unittest.TestCase):
    def test_spatial_search_preserves_nearest_indices_and_lowest_index_ties(self):
        import numpy as np

        from body_scan.contours import _nearest_vertex

        vertices = np.array(
            [[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
        )
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
        self.assertEqual(
            _nearest_vertex(points, vertices, "spatial").tolist(), [0, 0, 0, 1, 3]
        )
        rng = np.random.default_rng(418)
        vertices = rng.normal(size=(1200, 2)) * 200 + 320
        vertices[100:110] = vertices[30:40]
        points = np.vstack([vertices[100:110], rng.normal(size=(300, 2)) * 200 + 320])
        np.testing.assert_array_equal(
            _nearest_vertex(points, vertices, "spatial"),
            _nearest_vertex(points, vertices),
        )
