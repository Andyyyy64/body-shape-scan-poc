import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires model runtime")
class ContourDistanceTests(unittest.TestCase):
    def test_subpixel_distances_at_image_coordinates_are_not_cancelled(self):
        import torch

        from body_scan.contour_distance import pairwise_distances

        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(914)
        points = torch.rand(1, 256, 2, generator=generator) * 400 + 100
        distances = pairwise_distances(points, points)
        diagonal = distances.diagonal(dim1=-2, dim2=-1)
        self.assertEqual(float(diagonal.max()), 0.0)
        moved = points + points.new_tensor([0.25, 0.0])
        measured = pairwise_distances(points, moved).diagonal(dim1=-2, dim2=-1)
        torch.testing.assert_close(
            measured, torch.full_like(measured, 0.25), atol=1e-5, rtol=0
        )

    def test_boundary_distance_is_invariant_to_sampling_along_straight_edges(self):
        import torch

        from body_scan.contour_distance import nearest_contour_distances

        corners = torch.tensor(
            [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]]
        )
        dense = torch.tensor(
            [
                [
                    [100.0, 100.0],
                    [125.0, 100.0],
                    [175.0, 100.0],
                    [200.0, 100.0],
                    [200.0, 150.0],
                    [200.0, 200.0],
                    [150.0, 200.0],
                    [100.0, 200.0],
                    [100.0, 150.0],
                ]
            ]
        )
        f, b = nearest_contour_distances(dense, corners, "segments")
        self.assertEqual(float(f.max()), 0)
        self.assertEqual(float(b.max()), 0)
        sampled_f, _ = nearest_contour_distances(dense, corners, "direct")
        self.assertGreater(float(sampled_f.max()), 25)

    def test_subpixel_boundary_displacement_has_a_finite_correct_gradient(self):
        import torch

        from body_scan.contour_distance import nearest_contour_distances

        target = torch.tensor(
            [[[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]]]
        )
        offset = torch.tensor(0.25, requires_grad=True)
        projection = torch.stack([torch.stack([offset + 150, offset * 0 + 100])])[None]
        # A tangential displacement stays on the boundary.
        tangential, _ = nearest_contour_distances(projection, target, "segments")
        self.assertEqual(float(tangential.max()), 0)
        projection = torch.stack([torch.stack([offset * 0 + 150, 100 - offset])])[None]
        normal, _ = nearest_contour_distances(projection, target, "segments")
        torch.testing.assert_close(normal, torch.tensor([[0.25]]))
        normal.sum().backward()
        self.assertAlmostEqual(float(offset.grad), 1.0)
