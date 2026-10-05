import unittest

import numpy as np

from body_scan.io import InputError
from body_scan.local_shape import torso_basis, transport_basis


class LocalShapeTests(unittest.TestCase):
    def test_canonical_residual_follows_known_nonrigid_vertex_transport(self):
        rng = np.random.default_rng(37)
        rest = rng.normal(0, 0.01, (8, 3, 45))
        transforms = np.repeat(np.eye(3)[None], 8, 0)
        transforms[:, 0, 0] = np.linspace(0.9, 1.1, 8)
        transforms[:, 0, 1] = np.linspace(-0.1, 0.1, 8)
        posed = transforms @ rest
        local = rng.normal(0, 0.03, (8, 3, 2))
        result = transport_basis(rest, posed, local)
        np.testing.assert_allclose(result, transforms @ local, atol=1e-7)

    def test_unidentifiable_or_inconsistent_transport_is_rejected(self):
        rest = np.zeros((1, 3, 45))
        rest[0, 0, 0] = 1
        local = np.ones((1, 3, 1))
        with self.assertRaises(InputError):
            transport_basis(rest, rest, local)
        rng = np.random.default_rng(41)
        rest = rng.normal(0, 0.01, (1, 3, 45))
        posed = rest.copy()
        posed[0, 0, 17] += 0.01
        with self.assertRaises(InputError):
            transport_basis(rest, posed, local)
        np.testing.assert_array_equal(
            transport_basis(rest, posed, np.zeros_like(local)), 0
        )

    def test_basis_is_rotation_equivariant_and_preserves_connections(self):
        anchors = np.array([[-0.2, 1, 0], [0.2, 1, 0], [-0.1, 0, 0], [0.1, 0, 0]])
        vertices = np.array(
            [[0.15, 0.32, 0.1], [0.1, 0, 0.1], [0.1, 1, 0.1], [0.8, 0.32, 0]]
        )
        basis = torso_basis(vertices, anchors)
        self.assertGreater(np.linalg.norm(basis[0]), 0)
        np.testing.assert_array_equal(basis[1:], 0)
        angle = 0.6
        rotation = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        translated = vertices @ rotation.T + [1, 2, 3]
        moved = torso_basis(translated, anchors @ rotation.T + [1, 2, 3])
        expected = np.einsum("ij,njd->nid", rotation, basis)
        np.testing.assert_allclose(moved, expected, atol=1e-7)
        with self.assertRaises(InputError):
            torso_basis(vertices, np.zeros((4, 3)))
