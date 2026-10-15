"""Pixel contour distances with explicit numerical precision strategy."""

from __future__ import annotations

from .io import InputError


def pairwise_distances(projection, target, mode="direct"):
    import torch

    if mode not in ("legacy", "direct"):
        raise InputError("Unknown contour distance mode")
    return torch.cdist(
        projection,
        target,
        compute_mode="donot_use_mm_for_euclid_dist"
        if mode == "direct"
        else "use_mm_for_euclid_dist_if_necessary",
    )


def nearest_contour_distances(projection, target, mode="direct"):
    """Directional distances; segments require perimeter-ordered closed polygons."""
    import torch

    if mode != "segments":
        distances = pairwise_distances(projection, target, mode)
        return distances.min(2).values, distances.min(1).values

    def nearest(points, polygon):
        # The discrete minimum has the selected segment's piecewise gradient.
        # Select without retaining a quadratic-size autograd graph.
        with torch.no_grad():
            start = polygon[:, None]
            edges = polygon.roll(-1, dims=1)[:, None] - start
            relative = points[:, :, None] - start
            fraction = (relative * edges).sum(-1) / edges.square().sum(-1).clamp_min(
                1e-12
            )
            residual = relative - fraction.clamp(0, 1)[..., None] * edges
            index = residual.square().sum(-1).argmin(-1)
        selected = index[..., None].expand(-1, -1, 2)
        start = polygon.gather(1, selected)
        edges = polygon.roll(-1, dims=1).gather(1, selected) - start
        relative = points - start
        fraction = (relative * edges).sum(-1) / edges.square().sum(-1).clamp_min(1e-12)
        return torch.linalg.vector_norm(
            relative - fraction.clamp(0, 1)[..., None] * edges, dim=-1
        )

    return nearest(projection, target), nearest(target, projection)
