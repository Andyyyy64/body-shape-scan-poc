"""Matched local RGB/model edge curves at unchanged native pixel-row targets.

No turn-wide correction, renderer labels or deformation truth. All seven actual
neighboring pixel rows enter each fit. Both pixel-row endpoints are fitted before
the original fractional-row interpolation. Disabled calls use original code.
"""

import numpy as np
import torch

from .contour_width import pixel_row_widths
from .io import InputError
from .subpixel_edges import refine_pixel_row_positions, refine_positions


def connected_positions(polygon, rows, centers):
    """Piecewise differentiable endpoints, with the original half-open fill rule."""
    if (polygon.ndim != 3 or polygon.shape[-1] != 2 or rows.ndim != 2
            or centers.shape != rows.shape or polygon.shape[0] != rows.shape[0]
            or polygon.shape[1] < 3 or not all(torch.isfinite(a).all() for a in (polygon, rows, centers))):
        raise InputError('Invalid model edge-curve geometry')
    vertices = polygon.detach().cpu().numpy().astype(float)
    end = np.roll(vertices, -1, axis=1)
    y = rows.detach().cpu().numpy()[..., None]
    center = centers.detach().cpu().numpy()[..., None]
    crossing = ((vertices[:, None, :, 1] <= y) & (y < end[:, None, :, 1])) | ((end[:, None, :, 1] <= y) & (y < vertices[:, None, :, 1]))
    difference = end[:, :, 1] - vertices[:, :, 1]
    fraction = (y - vertices[:, None, :, 1]) / np.where(abs(difference[:, None]) > 1e-12, difference[:, None], 1.)
    x = vertices[:, None, :, 0] + fraction * (end - vertices)[:, None, :, 0]
    left = np.where(crossing & (x <= center), x, -np.inf)
    right = np.where(crossing & (x > center), x, np.inf)
    inside = (np.sum(crossing & (x <= center), axis=-1) % 2) == 1
    if not (inside & np.isfinite(left.max(-1)) & np.isfinite(right.min(-1))).all():
        raise InputError('Model neighborhood lacks a central interval; rows are not replaced')
    values = []
    for index in (left.argmax(-1), right.argmin(-1)):
        selected = torch.as_tensor(index, dtype=torch.long, device=polygon.device)
        a = polygon.gather(1, selected[..., None].expand(-1, -1, 2))
        b = polygon.roll(-1, dims=1).gather(1, selected[..., None].expand(-1, -1, 2))
        ratio = (rows - a[..., 1]) / (b[..., 1] - a[..., 1])
        values.append(a[..., 0] + ratio * (b[..., 0] - a[..., 0]))
    return torch.stack(values, dim=-1)


def local_curve(values, mode, *, max_iterations=200):
    """Fit local quadratics; robust fits use convex pseudo-Huber in actual pixels."""
    if mode not in ('quadratic', 'robust_quadratic') or values.shape[-1] != 7 or not torch.isfinite(values).all():
        raise InputError('A local quadratic needs all seven finite pixel observations')
    if not values.dtype.is_floating_point:
        raise InputError('Pixel curve coordinates must be floating point')
    offsets = torch.arange(-3, 4, dtype=values.dtype, device=values.device)
    design = torch.stack([torch.ones_like(offsets), offsets, offsets**2], dim=-1)
    origin = values[..., 3:4]
    response = values - origin
    coefficient = response @ torch.linalg.pinv(design).T
    if mode == 'quadratic':
        return coefficient[..., 0] + origin[..., 0]
    # Convex pseudo-Huber Newton steps with separate Armijo searches per curve.
    # Every one of the seven samples remains in both gradient and Hessian.
    def loss(residual):
        root = torch.sqrt(1 + residual**2)
        return torch.sum(residual**2 / (root + 1), dim=-1)

    for _ in range(max_iterations):
        residual = coefficient @ design.T - response
        root = torch.sqrt(1 + residual**2)
        gradient = (residual / root) @ design
        done = gradient.abs().max(dim=-1).values <= 1e-9
        if bool(done.all().detach()):
            return coefficient[..., 0] + origin[..., 0]
        curvature = root.pow(-3)
        hessian = torch.einsum('...n,ni,nj->...ij', curvature, design, design)
        direction = torch.linalg.solve(hessian, gradient)
        direction = torch.where(done[..., None], torch.zeros_like(direction), direction)
        derivative = torch.sum(gradient * direction, dim=-1)
        current_loss = loss(residual)
        alpha = torch.ones_like(current_loss)
        slack = 8 * torch.finfo(values.dtype).eps * (1 + current_loss)
        for _ in range(40):
            trial = coefficient - alpha[..., None] * direction
            trial_loss = loss(trial @ design.T - response)
            accepted = torch.isfinite(trial_loss) & (trial_loss <= current_loss - 1e-4 * alpha * derivative + slack)
            if bool(accepted.all().detach()):
                coefficient = trial
                break
            alpha = torch.where(accepted, alpha, alpha / 2)
        else:
            raise InputError('Local convex edge curve line search failed; no omitted rows')
    raise InputError('Local convex edge curve did not converge; no row omission or QC relabel')



class PixelCurveObserver:
    def __init__(self, mode):
        if mode not in ('original', 'quadratic', 'robust_quadratic'):
            raise InputError('Unknown fixed local edge-curve observer')
        self.mode = mode
        self.original_model = pixel_row_widths
        self.original_rgb = refine_pixel_row_positions

    def model_widths(self, polygon, rows, centers):
        if self.mode == 'original':
            return self.original_model(polygon, rows, centers)
        low, high = torch.floor(rows).detach(), torch.ceil(rows).detach()
        fraction = rows - low
        positions = []
        for target in (low, high):
            neighbors = torch.stack([connected_positions(polygon, target + offset, centers)
                                     for offset in range(-3, 4)], dim=-1)
            positions.append(local_curve(neighbors, self.mode))
        edges = positions[0] * (1 - fraction[..., None]) + positions[1] * fraction[..., None]
        widths = edges[..., 1] - edges[..., 0]
        if not torch.isfinite(widths).all() or (widths <= 0).any():
            raise InputError('Local model edge curves invert an observed width')
        return widths

    def refine(self, bgr, polygon, rows, centers):
        if self.mode == 'original':
            return self.original_rgb(bgr, polygon, rows, centers)
        from .subpixel_edges import central_positions

        rows, centers = np.asarray(rows, float), np.asarray(centers, float)
        if rows.ndim != 1 or centers.shape != rows.shape or not np.isfinite(rows).all() or not np.isfinite(centers).all():
            raise InputError('Invalid nominal local edge-curve rows')
        targets = np.stack([np.floor(rows), np.ceil(rows)], axis=-1)
        expanded = targets[..., None] + np.arange(-3, 4)
        repeated = np.broadcast_to(centers[:, None, None], expanded.shape)
        seed = central_positions(polygon, expanded.ravel(), repeated.ravel())
        result = refine_positions(bgr, expanded.ravel(), seed)
        neighbor_positions = result['positions'].reshape(len(rows), 2, 7, 2)
        fitted = local_curve(torch.as_tensor(neighbor_positions.transpose(0, 1, 3, 2)), self.mode).numpy()
        fraction = rows - np.floor(rows)
        edges = fitted[:, 0] * (1 - fraction[:, None]) + fitted[:, 1] * fraction[:, None]
        if not np.isfinite(edges).all() or (edges[:, 1] <= edges[:, 0]).any():
            raise InputError('Local RGB edge curves invert an observed width')
        seeds = seed.reshape(len(rows), 2, 7, 2)
        original = seeds[:, 0, 3] * (1 - fraction[:, None]) + seeds[:, 1, 3] * fraction[:, None]
        refined = result['refined'].reshape(len(rows), 2, 7, 2)
        gradients = result['gradient_strength'].reshape(len(rows), 2, 7, 2)
        reasons = result['fallback_reason'].reshape(len(rows), 2, 7, 2)
        return {'positions': edges, 'original_positions': original,
                'refined': np.all(refined, axis=(1, 2)), 'gradient_strength': np.min(gradients, axis=(1, 2)),
                'fallback_reason': np.max(reasons, axis=(1, 2)), 'curve_neighbor_offsets_actual_px': np.arange(-3, 4),
                'curve_target_integer_rows': targets, 'curve_neighbor_positions': neighbor_positions,
                'curve_neighbor_original_positions': seeds, 'curve_fitted_integer_row_positions': fitted,
                'curve_neighbor_refined': refined, 'curve_neighbor_gradient_strength': gradients,
                'curve_neighbor_fallback_reason': reasons}
