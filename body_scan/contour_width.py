"""Research-only central connected silhouette width at anatomy-derived rows."""

from .io import InputError


class WidthObservationError(InputError):
    """A requested scanline has no central connected person interval."""


def central_widths(polygon, rows, centers):
    """Widths and piecewise gradients of the interval containing each center.

    Half-open crossings avoid double counting a polygon vertex. Arms in separate
    intervals are excluded; rows whose center is outside the person are unknown,
    never silently replaced with the outermost silhouette width.
    """
    import numpy as np
    import torch

    if (
        polygon.ndim != 3
        or polygon.shape[-1] != 2
        or rows.ndim != 2
        or centers.shape != rows.shape
        or rows.shape[0] != polygon.shape[0]
        or polygon.shape[1] < 3
        or rows.shape[1] < 1
        or polygon.shape[0] * polygon.shape[1] * rows.shape[1] > 3_000_000
        or not all(torch.isfinite(x).all() for x in (polygon, rows, centers))
    ):
        raise InputError("Invalid central-width observations")
    vertices = polygon.detach().cpu().numpy().astype(np.float64)
    end = np.roll(vertices, -1, axis=1)
    y = rows.detach().cpu().numpy().astype(np.float64)[..., None]
    c = centers.detach().cpu().numpy().astype(np.float64)[..., None]
    start_y, end_y = vertices[:, None, :, 1], end[:, None, :, 1]
    crossing = ((start_y <= y) & (y < end_y)) | ((end_y <= y) & (y < start_y))
    dy = end_y - start_y
    fraction = (y - start_y) / np.where(abs(dy) > 1e-12, dy, 1)
    x = vertices[:, None, :, 0] + fraction * (end - vertices)[:, None, :, 0]
    left = np.where(crossing & (x <= c), x, -np.inf)
    right = np.where(crossing & (x > c), x, np.inf)
    # Even-odd fill and a center strictly within one connected interval.
    inside = ((crossing & (x <= c)).sum(-1) % 2) == 1
    valid = inside & np.isfinite(left.max(-1)) & np.isfinite(right.min(-1))
    if not valid.all():
        raise WidthObservationError(
            "Central silhouette width is unobservable at requested rows"
        )
    indices = [left.argmax(-1), right.argmin(-1)]
    intersections = []
    for index in indices:
        selected = torch.as_tensor(index, dtype=torch.int64, device=polygon.device)
        a = polygon.gather(1, selected[..., None].expand(-1, -1, 2))
        b = polygon.roll(-1, dims=1).gather(1, selected[..., None].expand(-1, -1, 2))
        fraction = (rows - a[..., 1]) / (b[..., 1] - a[..., 1])
        intersections.append(a[..., 0] + fraction * (b[..., 0] - a[..., 0]))
    return intersections[1] - intersections[0]


def anatomy_rows(projected_anchors, count=41):
    """Fixed image rows spanning estimated hip-to-shoulder fractions .1.. .6."""
    import torch

    if (
        projected_anchors.ndim != 3
        or projected_anchors.shape[1:] != (4, 2)
        or not torch.isfinite(projected_anchors).all()
        or count != 41
    ):
        raise InputError("Width bands require four estimated shoulder/hip anchors")
    shoulder, hip = projected_anchors[:, :2].mean(1), projected_anchors[:, 2:].mean(1)
    fraction = torch.linspace(0.1, 0.6, count, dtype=hip.dtype, device=hip.device)
    points = hip[:, None] + fraction[None, :, None] * (shoulder - hip)[:, None]
    if ((shoulder[:, 1] - hip[:, 1]).abs() < 5).any():
        raise InputError("Projected torso height cannot support a width band")
    return points[..., 1].detach(), points[..., 0].detach()
