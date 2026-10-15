"""Research-only stable upper silhouette width for relative camera scale.

Observed image silhouettes define every scanline. No body/camera ground truth.
The head-region assumption is explicit; this is not real-person validation.
"""

import numpy as np

from body_scan.io import InputError
from body_scan.phase_registration import periodic_reference


def scan_head(polygon, mask_height):
    polygon = np.asarray(polygon, float)
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
        or not np.isfinite(mask_height)
        or mask_height <= 10
    ):
        raise InputError("Invalid observed head contour")
    rows = polygon[:, 1].min() + np.linspace(0.02, 0.14, 64) * mask_height
    end = np.roll(polygon, -1, axis=0)
    position = []
    counts = []
    for y in rows:
        crossed = ((polygon[:, 1] <= y) & (y < end[:, 1])) | (
            (end[:, 1] <= y) & (y < polygon[:, 1])
        )
        dy = end[crossed, 1] - polygon[crossed, 1]
        x = (
            polygon[crossed, 0]
            + (y - polygon[crossed, 1]) * (end[crossed, 0] - polygon[crossed, 0]) / dy
        )
        x = np.sort(x)
        if len(x) < 2 or len(x) % 2:
            raise InputError(
                "Head-region scanline has no observable closed foreground interval"
            )
        intervals = np.stack([x[::2], x[1::2]], axis=1)
        index = int(np.argmax(intervals[:, 1] - intervals[:, 0]))
        if intervals[index, 1] - intervals[index, 0] <= 1:
            raise InputError("Observed head-region interval is too narrow")
        position.append(intervals[index])
        counts.append(len(intervals))
    return rows, np.array(position), np.array(counts)


def representative_width(positions):
    positions = np.asarray(positions, float)
    if (
        positions.ndim != 3
        or positions.shape[1:] != (64, 2)
        or not np.isfinite(positions).all()
    ):
        raise InputError("Incomplete64-row head observation")
    widths = positions[..., 1] - positions[..., 0]
    if (widths <= 0).any():
        raise InputError("Invalid head width")
    return np.median(np.sort(widths, axis=1)[:, -8:], axis=1)


def scale_from_reference(
    reference_heading, reference_width_model, query_heading, query_width_model
):
    reference_width_model, query_width_model = map(
        lambda x: np.asarray(x, float), [reference_width_model, query_width_model]
    )
    if (
        reference_width_model.shape != (48,)
        or query_width_model.shape != (48,)
        or not np.isfinite(reference_width_model).all()
        or not np.isfinite(query_width_model).all()
        or (reference_width_model <= 0).any()
        or (query_width_model <= 0).any()
    ):
        raise InputError("Head scale requires all48 positive widths")
    expected = periodic_reference(
        reference_heading,
        reference_width_model[:, None],
        query_heading,
        interpolation="linear",
    )[:, 0]
    ratios = expected / query_width_model
    return {
        "median_depth_scale": float(np.median(ratios)),
        "per_view_depth_scale": ratios,
        "relative_head_residual_rms": float(
            np.sqrt(np.mean((query_width_model / expected - 1) ** 2))
        ),
        "all48_head_observations_retained": True,
        "independent_image_width": True,
    }
