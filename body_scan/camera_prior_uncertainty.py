"""Conservative heuristic for a clip's correlated image head-scale observations.

Adjacent angular views are not assumed independent. This is not a calibrated
physical confidence interval; the fixed precision gates still decide adoption.
"""

import numpy as np

from body_scan.io import InputError


def head_camera_prior_sigma(scales, depth):
    scales = np.asarray(scales, float)
    if (
        scales.shape != (48,)
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
        or not np.isfinite(depth)
        or depth <= 0.1
    ):
        raise InputError("Invalid head-scale camera prior observations")
    spread = 1.4826 * float(np.median(abs(scales - np.median(scales))))
    error = scales - scales.mean()
    variance = float(np.mean(error**2))
    correlations = []
    inflation = 1.0
    for lag in range(1, 13):
        correlation = (
            0.0
            if variance <= 1e-24
            else float(np.mean(error * np.roll(error, lag)) / variance)
        )
        positive = max(0.0, min(1.0, correlation))
        correlations.append(positive)
        inflation += 2 * (1 - lag / 13) * positive
    effective_views = 48 / inflation
    return {
        "sigma_model_m": max(0.0025 * depth, spread * depth / np.sqrt(effective_views)),
        "effective_head_views": effective_views,
        "head_scale_mad": spread,
        "positive_circular_lag_correlations": correlations,
        "independent48_sigma_model_m": max(
            0.0025 * depth, spread * depth / np.sqrt(48)
        ),
        "calibrated_physical_confidence": False,
        "all48_views_retained": True,
    }
