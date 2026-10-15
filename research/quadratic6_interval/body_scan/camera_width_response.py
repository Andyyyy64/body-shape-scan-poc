"""Local camera response of actual silhouette rows with fixed RGB edge offsets.

This linearization proposes a camera offset. The final fit must reobserve RGB
at the proposed physical rows, never use this approximation as final pixels.
"""

import numpy as np

from body_scan.calibrated_width import estimated_depth, observe
from body_scan.depth_row_observation import adjusted_camera
from body_scan.io import InputError
from body_scan.perspective_width import correction


def width_camera_derivative(data, calibration, target_depth, step=1e-4):
    if not np.isfinite(step) or not 1e-6 <= step <= 0.01:
        raise InputError("Invalid camera response step")
    raw_rgb = np.asarray(data["observed_width_px"], float)
    mask_data = {
        k: v
        for k, v in data.items()
        if k
        not in [
            "observed_width_px",
            "width_observation_rows",
            "width_observation_centers",
        ]
    }
    _, mask_normalized, _, _ = observe(
        mask_data, calibration["initial"], row_mode="fitted_camera"
    )
    mask_raw = mask_normalized * (data["shared_mask_height"][:, None] / 256)
    rgb_offset = raw_rgb - mask_raw
    curves = []
    for offset in [-step, step]:
        trial = {
            **mask_data,
            "normalization_camera": adjusted_camera(data, calibration, offset),
        }
        _, mask_normalized, _, _ = observe(
            trial, calibration["initial"], row_mode="fitted_camera"
        )
        mask_raw = mask_normalized * (trial["shared_mask_height"][:, None] / 256)
        depth = estimated_depth(trial, trial["normalization_camera"])
        normalizer = (
            trial["shared_focal"]
            / depth
            * calibration["reference_height_model_m"]
            / 256
        )[:, None]
        curves.append(
            (mask_raw + rgb_offset) / normalizer
            - correction(trial, calibration, target_depth)["correction"]
        )
    derivative = (curves[1] - curves[0]) / (2 * step)
    if derivative.shape != raw_rgb.shape or not np.isfinite(derivative).all():
        raise InputError("Unobservable local camera width response")
    return derivative
