"""Fresh RGB widths at anatomy rows corresponding to the corrected camera."""

import hashlib

import numpy as np

from body_scan.io import InputError
from body_scan.subpixel_edges import (
    central_positions,
    pixel_row_positions,
    refine_pixel_row_positions,
    refine_positions,
)


def adjusted_camera(data, calibration, offset):
    beta = (
        np.asarray(calibration["initial"])
        - data["initial"]
        + calibration["global_coefficients"]
    )
    anchors = data["shared_anchor_base"] + data["shared_anchor_basis"] @ beta
    hip, shoulder = anchors[:, 2:].mean(1), anchors[:, :2].mean(1)
    point = hip + 0.32 * (shoulder - hip)
    camera = np.asarray(data["normalization_camera"], float).copy()
    if offset == 0:
        return camera
    old_depth = point[:, 2] + camera[:, 2]
    new_depth = old_depth + offset
    if (
        not np.isfinite(offset)
        or not np.isfinite(new_depth).all()
        or (new_depth <= 0.1).any()
    ):
        raise InputError("Unobservable corrected measurement camera")
    camera[:, 2] += offset
    camera[:, :2] = (point[:, :2] + camera[:, :2]) * (new_depth / old_depth)[
        :, None
    ] - point[:, :2]
    return camera


def resample(video, data, rows, centers, expected_decoded_hashes):
    import cv2

    rows, centers = map(np.asarray, [rows, centers])
    frames = data["frames"][data["shared_source_order"].astype(int)]
    if (
        len(frames) != 48
        or len(set(frames)) != 48
        or rows.shape != (48, 41)
        or centers.shape != rows.shape
    ):
        raise InputError("Incomplete corrected row observations")
    pixel_observer = data.get("width_observer") == "pixel_row_bilinear"
    positions_function = pixel_row_positions if pixel_observer else central_positions
    original = np.array(
        [
            positions_function(poly, y, x)
            for poly, y, x in zip(data["width_target"], rows, centers)
        ]
    )
    lookup = {int(f): i for i, f in enumerate(frames)}
    refinements = [None] * 48
    decoded = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for frame in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, image = capture.read()
            if not ok:
                raise InputError("Frozen corrected row capture decode failed")
            if frame not in lookup:
                continue
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest != expected_decoded_hashes[str(frame)]:
                raise InputError(
                    "Corrected row source RGB differs from frozen observation"
                )
            decoded[str(frame)] = digest
            i = lookup[frame]
            refinements[i] = (
                refine_pixel_row_positions(image, data["width_target"][i], rows[i], centers[i])
                if pixel_observer else refine_positions(image, rows[i], original[i])
            )
    finally:
        capture.release()
    if any(r is None for r in refinements):
        raise InputError("Missing corrected row RGB observations")
    position = np.array([r["positions"] for r in refinements])
    pixel_evidence = {
        key: np.array([r[key] for r in refinements])
        for key in ["pixel_row_indices", "pixel_row_weights", "pixel_row_original_positions",
                    "pixel_row_rgb_positions", "pixel_row_refined_edges", "pixel_row_fallback_reason",
                    "pixel_row_gradient_strength"]
    } if pixel_observer else {}
    return {
        "width_observation_rows": rows,
        "width_observation_centers": centers,
        "mask_width_px": original[..., 1] - original[..., 0],
        "rgb_width_px": position[..., 1] - position[..., 0],
        "original_positions": original,
        "rgb_positions": position,
        "refined_edges": np.array([r["refined"] for r in refinements]),
        "fallback_reason": np.array([r["fallback_reason"] for r in refinements]),
        "gradient_strength": np.array([r["gradient_strength"] for r in refinements]),
        "observed_frames": frames,
        **pixel_evidence,
    }, decoded
