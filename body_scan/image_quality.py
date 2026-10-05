"""Observed silhouette sharpness, with an explicit noise floor; no anatomy truth."""

from __future__ import annotations

from .io import InputError


def boundary_detail(image, mask):
    import cv2
    import numpy as np

    image, mask = np.asarray(image), np.asarray(mask, dtype=np.uint8)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[:2] != mask.shape
        or image.shape[2] != 3
        or not mask.any()
    ):
        raise InputError("Invalid boundary-detail observations")
    yy, xx = np.where(mask)
    y0, y1 = max(0, yy.min() - 3), min(mask.shape[0], yy.max() + 4)
    x0, x1 = max(0, xx.min() - 3), min(mask.shape[1], xx.max() + 4)
    gray = (
        cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    )
    height, width = gray.shape
    size = (max(16, round(width * 256 / height)), 256)
    gray = cv2.resize(
        gray, size, interpolation=cv2.INTER_AREA if height > 256 else cv2.INTER_LINEAR
    )
    foreground = cv2.resize(
        (mask[y0:y1, x0:x1] != 0).astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    )
    kernel = np.ones((5, 5), np.uint8)
    ring = (cv2.dilate(foreground, kernel) - cv2.erode(foreground, kernel)) > 0
    response = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0.5), cv2.CV_32F, ksize=3)
    nonedge = response[~ring]
    noise = (
        float(np.median(np.abs(nonedge - np.median(nonedge))) / 0.67448975) ** 2
        if nonedge.size
        else 0.0
    )
    energy = np.sort(response[ring] ** 2)
    trimmed = energy[: max(1, round(len(energy) * 0.9))]
    contrast = float(gray.var())
    score = (
        max(0, float(trimmed.mean()) - noise) / max(contrast, 1e-6)
        if len(energy) >= 30
        else 0.0
    )
    return {
        "boundary_detail": score,
        "noise_response_variance": noise,
        "contrast_variance": contrast,
        "boundary_pixels": int(ring.sum()),
        "reference_height_px": 256,
        "metric": "observed_boundary_noise_corrected_laplacian.v1",
    }
