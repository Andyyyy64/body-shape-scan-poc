"""Bounded RGB refinement of an observed person mask; no model geometry inputs."""

from __future__ import annotations

import numpy as np

from .io import InputError
from .quality import FrameRejected


def refine_person_mask(image, mask, radius=7, iterations=3):
    import cv2

    image, mask = np.asarray(image), np.asarray(mask)
    if (
        image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or mask.shape != image.shape[:2]
        or max(mask.shape) > 4096
        or not 1 <= radius <= 15
        or not 1 <= iterations <= 5
    ):
        raise InputError("Invalid image or mask refinement budget")
    mask = (mask != 0).astype("uint8")
    if not mask.any() or mask.all():
        raise FrameRejected("insufficient_mask_refinement_seeds")
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    core = cv2.erode(mask, kernel)
    envelope = cv2.dilate(mask, kernel)
    if core.sum() < 32 or (envelope == 0).sum() < 32:
        raise FrameRejected("insufficient_mask_refinement_seeds")
    labels = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    labels[envelope != 0] = cv2.GC_PR_BGD
    labels[mask != 0] = cv2.GC_PR_FGD
    labels[core != 0] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(
        np.ascontiguousarray(image),
        labels,
        None,
        np.zeros((1, 65), np.float64),
        np.zeros((1, 65), np.float64),
        iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    refined = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    ratio = float(refined.sum() / mask.sum())
    if not 0.75 <= ratio <= 1.25:
        raise FrameRejected("excessive_mask_refinement")
    return refined, {
        "method": "observed_rgb_grabcut.v1",
        "radius_px": radius,
        "iterations": iterations,
        "area_ratio": ratio,
        "changed_pixels": int((refined != (mask != 0)).sum()),
    }
