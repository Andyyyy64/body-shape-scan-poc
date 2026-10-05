"""Independent 2D anatomy observations, associated to the observed person."""

from __future__ import annotations

import numpy as np

from .io import InputError
from .quality import FrameRejected


def select_anchors(keypoints, boxes, mask, projected_reference, threshold=0.5):
    points, boxes, mask, reference = [
        np.asarray(v) for v in (keypoints, boxes, mask, projected_reference)
    ]
    if (
        points.ndim != 3
        or points.shape[1:] != (17, 3)
        or boxes.shape != (len(points), 4)
        or mask.ndim != 2
        or reference.shape != (4, 2)
        or not all(np.isfinite(v).all() for v in (points, boxes, reference))
        or ((points[:, :, 2] < 0) | (points[:, :, 2] > 1)).any()
    ):
        raise InputError("Invalid independent anatomy observations")
    if not len(points) or not mask.any():
        raise FrameRejected("independent_anatomy_unobserved")
    yy, xx = np.where(mask)
    low = np.array([xx.min(), yy.min()])
    high = np.array([xx.max() + 1, yy.max() + 1])
    intersection = np.maximum(
        np.minimum(boxes[:, 2:], high) - np.maximum(boxes[:, :2], low), 0
    ).prod(1)
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    iou = intersection / np.maximum(areas + (high - low).prod() - intersection, 1)
    index = int(iou.argmax())
    if iou[index] < 0.5:
        raise FrameRejected("independent_anatomy_unobserved")
    anchors = points[index, [5, 6, 11, 12]].copy()
    # Match the two shoulder/hip labels to the estimated model projection.
    # Association changes labels only; it never changes observed coordinates.
    for start in (0, 2):
        pair = anchors[start : start + 2]
        expected = reference[start : start + 2]
        if np.sum((pair[:, :2] - expected) ** 2) > np.sum(
            (pair[::-1, :2] - expected) ** 2
        ):
            anchors[start : start + 2] = pair[::-1]
    valid = (
        (anchors[:, 2] >= threshold)
        & (anchors[:, 0] >= 0)
        & (anchors[:, 0] < mask.shape[1])
        & (anchors[:, 1] >= 0)
        & (anchors[:, 1] < mask.shape[0])
    )
    if valid.sum() < 2:
        raise FrameRejected("independent_anatomy_unobserved")
    return anchors[:, :2].astype("float32"), np.where(valid, anchors[:, 2], 0).astype(
        "float32"
    )
