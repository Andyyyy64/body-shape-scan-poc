"""Per-frame silhouette observations for the shared-shape contour fit.

For one frame this projects the initial posed mesh, extracts its outer
boundary, keeps the mesh vertices that form that boundary, and pairs them
with the observed person-mask boundary. The vertex set and per-frame pose
are then held fixed by the optimizer (a local approximation).
"""

from __future__ import annotations

import numpy as np

from .io import InputError

CONTOUR_SAMPLES = 256
MIN_CONTOUR_POINTS = 30
LINEARITY_TOLERANCE = 1e-4
BORDER_PX = 2
NEAR_PLANE = 0.1


def _outer_contour(binary, width: int, height: int):
    import cv2

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise InputError("No silhouette boundary")
    points = max(contours, key=len)[:, 0, :]
    points = points[
        (points[:, 0] > BORDER_PX)
        & (points[:, 0] < width - 1 - BORDER_PX)
        & (points[:, 1] > BORDER_PX)
        & (points[:, 1] < height - 1 - BORDER_PX)
    ]
    if len(points) < MIN_CONTOUR_POINTS:
        raise InputError("Insufficient untruncated silhouette boundary")
    return points


def _nearest_vertex(points, projected):
    ids = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 256):
        block = points[start : start + 256].astype(np.float64)
        d = ((block[:, None, :] - projected[None, :, :]) ** 2).sum(-1)
        ids[start : start + 256] = d.argmin(1)
    return ids


def contour_observation(
    mask, faces, generate, initial_shape, camera, focal: float
) -> dict:
    """Build one frame's fixed-correspondence contour observation.

    ``mask`` is the observed person mask (H×W, nonzero = person); ``generate``
    maps a shape vector to posed vertices [N,3] in the frame's camera
    convention (x right, y down, z forward, model metres).
    """
    import cv2

    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise InputError("Mask must be a single-channel image")
    height, width = mask.shape
    initial = np.asarray(initial_shape, dtype=np.float32)
    camera = np.asarray(camera, dtype=np.float32).reshape(3)
    posed = np.asarray(generate(initial), dtype=np.float32)
    position = posed + camera
    if (position[:, 2] <= NEAR_PLANE).any():
        raise InputError("Invalid initial camera geometry")
    projected = position[:, :2] / position[:, 2:] * focal + [width / 2, height / 2]
    raster = np.zeros((height, width), np.uint8)
    for face in faces:
        cv2.fillConvexPoly(
            raster, np.clip(projected[face], -10000, 10000).astype("int32"), 1
        )
    predicted = _outer_contour(raster, width, height)
    observed = _outer_contour((mask != 0).astype("uint8"), width, height)
    ids = np.unique(_nearest_vertex(predicted, projected))
    ids = ids[np.linspace(0, len(ids) - 1, CONTOUR_SAMPLES).astype(int)]
    observed = observed[np.linspace(0, len(observed) - 1, CONTOUR_SAMPLES).astype(int)]
    base = posed[ids]
    basis = []
    for dim in range(len(initial)):
        changed = initial.copy()
        changed[dim] += 1
        basis.append(np.asarray(generate(changed), dtype=np.float32)[ids] - base)
    basis = np.stack(basis, axis=-1)
    perturbation = np.linspace(-0.1, 0.1, len(initial)).astype("float32")
    residual = np.max(
        np.abs(
            np.asarray(generate(initial + perturbation), dtype=np.float32)[ids]
            - (base + basis @ perturbation)
        )
    )
    if residual > LINEARITY_TOLERANCE:
        raise InputError("MHR local shape basis failed the linearity check")
    return {
        "base": base,
        "basis": basis,
        "target": observed.astype("float32"),
        "camera": camera,
        "focal": float(focal),
        "center": np.array([width / 2, height / 2], np.float32),
    }
