"""Opt-in synthetic renderer preserving continuous vertices and pixel coverage."""

import numpy as np

from .io import InputError


def paint_person(background, projected, faces, colors, depth_order, *, scale=8):
    import cv2

    background = np.asarray(background)
    projected = np.asarray(projected, float)
    faces, colors, depth_order = map(np.asarray, (faces, colors, depth_order))
    if (
        background.ndim != 3
        or background.shape[2] != 3
        or background.dtype != np.uint8
        or min(background.shape[:2]) < 8
        or projected.ndim != 2
        or projected.shape[1] != 2
        or not np.isfinite(projected).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or (faces < 0).any()
        or (faces >= len(projected)).any()
        or colors.shape != (len(faces), 3)
        or not np.isfinite(colors).all()
        or (colors < 0).any()
        or (colors > 255).any()
        or depth_order.shape != (len(faces),)
        or not np.issubdtype(depth_order.dtype, np.integer)
        or not np.array_equal(np.sort(depth_order), np.arange(len(faces)))
        or scale not in (4, 8, 16)
    ):
        raise InputError("Invalid continuous-coverage render request")

    # Low-resolution pixel centers are integers. High-resolution sample centers
    # span each pixel's [-.5,.5) footprint. Fixed point preserves subpixel vertices.
    points = (projected + 0.5) * scale - 0.5
    fixed = np.rint(points[faces] * 256)
    if abs(fixed).max() >= 2**30:
        raise InputError("Projected coordinates exceed fixed-point render bounds")
    polygons = fixed.astype(np.int32)
    high = cv2.resize(
        background, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    mask = np.zeros(high.shape[:2], np.uint8)
    for face in depth_order:
        cv2.fillConvexPoly(high, polygons[face], tuple(map(int, colors[face])), shift=8)
        cv2.fillConvexPoly(mask, polygons[face], 255, shift=8)
    size = (background.shape[1], background.shape[0])
    return (
        cv2.resize(high, size, interpolation=cv2.INTER_AREA),
        cv2.resize(mask, size, interpolation=cv2.INTER_AREA),
    )
