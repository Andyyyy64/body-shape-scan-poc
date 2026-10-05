"""Small anatomical torso residual basis, distinct from synthetic deformation."""

from __future__ import annotations

import numpy as np

from .io import InputError


def transport_basis(rest_shape_basis, posed_shape_basis, local_basis):
    """Infer each fixed-skeleton vertex's linear transport from the MHR basis.

    Shape-independent skinning maps the complete rest basis into the posed
    basis. Require rank and reconstruction checks before transporting a new
    canonical residual. Renderer geometry and poses are never accepted here.
    """
    rest, posed, local = [
        np.asarray(x, float) for x in (rest_shape_basis, posed_shape_basis, local_basis)
    ]
    if (
        rest.ndim != 3
        or rest.shape[1] != 3
        or posed.shape != rest.shape
        or local.ndim != 3
        or local.shape[:2] != rest.shape[:2]
        or not all(np.isfinite(x).all() for x in (rest, posed, local))
    ):
        raise InputError("Invalid canonical residual transport")
    active = np.linalg.norm(local, axis=(1, 2)) > 1e-12
    result = np.zeros_like(local)
    if not active.any():
        return result.astype("float32")
    selected = rest[active]
    singular = np.linalg.svd(selected, compute_uv=False)
    if (singular[:, -1] <= np.maximum(singular[:, 0] * 1e-6, 1e-9)).any():
        raise InputError("Canonical shape basis cannot identify residual transport")
    transport = posed[active] @ np.linalg.pinv(selected, rcond=1e-6)
    residual = np.max(abs(transport @ selected - posed[active]))
    if residual > 1e-4:
        raise InputError("Posed basis does not preserve canonical shape transport")
    result[active] = transport @ local[active]
    return result.astype("float32")


def torso_basis(vertices, anchors, mode="independent"):
    """Return radial fractional x/z modes in the estimated shoulder/hip frame."""
    v, anchors = np.asarray(vertices), np.asarray(anchors)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or anchors.shape != (4, 3)
        or not np.isfinite(v).all()
        or not np.isfinite(anchors).all()
    ):
        raise InputError("Invalid anatomical residual geometry")
    if mode not in ("independent", "radial"):
        raise InputError("Invalid torso residual mode")
    hip = anchors[2:].mean(0)
    up = anchors[:2].mean(0) - hip
    span = float(np.linalg.norm(up))
    if span <= 0.01:
        raise InputError("Degenerate anatomical torso span")
    up = up / span
    right = anchors[1] - anchors[0]
    right = right - up * (right @ up)
    width = float(np.linalg.norm(right))
    if width <= 0.01:
        raise InputError("Degenerate anatomical torso width")
    right = right / width
    forward = np.cross(right, up)
    offset = v - hip
    x, y, z = offset @ right, offset @ up / span, offset @ forward
    t = (y - 0.32) / 0.28
    height_weight = np.maximum(1 - t**2, 0) ** 2 * (abs(t) < 1)
    lateral_weight = np.clip((1.25 - abs(x) / (width / 2)) / 0.25, 0, 1)
    weight = height_weight * lateral_weight
    bx = (weight * x)[:, None] * right
    bz = (weight * z)[:, None] * forward
    basis = np.stack([bx, bz], -1) if mode == "independent" else (bx + bz)[..., None]
    return basis.astype("float32")
