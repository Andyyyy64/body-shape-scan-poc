"""Research-only axial change fields built from fitted estimated anatomy."""

import numpy as np

from .io import InputError
from .local_shape import torso_basis, transport_basis


def spline_basis(vertices, anchors, mode="spline"):
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
    if mode not in ("independent", "radial", "spline"):
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
    if mode == "spline":
        # Five overlapping anatomical height modes; no renderer targets.
        centers = np.linspace(0.08, 0.56, 5)
        q = abs((y[:, None] - centers) / 0.12)
        heights = np.where(
            q < 1, (4 - 6 * q**2 + 3 * q**3) / 6, np.where(q < 2, (2 - q) ** 3 / 6, 0)
        )
        lateral = np.clip((1.25 - abs(x) / (width / 2)) / 0.25, 0, 1)
        wx = lateral[:, None] * heights * x[:, None]
        wz = lateral[:, None] * heights * z[:, None]
        return np.concatenate(
            [
                wx[:, None, :] * right[None, :, None],
                wz[:, None, :] * forward[None, :, None],
            ],
            -1,
        ).astype("float32")
    t = (y - 0.32) / 0.28
    height_weight = np.maximum(1 - t**2, 0) ** 2 * (abs(t) < 1)
    lateral_weight = np.clip((1.25 - abs(x) / (width / 2)) / 0.25, 0, 1)
    weight = height_weight * lateral_weight
    bx = (weight * x)[:, None] * right
    bz = (weight * z)[:, None] * forward
    basis = np.stack([bx, bz], -1) if mode == "independent" else (bx + bz)[..., None]
    return basis.astype("float32")


def reference_change(data, fit):
    beta = np.asarray(fit["shape_delta"])
    global_beta = beta[:45]
    if beta.shape != (47,) or "rest_anchor_basis" not in data:
        raise InputError("Axial change requires fitted reference anatomy")
    canonical_local = torso_basis(data["rest_base"], data["rest_anchors"])
    vertices = (
        data["rest_base"]
        + data["rest_basis"] @ global_beta
        + canonical_local @ beta[-2:]
    )
    anchors = data["rest_anchors"] + data["rest_anchor_basis"] @ global_beta
    canonical = spline_basis(vertices, anchors)
    template = transport_basis(data["rest_basis"], data["template_basis"], canonical)
    fields = []
    for i, order in enumerate(data["shared_source_order"]):
        rotation = data["frame_rotations"][order]
        posed = (
            np.einsum("ij,njd->nid", rotation, template)
            * np.array([1, -1, -1])[None, :, None]
        )
        edge = data["shared_edge_vertex_indices"][i]
        weights = data["shared_edge_weights"][i]
        fields.append(np.einsum("ne,necd->ncd", weights, posed[edge]))
    return canonical, np.asarray(fields)
