"""Research-only perspective transport of a fixed estimated reference surface."""

import numpy as np

from .calibrated_width import project
from .contour_width import anatomy_rows, central_widths, pixel_row_widths
from .io import InputError


def model_transport(
    vertices, anchors, camera, focal, center, target_depth, reference_height,
    *, width_observer="point",
):
    """Compare actual and virtual projections at the same anatomical band.

    The virtual camera preserves the projected torso-center point. Only model
    rows move; actual pixel observations are never resampled or replaced here.
    """
    vertices, anchors, camera, focal, center = map(
        np.asarray, [vertices, anchors, camera, focal, center]
    )
    if (
        vertices.ndim != 3
        or vertices.shape[-1] != 3
        or anchors.shape != (len(vertices), 4, 3)
        or camera.shape != (len(vertices), 3)
        or focal.shape != (len(vertices),)
        or center.shape != (len(vertices), 2)
        or not all(
            np.isfinite(v).all() for v in [vertices, anchors, camera, focal, center]
        )
        or (focal <= 0).any()
        or not np.isfinite(target_depth)
        or target_depth <= 0.1
        or not np.isfinite(reference_height)
        or reference_height <= 0
    ):
        raise InputError("Invalid perspective width transport inputs")
    hip, shoulder = anchors[:, 2:].mean(1), anchors[:, :2].mean(1)
    point = hip + 0.32 * (shoulder - hip)
    depth = camera[:, 2] + point[:, 2]
    if (depth <= 0.1).any():
        raise InputError("Perspective torso depth is unobservable")
    virtual = camera.astype(float, copy=True)
    virtual[:, 2] += target_depth - depth
    virtual[:, :2] = (camera[:, :2] + point[:, :2]) * (target_depth / depth)[
        :, None
    ] - point[:, :2]
    actual_anchor = project(anchors, camera, focal, center)
    virtual_anchor = project(anchors, virtual, focal, center)
    rows, centers = anatomy_rows(actual_anchor)
    virtual_rows, virtual_centers = anatomy_rows(virtual_anchor)
    if width_observer not in ("point", "pixel_row_bilinear"):
        raise InputError("Unknown perspective width observer")
    width_function = pixel_row_widths if width_observer == "pixel_row_bilinear" else central_widths
    actual_width = width_function(
        project(vertices, camera, focal, center), rows, centers
    ).numpy()
    virtual_width = width_function(
        project(vertices, virtual, focal, center), virtual_rows, virtual_centers
    ).numpy()
    normalizer = (focal / depth * reference_height / 256)[:, None]
    virtual_normalizer = (focal / target_depth * reference_height / 256)[:, None]
    actual_normalized, virtual_normalized = (
        actual_width / normalizer,
        virtual_width / virtual_normalizer,
    )
    actual_center = project(point[:, None], camera, focal, center).numpy()[:, 0]
    virtual_center = project(point[:, None], virtual, focal, center).numpy()[:, 0]
    if not np.allclose(actual_center, virtual_center, rtol=0, atol=1e-9):
        raise InputError("Perspective transport moved the projected torso center")
    return {
        "correction": actual_normalized - virtual_normalized,
        "actual_normalized_model_width": actual_normalized,
        "virtual_normalized_model_width": virtual_normalized,
        "actual_rows": rows.numpy(),
        "actual_centers": centers.numpy(),
        "virtual_rows": virtual_rows.numpy(),
        "virtual_centers": virtual_centers.numpy(),
        "actual_torso_depth": depth,
        "virtual_camera": virtual,
    }


def correction(data, calibration, target_depth):
    shift = np.asarray(calibration["initial"]) - data["initial"]
    beta = shift + calibration["global_coefficients"]
    vertices = data["shared_base"] + data["shared_basis"] @ beta
    vertices = (
        vertices
        + data["shared_canonical_local_independent"]
        @ calibration["baseline_coefficients"]
    )
    anchors = data["shared_anchor_base"] + data["shared_anchor_basis"] @ beta
    result = model_transport(
        vertices,
        anchors,
        data["normalization_camera"],
        data["shared_focal"],
        data["shared_center"],
        target_depth,
        calibration["reference_height_model_m"],
        width_observer=calibration.get("width_observer", "point"),
    )
    if "width_observation_rows" in data:
        if not np.allclose(
            result["actual_rows"], data["width_observation_rows"], rtol=0, atol=1e-8
        ) or not np.allclose(
            result["actual_centers"],
            data["width_observation_centers"],
            rtol=0,
            atol=1e-8,
        ):
            raise InputError(
                "Perspective model rows differ from frozen pixel observations"
            )
    return result
