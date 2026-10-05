"""Research-only longitudinal width calibration without measurement labels."""

import numpy as np

from .io import InputError
from .phase_registration import native_headings, periodic_reference


def project(vertices, camera, focal, center):
    import torch

    position = np.asarray(vertices, float) + np.asarray(camera, float)[:, None]
    if not np.isfinite(position).all() or (position[..., 2] <= 0.1).any():
        raise InputError("Calibrated width projection is not observable")
    uv = position[..., :2] / position[..., 2:] * np.asarray(focal)[:, None, None]
    return torch.as_tensor(uv + np.asarray(center)[:, None], dtype=torch.float64)


def observe(data, reference_initial, *, normalizer=None, row_mode="native_camera"):
    """Native pose and actual segmented contour are the only observations."""
    import torch

    from .contour_width import anatomy_rows, central_widths, pixel_row_widths

    shift = np.asarray(reference_initial) - data["initial"]
    anchors = data["shared_anchor_base"] + data["shared_anchor_basis"] @ shift
    camera = np.broadcast_to(
        torch.as_tensor(data["shared_camera"]).median(0).values.numpy(),
        data["shared_camera"].shape,
    )
    if row_mode == "fitted_camera":
        camera = data["normalization_camera"]
        if "depth_global_coefficients" in data:
            anchors = (
                anchors
                + data["shared_anchor_basis"] @ data["depth_global_coefficients"]
            )
    if row_mode == "image_anchors":
        projected = torch.as_tensor(data["shared_anchor_target"], dtype=torch.float64)
    elif row_mode in ("native_camera", "fitted_camera"):
        projected = project(
            anchors, camera, data["shared_focal"], data["shared_center"]
        )
    else:
        raise InputError("Invalid width row correspondence")
    rows, centers = anatomy_rows(projected)
    observer = data.get("width_observer", "point")
    if observer not in ("point", "pixel_row_bilinear"):
        raise InputError("Unknown calibrated width observer")
    width_function = pixel_row_widths if observer == "pixel_row_bilinear" else central_widths
    widths = width_function(
        torch.as_tensor(
            data.get("width_target", data["shared_target"]), dtype=torch.float64
        ),
        rows,
        centers,
    ).numpy()
    if "observed_width_px" in data:
        measured = np.asarray(data["observed_width_px"], float)
        if (
            measured.shape != widths.shape
            or not np.isfinite(measured).all()
            or (measured <= 0).any()
        ):
            raise InputError("Invalid cached RGB width observation")
        if not np.allclose(
            rows.numpy(), data["width_observation_rows"], atol=1e-8, rtol=0
        ) or not np.allclose(
            centers.numpy(), data["width_observation_centers"], atol=1e-8, rtol=0
        ):
            raise InputError(
                "Cached RGB measurement rows differ from estimated geometry"
            )
        widths = measured

    height = np.asarray(data["shared_mask_height"], float)
    if (height <= 0).any():
        raise InputError("Person height is unobservable")
    heading = native_headings(data["frame_rotations"], data["root_tilt_reference"])
    order = np.asarray(data["shared_source_order"])
    if order.shape != height.shape or not np.issubdtype(order.dtype, np.integer):
        raise InputError("Invalid observed pose ordering")
    heading = heading[order]
    periodic_reference(heading, np.ones(len(heading)), heading)
    scale = height[:, None] / 256 if normalizer is None else np.asarray(normalizer)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise InputError("Width projection scale is unobservable")
    return heading, widths / scale, rows, centers


def estimated_depth(data, camera):
    anchors = data["shared_anchor_base"]
    if "depth_global_coefficients" in data:
        anchors = (
            anchors + data["shared_anchor_basis"] @ data["depth_global_coefficients"]
        )
    hip, shoulder = anchors[:, 2:].mean(1), anchors[:, :2].mean(1)
    point = hip + 0.32 * (shoulder - hip)
    depth = np.asarray(camera)[:, 2] + point[:, 2]
    if (depth <= 0.1).any() or not np.isfinite(depth).all():
        raise InputError("Estimated torso depth is unobservable")
    return depth


def calibrate(
    data,
    fit,
    *,
    normalization="mask_height",
    row_mode="native_camera",
    noise_mode="uniform",
):
    """Freeze an observed reference and estimated local response Jacobian."""
    from .contour_width import anatomy_rows, central_widths, pixel_row_widths

    camera = data["shared_camera"] + np.asarray(fit["camera_delta"])
    data = {**data, "normalization_camera": camera}
    heading, observed, rows, centers = observe(data, data["initial"], row_mode=row_mode)
    beta = np.asarray(fit["shape_delta"], float)
    if beta.shape != (47,) or not np.isfinite(beta).all():
        raise InputError("Calibration requires two estimated canonical local modes")
    data = {**data, "depth_global_coefficients": beta[:45]}
    camera = data["shared_camera"] + np.asarray(fit["camera_delta"])
    base = data["shared_base"] + data["shared_basis"] @ beta[:45]
    change_basis = data.get("shared_change_basis")
    if change_basis is not None:
        change_basis = np.asarray(change_basis)
        if change_basis.shape[:3] != base.shape or not np.isfinite(change_basis).all():
            raise InputError("Invalid estimated axial change transport")
        base = base + data["shared_canonical_local_independent"] @ beta[-2:]
        change_reference = np.zeros(change_basis.shape[-1])
    else:
        change_basis = data["shared_canonical_local_independent"]
        change_reference = beta[-2:]
    anchors = data["shared_anchor_base"] + data["shared_anchor_basis"] @ beta[:45]
    _, predicted_centers = anatomy_rows(
        project(anchors, camera, data["shared_focal"], data["shared_center"])
    )
    normalizer = data["shared_mask_height"][:, None] / 256
    reference_height = float(
        np.median(
            data["shared_mask_height"]
            * estimated_depth(data, camera)
            / data["shared_focal"]
        )
    )
    if normalization == "depth":
        normalizer = (
            data["shared_focal"]
            / estimated_depth(data, camera)
            * reference_height
            / 256
        )[:, None]
        heading, observed, rows, centers = observe(
            data, data["initial"], normalizer=normalizer, row_mode=row_mode
        )
    elif normalization not in ("mask_height", "model_width"):
        raise InputError("Invalid reference projection normalization")

    def widths(coefficient):
        projection = project(
            base + change_basis @ coefficient,
            camera,
            data["shared_focal"],
            data["shared_center"],
        )
        width_function = pixel_row_widths if data.get("width_observer") == "pixel_row_bilinear" else central_widths
        return width_function(projection, rows, predicted_centers).numpy() / normalizer

    count = len(change_reference)
    jacobian = np.stack(
        [
            (
                widths(change_reference + np.eye(count)[i] * 0.001)
                - widths(change_reference - np.eye(count)[i] * 0.001)
            )
            / 0.002
            for i in range(count)
        ],
        axis=-1,
    )
    result = {
        "heading": heading,
        "width": observed,
        "jacobian": jacobian,
        "baseline_coefficients": beta[-2:].copy(),
        "global_coefficients": beta[:45].copy(),
        "model_width": widths(change_reference),
        "change_mode": "axial" if "shared_change_basis" in data else "independent",
        "normalization": normalization,
        "row_mode": row_mode,
        "reference_height_model_m": reference_height,
        "initial": data["initial"].copy(),
        "row_count": 41,
        "source": "estimated native XYZ pose and segmented images",
        "width_observer": data.get("width_observer", "point"),
        "physical_accuracy_validated": False,
    }
    if noise_mode == "reference_cv":
        errors = []
        for i in range(len(heading)):
            keep = np.arange(len(heading)) != i
            predicted = periodic_reference(
                heading[keep], observed[keep], heading[i : i + 1], interpolation="cubic"
            )
            errors.append(observed[i] - predicted[0])
        pixel_floor = float(np.median(1 / normalizer))
        result["noise_sigma"] = np.maximum(
            np.sqrt(2 * np.mean(np.square(errors), axis=0)), pixel_floor
        )
        result["noise_mode"] = noise_mode
        result["noise_pixel_floor"] = pixel_floor
    elif noise_mode != "uniform":
        raise InputError("Invalid calibrated width uncertainty")
    return result


def infer(
    calibration,
    data,
    *,
    softening=1.0,
    angular_quadrature=True,
    interpolation="linear",
    refine_phase=False,
    observations=None,
):
    if "global_coefficients" in calibration:
        data = {**data, "depth_global_coefficients": calibration["global_coefficients"]}
    normalization = calibration.get("normalization", "mask_height")
    camera = data.get("normalization_camera", data["shared_camera"])
    normalizer = None
    if normalization == "depth":
        normalizer = (
            data["shared_focal"]
            / estimated_depth(data, camera)
            * calibration["reference_height_model_m"]
            / 256
        )[:, None]
    heading, observed, rows, _ = observe(
        data,
        calibration["initial"],
        normalizer=normalizer,
        row_mode=calibration.get("row_mode", "native_camera"),
    )
    if observations is not None:
        if (
            len(observations) != 4
            or not np.array_equal(observations[0], heading)
            or not np.array_equal(np.asarray(observations[2]), np.asarray(rows))
        ):
            raise InputError("Local corrected observation identities differ")
        corrected = np.asarray(observations[1], float)
        if corrected.shape != observed.shape or not np.isfinite(corrected).all():
            raise InputError("Invalid local corrected widths")
        observed = corrected

    def interpolated(values, phase):
        return periodic_reference(
            calibration["heading"], values, phase, interpolation=interpolation
        )

    reference = interpolated(calibration["width"], heading)
    jacobian = interpolated(calibration["jacobian"], heading)
    count = jacobian.shape[-1]
    matrix, signal = jacobian.reshape(-1, count), (observed - reference).reshape(-1)
    if normalization == "model_width":
        if calibration.get("change_mode", "independent") != "independent":
            raise InputError("Axial calibration requires depth normalization")
        from .contour_width import anatomy_rows, central_widths

        shift = calibration["initial"] - data["initial"]
        base = data["shared_base"] + data["shared_basis"] @ (
            shift + calibration["global_coefficients"]
        )
        anchors = data["shared_anchor_base"] + data["shared_anchor_basis"] @ shift
        _, centers = anatomy_rows(
            project(anchors, camera, data["shared_focal"], data["shared_center"])
        )
        local = data["shared_canonical_local_independent"]
        beta = calibration["baseline_coefficients"]

        def projected_width(coefficients):
            return central_widths(
                project(
                    base + local @ coefficients,
                    camera,
                    data["shared_focal"],
                    data["shared_center"],
                ),
                rows,
                centers,
            ).numpy()

        expected = projected_width(beta)
        if (expected <= 1).any():
            raise InputError("Projected width cannot identify a local scale")
        raw_observed = observed * (data["shared_mask_height"][:, None] / 256)
        reference_ratio = calibration["width"] / calibration["model_width"]
        reference = interpolated(reference_ratio, heading)
        scale = interpolated(calibration["model_width"], heading)
        signal = ((raw_observed / expected - reference) * scale).reshape(-1)
        query_jacobian = np.stack(
            [
                (
                    projected_width(beta + np.eye(2)[i] * 0.001)
                    - projected_width(beta - np.eye(2)[i] * 0.001)
                )
                / 0.002
                for i in range(2)
            ],
            axis=-1,
        )
        matrix = (query_jacobian / expected[..., None] * scale[..., None]).reshape(
            -1, 2
        )
    if refine_phase:
        values = (
            reference_ratio if normalization == "model_width" else calibration["width"]
        )
        derivative = (
            interpolated(values, heading + 0.001)
            - interpolated(values, heading - 0.001)
        ) / 0.002
        if normalization == "model_width":
            derivative = derivative * scale
        matrix = np.c_[matrix, derivative.reshape(-1)]
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] <= 1e-8 or singular[-1] < singular[0] * 1e-4:
        raise InputError("Two local shape changes are not independently observable")
    angular = np.ones(len(heading))
    if angular_quadrature:
        phase = np.mod(heading, 2 * np.pi)
        order = np.argsort(phase)
        sorted_phase = phase[order]
        gaps = np.diff(np.r_[sorted_phase, sorted_phase[0] + 2 * np.pi])
        angular[order] = (gaps + np.roll(gaps, 1)) / 2 * len(heading) / (2 * np.pi)
    angular = np.repeat(angular, observed.shape[1])
    if "noise_sigma" in calibration:
        sigma = np.asarray(calibration["noise_sigma"])
        if (
            sigma.shape != (observed.shape[1],)
            or not np.isfinite(sigma).all()
            or (sigma <= 0).any()
        ):
            raise InputError("Invalid reference width noise")
        angular = angular / np.tile(sigma**2, len(heading))
    weight = angular.copy()
    coefficient = np.zeros(matrix.shape[1])
    for _ in range(8):
        coefficient = np.linalg.lstsq(
            matrix * np.sqrt(weight[:, None]), signal * np.sqrt(weight), rcond=1e-9
        )[0]
        error = matrix @ coefficient - signal
        weight = angular / np.sqrt(1 + (error / softening) ** 2)
    phase = coefficient[count] if refine_phase else 0.0
    coefficient = coefficient[:count]
    baseline_limit = (
        calibration.get("change_mode", "independent") == "independent"
        and abs(coefficient + calibration["baseline_coefficients"]).max() > 0.25
    )
    if abs(phase) > np.deg2rad(10) or abs(coefficient).max() > 0.1 or baseline_limit:
        raise InputError("Longitudinal residual exceeds its local domain")
    return {
        "delta_coefficients": coefficient,
        "singular_values": singular,
        "width_residual_rms_reference_px": float(np.sqrt(np.mean(error**2))),
        "reference_width_change_mean_reference_px": float(np.mean(signal)),
        "registration": "periodic native estimated pose, independent of frame labels/order",
        "registration_interpolation": interpolation,
        "phase_refinement_deg": float(np.rad2deg(phase)),
        "projection_normalization": normalization,
        "row_correspondence": calibration.get("row_mode", "native_camera"),
        "frame_count": len(heading),
        "row_count": observed.shape[1],
        "physical_accuracy_validated": False,
    }
