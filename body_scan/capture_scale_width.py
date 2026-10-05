"""Experimental paired RGB measurement with shared head and calf scale cues.

The camera receives one joint local proposal, followed by fresh RGB rows and
shape/phase polishing. This is not an iterative camera optimum or human accuracy.
No case labels, renderer parameters or domain truth enter this module.
"""

from __future__ import annotations

import numpy as np

from . import calibrated_width
from .camera_prior_uncertainty import head_camera_prior_sigma
from .camera_width_response import width_camera_derivative
from .capture_width import _perimeter
from .depth_row_observation import adjusted_camera
from .framewise_camera import fit_curve_camera
from .framewise_width import fit_curve, pose_sigma
from .head_scale import scale_from_reference
from .io import InputError
from .perspective_width import correction
from .phase_registration import periodic_reference


def _cue_widths(cues, data):
    frames = data["frames"][data["shared_source_order"].astype(int)]
    if len(frames) != 48 or not np.array_equal(cues["observed_frames"], frames):
        raise InputError("Stable scale cues differ from all48 native frame identities")
    widths = []
    for key in ["rgb_head_width_px", "rgb_leg_width_px"]:
        value = np.asarray(cues[key], float)
        if value.shape != (48,) or not np.isfinite(value).all() or (value <= 0).any():
            raise InputError("All48 positive head and calf RGB widths are required")
        widths.append(value)
    return np.stack(widths)


def _variance(heading, observed, normalizer):
    errors = []
    for i in range(len(heading)):
        keep = np.arange(len(heading)) != i
        expected = periodic_reference(
            heading[keep], observed[keep], heading[i : i + 1], interpolation="cubic"
        )
        errors.append(observed[i] - expected[0])
    floor = float(np.median(1 / normalizer))
    return np.square(errors) + floor**2


def prepare_scale_reference(prepared, cues):
    prepared = {
        **prepared,
        "reference": dict(prepared["reference"]),
        "calibration": dict(prepared["calibration"]),
    }
    c = prepared["calibration"]
    data = prepared["reference"]
    data["depth_global_coefficients"] = c["global_coefficients"]
    depth = calibrated_width.estimated_depth(data, data["normalization_camera"])
    normalizer = data["shared_focal"] / depth * c["reference_height_model_m"] / 256
    raw = _cue_widths(cues, data)
    prepared["scale_reference_cues_model"] = (
        raw * depth[None, :] / data["shared_focal"][None, :]
    )
    prepared["scale_target_depth_model_m"] = float(np.median(depth))
    c["noise_variance_curve"] = _variance(c["heading"], c["width"], normalizer)
    c["noise_mode"] = (
        "reference_plus_query_angle_and_row_cv_variance_fixed_at_native_query_heading"
    )
    c["width"] = (
        c["width"]
        - correction(data, c, prepared["scale_target_depth_model_m"])["correction"]
    )
    return prepared


def measure_scale_capture(
    prepared, query, query_fit, query_pose, reference_pose, cues, *, reobserve
):
    data = dict(query)
    ref = prepared["reference"]
    c = dict(prepared["calibration"])
    if any(
        not np.array_equal(data[k], ref[k])
        for k in ["faces", "section_y", "section_center"]
    ):
        raise InputError("Query and reference measurement geometry differ")
    camera = np.asarray(query_fit["camera_delta"])
    if camera.shape != data["shared_camera"].shape or not np.isfinite(camera).all():
        raise InputError("Query camera/frame correspondence is invalid")
    data["normalization_camera"] = (
        data["shared_camera"] + camera + prepared["camera_geometry_offset"]
    )
    data["depth_global_coefficients"] = c["global_coefficients"]
    depth = calibrated_width.estimated_depth(data, data["normalization_camera"])
    normalizer = (data["shared_focal"] / depth * c["reference_height_model_m"] / 256)[
        :, None
    ]
    heading, raw_observed, rows, centers = calibrated_width.observe(
        data, c["initial"], normalizer=normalizer, row_mode="fitted_camera"
    )
    query_widths = (
        _cue_widths(cues, data) * depth[None, :] / data["shared_focal"][None, :]
    )
    cue_diagnostics = [
        scale_from_reference(c["heading"], reference, heading, observed)
        for reference, observed in zip(
            prepared["scale_reference_cues_model"], query_widths
        )
    ]
    ratios = np.sqrt(
        cue_diagnostics[0]["per_view_depth_scale"]
        * cue_diagnostics[1]["per_view_depth_scale"]
    )
    median_depth = float(np.median(depth))
    mean = (np.median(ratios) - 1) * median_depth
    uncertainty = head_camera_prior_sigma(ratios, median_depth)
    c["query_noise_variance"] = _variance(heading, raw_observed, normalizer)
    observed = (
        raw_observed
        - correction(data, c, prepared["scale_target_depth_model_m"])["correction"]
    )
    # A local observation override avoids mutable process-global monkeypatches.
    initial = calibrated_width.infer(
        c,
        data,
        interpolation="cubic",
        refine_phase=True,
        observations=(heading, observed, rows, centers),
    )
    q, r = (
        np.asarray(query_pose["covariance"]),
        np.asarray(reference_pose["covariance"]),
    )
    if q.shape != (48, 48) or r.shape != (48, 48):
        raise InputError("Scale-camera phase covariance ordering differs")
    transport = periodic_reference(
        c["heading"], np.eye(48), heading, interpolation="linear"
    )
    relative = q + transport @ r @ transport.T + 2 * np.deg2rad(0.25) ** 2 * np.eye(48)
    sigma = pose_sigma(query_pose["sigma"], reference_pose["sigma"])
    joint = fit_curve_camera(
        c,
        heading,
        observed,
        sigma,
        initial,
        camera_derivative=width_camera_derivative(
            data, c, prepared["scale_target_depth_model_m"]
        ),
        camera_prior_mean=mean,
        camera_prior_sigma=uncertainty["sigma_model_m"],
        camera_bound=0.1 * median_depth,
        phase_covariance=relative,
    )
    offset = joint["camera_depth_offset_model_m"]
    data["normalization_camera"] = adjusted_camera(data, c, offset)
    for key in [
        "observed_width_px",
        "width_observation_rows",
        "width_observation_centers",
    ]:
        data.pop(key)
    _, _, new_rows, new_centers = calibrated_width.observe(
        data, c["initial"], row_mode="fitted_camera"
    )
    sampled = reobserve(data, new_rows.numpy(), new_centers.numpy())
    data["observed_width_px"] = sampled["rgb_width_px"]
    data["width_observation_rows"] = sampled["width_observation_rows"]
    data["width_observation_centers"] = sampled["width_observation_centers"]
    fresh_depth = calibrated_width.estimated_depth(data, data["normalization_camera"])
    normalizer = (
        data["shared_focal"] / fresh_depth * c["reference_height_model_m"] / 256
    )[:, None]
    fresh_heading, observed, _, _ = calibrated_width.observe(
        data, c["initial"], normalizer=normalizer, row_mode="fitted_camera"
    )
    if not np.array_equal(fresh_heading, heading):
        raise InputError("Fresh RGB phase identity changed")
    c["query_noise_variance"] = _variance(heading, observed, normalizer)
    observed = (
        observed
        - correction(data, c, prepared["scale_target_depth_model_m"])["correction"]
    )
    measured = fit_curve(c, heading, observed, sigma, joint, phase_covariance=relative)
    coefficients = measured.pop("delta_coefficients")
    value = _perimeter(prepared, coefficients)
    measured["singular_values"] = measured["singular_values"].tolist()
    measured["joint_camera_stage"] = {
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in joint.items()
    }
    measured["scale_camera"] = {
        "offset_model_m": offset,
        "head_only_prior_offset_model_m": (cue_diagnostics[0]["median_depth_scale"] - 1)
        * median_depth,
        "calf_only_prior_offset_model_m": (cue_diagnostics[1]["median_depth_scale"] - 1)
        * median_depth,
        "combined_prior_offset_model_m": float(mean),
        "uncertainty": uncertainty,
        "all48_views_and41_torso_rows_retained": True,
        "scale_rule": "equal log-scale head and calf RGB cues",
        "fresh_RGB_rows_required": True,
        "single_joint_camera_proposal": True,
        "iterative_camera_optimum": False,
    }
    return {
        "perimeter_model_mm": value,
        "difference_from_calibration_model_mm": value - prepared["baseline_model_mm"],
        "delta_coefficients": coefficients.tolist(),
        "diagnostics": measured,
    }
