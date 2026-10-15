"""Reference calibration and paired experimental width measurement.

Callers supply validated native observations and their RGB width/pose evidence.
No renderer truth, case labels or time-derived rotation angles enter this core.
"""

from __future__ import annotations

import numpy as np

from .calibrated_width import calibrate
from .framewise_width import infer
from .io import InputError
from .local_shape import torso_basis
from .longitudinal_basis import reference_change
from .mesh import section_perimeter


def prepare_reference(reference, reference_fit, original_fit, *, observed_rgb=False):
    """Keep one common fitted reference and its fixed camera/geometry origin."""
    data = dict(reference)
    camera = np.asarray(reference_fit["camera_delta"])
    original_camera = np.asarray(original_fit["camera_delta"])
    if (
        camera.shape != original_camera.shape
        or camera.ndim != 2
        or camera.shape != (len(data["shared_source_order"]), 3)
        or not np.isfinite(camera).all()
        or not np.isfinite(original_camera).all()
    ):
        raise InputError("Reference camera/frame correspondence is invalid")
    difference = camera - original_camera
    offset = difference.mean(axis=0)
    if np.max(np.abs(difference - offset)) >= 1e-8:
        raise InputError("Reference geometry lacks a shared camera-origin offset")
    canonical_change, data["shared_change_basis"] = reference_change(
        data, reference_fit
    )
    if observed_rgb:
        data["depth_global_coefficients"] = np.asarray(reference_fit["shape_delta"])[
            :45
        ]
    calibration = calibrate(
        data,
        reference_fit,
        normalization="depth",
        row_mode="fitted_camera",
        noise_mode="uniform",
    )
    global_coefficients = np.asarray(reference_fit["shape_delta"])[:45]
    local_basis = torso_basis(data["rest_base"], data["rest_anchors"], "independent")
    base = (
        data["rest_base"]
        + data["rest_basis"] @ global_coefficients
        + local_basis @ calibration["baseline_coefficients"]
    )
    prepared = {
        "calibration": calibration,
        "reference": data,
        "canonical_change": canonical_change,
        "base_vertices": base,
        "camera_geometry_offset": offset,
        "section": {
            "y": float(data["section_y"]),
            "center_xz": data["section_center"].tolist(),
        },
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
    }
    prepared["baseline_model_mm"] = _perimeter(
        prepared, np.zeros(canonical_change.shape[-1])
    )
    return prepared


def _perimeter(prepared, coefficients):
    value = section_perimeter(
        {
            "vertices": (
                prepared["base_vertices"] + prepared["canonical_change"] @ coefficients
            ).tolist(),
            "faces": prepared["reference"]["faces"].tolist(),
        },
        prepared["section"],
    )
    if value is None or not np.isfinite(value) or value <= 0:
        raise InputError("Experimental torso section is unobservable")
    return value * 1000


def measure_capture(prepared, query, query_fit, query_pose, reference_pose):
    data = dict(query)
    reference = prepared["reference"]
    if any(
        not np.array_equal(data[key], reference[key])
        for key in ("faces", "section_y", "section_center")
    ):
        raise InputError("Query and reference measurement geometry differ")
    camera_delta = np.asarray(query_fit["camera_delta"])
    if (
        camera_delta.shape != data["shared_camera"].shape
        or not np.isfinite(camera_delta).all()
    ):
        raise InputError("Query camera/frame correspondence is invalid")
    data["normalization_camera"] = (
        data["shared_camera"] + camera_delta + prepared["camera_geometry_offset"]
    )
    measured = infer(
        prepared["calibration"],
        data,
        query_sigma=query_pose["sigma"],
        reference_sigma=reference_pose["sigma"],
        query_covariance=query_pose["covariance"],
        reference_covariance=reference_pose["covariance"],
    )
    coefficients = measured.pop("delta_coefficients")
    measured["singular_values"] = measured["singular_values"].tolist()
    value = _perimeter(prepared, coefficients)
    return {
        "perimeter_model_mm": value,
        "difference_from_calibration_model_mm": value - prepared["baseline_model_mm"],
        "delta_coefficients": coefficients.tolist(),
        "diagnostics": measured,
    }
