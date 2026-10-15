"""Explicit synthetic research variants; not enabled in the capture UI."""

from __future__ import annotations

import hashlib
import json
import math

from .io import InputError


def options(data: dict | None) -> dict:
    if data is None:
        return {}
    if not isinstance(data, dict) or set(data) - {
        "camera",
        "region",
        "region_weight",
        "shape_weight",
        "qc",
        "fit",
    }:
        raise InputError("Unknown research option")
    out = dict(data)
    camera = data.get("camera")
    if camera is not None:
        if not isinstance(camera, dict) or set(camera) != {
            "width",
            "height",
            "fx",
            "fy",
            "cx",
            "cy",
            "provenance",
        }:
            raise InputError("Explicit camera intrinsics and provenance are required")
        if any(
            isinstance(camera[k], bool)
            or not isinstance(camera[k], (int, float))
            or not math.isfinite(camera[k])
            for k in ("width", "height", "fx", "fy", "cx", "cy")
        ):
            raise InputError("Invalid camera intrinsics")
        if (
            any(
                not isinstance(camera[k], int) or not 32 <= camera[k] <= 4096
                for k in ("width", "height")
            )
            or min(camera["fx"], camera["fy"]) <= 0
            or not 0 <= camera["cx"] < camera["width"]
            or not 0 <= camera["cy"] < camera["height"]
            or not isinstance(camera["provenance"], str)
            or not camera["provenance"]
        ):
            raise InputError("Invalid camera intrinsics")
        # The current contour projection assumes a centred square-pixel camera.
        if (
            camera["fx"] != camera["fy"]
            or camera["cx"] != camera["width"] / 2
            or camera["cy"] != camera["height"] / 2
        ):
            raise InputError(
                "Research contour fitter requires centred square-pixel intrinsics"
            )
    if data.get("region", "full") not in {"full", "torso", "abdomen"}:
        raise InputError("Unknown contour research region")
    for key, default, low, high in (
        ("region_weight", 1.0, 1, 32),
        ("shape_weight", 0.1, 0, 100),
    ):
        value = data.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise InputError("Invalid research fit weight")
    qc = data.get("qc", {})
    if not isinstance(qc, dict) or set(qc) - {
        "mask_support",
        "max_relative_fit_error",
        "require_rotation",
        "min_image_detail",
        "min_boundary_detail",
    }:
        raise InputError("Unknown research QC option")
    for key in ("mask_support", "require_rotation"):
        if key in qc and not isinstance(qc[key], bool):
            raise InputError("Research QC flags must be boolean")
    if "max_relative_fit_error" in qc:
        value = qc["max_relative_fit_error"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 0.1
        ):
            raise InputError("Invalid relative fit error threshold")
    for threshold in ("min_image_detail", "min_boundary_detail"):
        if threshold not in qc:
            continue
        value = qc[threshold]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 100
        ):
            raise InputError("Invalid image-detail threshold")
    fit = data.get("fit", {})
    if not isinstance(fit, dict) or set(fit) - {
        "normalize_observation",
        "anchor_weight",
        "pose_limit_deg",
        "pose_weight",
        "shared_skeleton",
        "refresh_contour",
        "max_frames",
        "diagnostics",
        "contour_softening",
        "anchor_softening",
        "shape_scale_mode",
        "camera_mode",
        "steps",
        "optimizer_mode",
        "distance_mode",
        "contour_primitive",
        "contour_samples",
        "mask_refinement",
        "local_mode",
        "local_weight",
        "global_shape",
        "initialization",
        "pose_mode",
        "root_mode",
        "local_transport",
        "anchor_source",
        "anchor_model_sha256",
        "local_limit",
        "heading_smoothness",
        "heading_model",
    }:
        raise InputError("Unknown research nuisance option")
    for key in (
        "normalize_observation",
        "shared_skeleton",
        "diagnostics",
        "global_shape",
    ):
        if key in fit and not isinstance(fit[key], bool):
            raise InputError("Nuisance fit flags must be boolean")
    for key, high in (
        ("anchor_weight", 32),
        ("pose_limit_deg", 10),
        ("pose_weight", 100),
        ("refresh_contour", 1),
    ):
        if key in fit:
            value = fit[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= high
            ):
                raise InputError("Invalid nuisance fit setting")
    if fit.get("pose_limit_deg", 0) and not fit.get("anchor_weight", 0):
        raise InputError("Yaw refinement requires landmark anchors")
    if fit.get("refresh_contour", 0) not in (0, 1):
        raise InputError("Contour refresh must be zero or one")
    for key in ("contour_softening", "anchor_softening"):
        if key in fit and (
            isinstance(fit[key], bool)
            or not isinstance(fit[key], (int, float))
            or not math.isfinite(fit[key])
            or not 0.05 <= fit[key] <= 8
        ):
            raise InputError("Invalid robust-loss softening")
    if fit.get("shape_scale_mode", "frame_std") not in ("frame_std", "fixed"):
        raise InputError("Invalid shape prior scale mode")
    if fit.get("camera_mode", "free") not in (
        "free",
        "fixed",
        "fixed_depth",
        "shared_depth",
        "shared",
    ):
        raise InputError("Invalid camera fit mode")
    if fit.get("distance_mode", "legacy") not in ("legacy", "direct", "segments"):
        raise InputError("Invalid contour distance mode")
    if fit.get("contour_primitive", "vertex") not in (
        "vertex",
        "continuous_visible_edge64_viewport",
    ) or (
        fit.get("contour_primitive") == "continuous_visible_edge64_viewport"
        and fit.get("distance_mode") != "segments"
    ):
        raise InputError("Invalid silhouette contour primitive")
    if fit.get("optimizer_mode", "adam") not in ("adam", "adam_lbfgs64"):
        raise InputError("Invalid contour optimizer mode")
    if fit.get("contour_samples", 256) not in (256, 512, 1024):
        raise InputError("Invalid contour sample count")
    heading = fit.get("heading_smoothness", 0)
    if (
        isinstance(heading, bool)
        or not isinstance(heading, (int, float))
        or not math.isfinite(heading)
        or not 0 <= heading <= 100000
    ):
        raise InputError("Invalid heading smoothness")
    if fit.get("heading_model", "acceleration") not in (
        "acceleration",
        "harmonic_bias",
    ):
        raise InputError("Invalid heading motion model")
    if (
        heading or fit.get("heading_model", "acceleration") != "acceleration"
    ) and fit.get("root_mode") != "profile_tilt_xyz":
        raise InputError(
            "Heading smoothing requires the native xyz standing-axis model"
        )
    if fit.get("mask_refinement", "none") not in ("none", "grabcut"):
        raise InputError("Invalid mask refinement mode")
    if fit.get("local_mode", "none") not in ("none", "radial", "independent"):
        raise InputError("Invalid anatomical residual mode")
    if fit.get("initialization", "frame_median") not in ("frame_median", "profile"):
        raise InputError("Invalid shared shape initialization")
    if fit.get("pose_mode", "frame") not in ("frame", "shared_recording", "profile"):
        raise InputError("Invalid local pose stabilization mode")
    if fit.get("root_mode", "frame") not in (
        "frame",
        "profile_tilt",
        "profile_tilt_xyz",
    ):
        raise InputError("Invalid root stabilization mode")
    if fit.get("local_transport", "posed") not in ("posed", "canonical"):
        raise InputError("Invalid local shape transport mode")
    if "local_limit" in fit and (
        isinstance(fit["local_limit"], bool)
        or not isinstance(fit["local_limit"], (int, float))
        or not math.isfinite(fit["local_limit"])
        or not 0.01 <= fit["local_limit"] <= 0.25
    ):
        raise InputError("Invalid local shape deformation limit")
    if fit.get("anchor_source", "sam_projection") not in ("sam_projection", "yolo17"):
        raise InputError("Invalid anatomy observation source")
    if fit.get("anchor_source") == "yolo17":
        digest = fit.get("anchor_model_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or not fit.get("anchor_weight", 0)
        ):
            raise InputError(
                "Independent anatomy requires its model checksum and anchors"
            )
    if fit.get("local_transport") == "canonical" and (
        fit.get("local_mode", "none") == "none" or not fit.get("shared_skeleton")
    ):
        raise InputError(
            "Canonical transport requires a local residual and shared skeleton"
        )
    if fit.get("local_mode", "none") != "none" and (
        not fit.get("anchor_weight", 0) or not fit.get("shared_skeleton")
    ):
        raise InputError("Local shape requires anatomy anchors and shared skeleton")
    if fit.get("global_shape") is False and fit.get("local_mode", "none") == "none":
        raise InputError("Disabling global shape requires an anatomical residual")
    if "local_weight" in fit and (
        isinstance(fit["local_weight"], bool)
        or not isinstance(fit["local_weight"], (int, float))
        or not math.isfinite(fit["local_weight"])
        or not 0 <= fit["local_weight"] <= 100
    ):
        raise InputError("Invalid anatomical residual prior")
    if "steps" in fit and (
        isinstance(fit["steps"], bool)
        or not isinstance(fit["steps"], int)
        or not 1 <= fit["steps"] <= 2000
    ):
        raise InputError("Invalid fit step budget")
    if "max_frames" in fit and (
        isinstance(fit["max_frames"], bool)
        or not isinstance(fit["max_frames"], int)
        or not 12 <= fit["max_frames"] <= 48
    ):
        raise InputError("Fit sampling budget must be 12 to 48 observed frames")
    return out


def method_id(data: dict) -> str:
    encoded = json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return (
        "fixed_pose_silhouette_fit.research.v4."
        + hashlib.sha256(encoded).hexdigest()[:16]
    )


def fit_region(vertices, joints, region: str):
    """Define a canonical region from estimated anatomy, without oracle inputs."""
    import numpy as np

    hip = (joints[9] + joints[10]) / 2
    shoulder = (joints[5] + joints[6]) / 2
    span = shoulder[1] - hip[1]
    if span <= 0 or region not in {"torso", "abdomen"}:
        raise InputError("Invalid canonical research region")
    half_width = max(abs(joints[5, 0] - shoulder[0]), abs(joints[6, 0] - shoulder[0]))
    lo, hi = (0.0, 1.0) if region == "torso" else (0.1, 0.65)
    indices = np.where(
        (vertices[:, 1] >= hip[1] + lo * span)
        & (vertices[:, 1] <= hip[1] + hi * span)
        & (abs(vertices[:, 0] - hip[0]) <= half_width)
    )[0]
    if len(indices) < 100:
        raise InputError("Insufficient canonical research region")
    return indices.tolist()


def torso_mask_support(points, mask) -> dict:
    """Check mask coverage at three estimated torso centres, not visibility proof."""
    import numpy as np

    points, mask = np.asarray(points), np.asarray(mask, dtype=bool)
    if points.shape != (70, 2) or mask.ndim != 2 or not np.isfinite(points).all():
        raise InputError("Invalid mask-support observations")
    shoulder, hip = points[[5, 6]].mean(0), points[[9, 10]].mean(0)
    radius = max(3, round(float(np.linalg.norm(shoulder - hip)) * 0.04))
    coverage = []
    for ratio in (0.2, 0.5, 0.8):
        center = shoulder * (1 - ratio) + hip * ratio
        x, y = (int(coordinate) for coordinate in np.rint(center))
        # Imputed landmarks can be far outside the image. Clamp both ends,
        # so an empty patch becomes unsupported rather than a negative array.
        yy, xx = np.mgrid[
            max(0, min(mask.shape[0], y - radius)) : max(
                0, min(mask.shape[0], y + radius + 1)
            ),
            max(0, min(mask.shape[1], x - radius)) : max(
                0, min(mask.shape[1], x + radius + 1)
            ),
        ]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
        coverage.append(float(mask[yy, xx][disk].mean()) if disk.any() else 0.0)
    return {
        "supported": min(coverage) >= 0.2,
        "coverage": coverage,
        "minimum_coverage": 0.2,
        "radius_px": radius,
    }


def relative_fit_error(loss: float, mask_heights) -> float:
    import numpy as np

    heights = np.asarray(mask_heights, dtype=float)
    if (
        heights.ndim != 1
        or heights.size == 0
        or not np.isfinite(heights).all()
        or (heights <= 0).any()
        or not math.isfinite(loss)
        or loss < 0
    ):
        raise InputError("Invalid normalized fit observations")
    return float(loss / np.median(heights))
