"""Measure validated capture observations using the common experimental core.

Native capture IDs, estimated poses and observed RGB enter this stage. No
renderer report, shape-change labels or time-derived ground truth are read.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np

from .capture_source import resolve_source
from .capture_width import measure_capture, prepare_reference
from .io import InputError, new_run, read_json, sha256, write_json


def load(cache, identifier):
    manifest = read_json(cache / "cache-manifest.json")
    rows = [r for r in manifest["cases"] if r["name"] == identifier]
    if len(rows) != 1:
        raise InputError("Capture observation identity is ambiguous")
    row = rows[0]
    file = (cache / row["cache"]).resolve()
    if file.parent != cache.resolve() or sha256(file) != row["cache_sha256"]:
        raise InputError("Frozen capture observations changed")
    with np.load(file, allow_pickle=False) as n:
        return {k: n[k] for k in n.files}, row


def image_target(observation, identifier, data, context):
    manifest = read_json(observation / "images/cache-manifest.json")
    if any(manifest.get(k) != v for k, v in context.metadata.items()):
        raise InputError("RGB observations belong to a different capture source")
    image, row = load(observation / "images", identifier)
    if (
        row["video_sha256"] != context.rows[identifier]["video_sha256"]
        or row["frame_estimate_sha256"]
        != context.rows[identifier]["frame_estimate_sha256"]
    ):
        raise InputError("RGB and native capture identities differ")
    if not np.array_equal(image["frames"], data["frames"]):
        raise InputError("RGB and native frame correspondence differs")
    data["width_target"] = image["shared_target"][
        data["shared_source_order"].astype(int)
    ]
    if data["width_target"].shape[1] != 1024:
        raise InputError("Experimental widths require the frozen dense contour")


def pose_prior(context, identifier, data, strength):
    from .phase_registration import native_headings
    from .pose_constraints import root_rotation, smooth_headings

    if strength != 64:
        raise InputError("This measurement candidate requires the fixed heading prior")
    session = context.store / context.rows[identifier]["session_id"]
    attempts = list(session.glob("attempt_*/frames.json"))
    if len(attempts) != 1:
        raise InputError("Native pose inference is ambiguous")
    frozen = {r["frame"]: r for r in read_json(attempts[0])["frames"]}
    parameters = []
    for frame in data["frames"]:
        file = attempts[0].parent / frozen[int(frame)]["output"]
        if (
            sha256(file)
            != context.rows[identifier]["frame_estimate_sha256"][str(int(frame))]
        ):
            raise InputError("Native pose estimate changed")
        with np.load(file, allow_pickle=False) as n:
            parameters.append(n["mhr_model_params"])
    local = data["root_tilt_reference"].T @ root_rotation(parameters)
    raw = np.unwrap(
        np.arctan2(local[:, 0, 2] - local[:, 2, 0], local[:, 0, 0] + local[:, 2, 2])
    )
    expected = smooth_headings(raw, strength, data["frames"])
    actual = np.unwrap(
        native_headings(data["frame_rotations"], data["root_tilt_reference"])
    )
    actual += np.rint(np.mean(expected - actual) / (2 * np.pi)) * (2 * np.pi)
    if np.max(np.abs(expected - actual)) >= 1e-5:
        raise InputError("Estimated pose smoother and measurement geometry differ")
    smoother = np.column_stack(
        [
            smooth_headings(np.eye(len(raw))[i], strength, data["frames"])
            for i in range(len(raw))
        ]
    )
    if not np.allclose(smoother @ raw, expected, atol=1e-10, rtol=0):
        raise InputError("Estimated pose covariance does not match the smoother")
    residual = raw - expected
    iid_scale = np.sqrt(np.sum((np.eye(len(raw)) - smoother) ** 2) / len(raw))
    mad = 1.4826 * np.median(abs(residual - np.median(residual)))
    heuristic_sigma = float(mad / iid_scale)
    smoothed_sigma = heuristic_sigma * np.sqrt(np.sum(smoother * smoother, axis=1))
    sigma = np.clip(heuristic_sigma, np.deg2rad(0.25), np.deg2rad(5))
    covariance = sigma**2 * smoother @ smoother.T
    order = data["shared_source_order"].astype(int)
    return {
        "sigma": smoothed_sigma[order],
        "covariance": covariance[np.ix_(order, order)],
    }


def rgb_width(context, identifier, data, fit, calibration, camera_offset, output):
    import cv2

    from .calibrated_width import observe
    from .subpixel_edges import central_positions, refine_positions

    data["normalization_camera"] = (
        data["shared_camera"] + np.asarray(fit["camera_delta"]) + camera_offset
    )
    data["depth_global_coefficients"] = calibration["global_coefficients"]
    _, normalized, rows, centers = observe(
        data, calibration["initial"], row_mode="fitted_camera"
    )
    rows, centers = rows.numpy(), centers.numpy()
    target = data["width_target"]
    original = np.array(
        [central_positions(poly, y, x) for poly, y, x in zip(target, rows, centers)]
    )
    mask_width = normalized * (data["shared_mask_height"][:, None] / 256)
    np.testing.assert_allclose(
        original[..., 1] - original[..., 0], mask_width, atol=1e-10
    )
    frames = data["frames"][data["shared_source_order"].astype(int)]
    if len(set(frames)) != len(frames):
        raise InputError("RGB frame identities are duplicated")
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    video = context.store / context.rows[identifier]["session_id"] / "capture.bin"
    if sha256(video) != context.rows[identifier]["video_sha256"]:
        raise InputError("Capture video changed before RGB measurement")
    refinements = [None] * len(frames)
    decoded = {}
    capture = cv2.VideoCapture(str(video))
    try:
        for frame in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, bgr = capture.read()
            if not ok:
                raise InputError("Native capture RGB decoding failed")
            if frame in lookup:
                i = lookup[frame]
                decoded[str(frame)] = hashlib.sha256(bgr.tobytes()).hexdigest()
                refinements[i] = refine_positions(bgr, rows[i], original[i])
    finally:
        capture.release()
    if any(r is None for r in refinements):
        raise InputError("Required native frames are missing from the RGB video")
    position = np.array([r["positions"] for r in refinements])
    data["observed_width_px"] = position[..., 1] - position[..., 0]
    data["width_observation_rows"] = rows
    data["width_observation_centers"] = centers
    np.savez_compressed(
        output,
        frames=data["frames"],
        observed_frames=frames,
        mask_width_px=mask_width,
        rgb_width_px=data["observed_width_px"],
        width_observation_rows=rows,
        width_observation_centers=centers,
        original_positions=original,
        rgb_positions=position,
        refined_edges=np.array([r["refined"] for r in refinements]),
        gradient_strength=np.array([r["gradient_strength"] for r in refinements]),
        fallback_reason=np.array([r["fallback_reason"] for r in refinements]),
    )
    output.chmod(0o600)
    return {"pixel_cache_sha256": sha256(output), "decoded_bgr_sha256": decoded}


def run(observation, output, *, scale_camera=False, anatomy_runtime_config=None):
    import cv2
    import psutil
    import torch

    from .synthetic_lab import snapshot_sources

    observation = Path(observation).resolve()
    if read_json(observation / "status.json").get("stage") != "complete_observations":
        raise InputError("Native capture observations are incomplete")
    plan = read_json(observation / "plan.json")
    cfgpath = plan["runtime_config"]
    if sha256(Path(cfgpath)) != plan["runtime_config_sha256"]:
        raise InputError("Frozen observation runtime changed")
    limits = read_json(cfgpath)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    detector = None
    anatomy_config = None
    if scale_camera:
        from .capture_scale_cues import create_detector

        if anatomy_runtime_config is None:
            raise InputError(
                "Scale-camera measurement requires an explicit frozen anatomy runtime"
            )
        anatomy_config = read_json(Path(anatomy_runtime_config))
        if anatomy_config["network_sandbox"] != limits["network_sandbox"]:
            raise InputError("Independent anatomy must use the same process isolation")
        detector = create_detector(anatomy_config)
    root = new_run(output)
    snapshot_sources(root, cfgpath)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > limits["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < limits["min_available_mb"] * 1024**2
            ):
                write_json(
                    root / "resource-failure.json",
                    {"reason": "resource_guard", "not_qc_refusal": True},
                )
                os._exit(75)
            time.sleep(0.25)

    threading.Thread(target=guard, daemon=True).start()
    query_source = observation / "contours"
    context = resolve_source(read_json(query_source / "cache-manifest.json"), cfgpath)
    if not context.direct:
        raise InputError("Measurement requires explicit verified capture inputs")
    inputs = read_json(observation / "capture-inputs.json")
    reference_id = inputs["reference_capture_id"]
    query_predictions = read_json(observation / "query-fits/predictions.json")
    if query_predictions["cache_manifest_sha256"] != sha256(
        query_source / "cache-manifest.json"
    ):
        raise InputError("Query fit provenance changed")
    fits = {r["name"]: r["fit"] for r in query_predictions["cases"]}
    reference_source = observation / "reference-contours"
    reference_context = resolve_source(
        read_json(reference_source / "cache-manifest.json"), cfgpath
    )
    if reference_context.metadata != context.metadata:
        raise InputError("Reference and query captures have different input provenance")
    reference, _ = load(reference_source, reference_id)
    geometry = read_json(reference_source / "cache-manifest.json")["reference_geometry"]
    if geometry["prediction_sha256"] != sha256(
        observation / "reference-fits/predictions.json"
    ) or geometry["parent_cache_manifest_sha256"] != sha256(
        observation / "reference-inputs/cache-manifest.json"
    ):
        raise InputError("Reference geometry/fit provenance changed")
    reference_predictions = read_json(observation / "reference-fits/predictions.json")
    if (
        reference_predictions["cache_manifest_sha256"]
        != geometry["parent_cache_manifest_sha256"]
    ):
        raise InputError("Reference fit source changed")
    reference_fit = next(
        r["fit"] for r in reference_predictions["cases"] if r["name"] == reference_id
    )
    image_target(observation, reference_id, reference, context)
    masked = prepare_reference(reference, reference_fit, fits[reference_id])
    rgb_width(
        context,
        reference_id,
        reference,
        reference_fit,
        masked["calibration"],
        np.zeros(3),
        root / "reference-rgb.npz",
    )
    prepared = prepare_reference(
        reference, reference_fit, fits[reference_id], observed_rgb=True
    )
    scale_cue_evidence = {}
    if scale_camera:
        from .capture_scale_cues import observe_cues
        from .capture_scale_width import prepare_scale_reference

        reference_cues, cue_evidence = observe_cues(
            context,
            reference_id,
            reference,
            detector,
            root / "reference-scale-cues.npz",
        )
        scale_cue_evidence["reference"] = cue_evidence
        prepared = prepare_scale_reference(prepared, reference_cues)
    reference_query, _ = load(query_source, reference_id)
    strength = read_json(observation / "query-options.json")["fit"][
        "heading_smoothness"
    ]
    reference_pose = pose_prior(context, reference_id, reference_query, strength)
    np.savez_compressed(
        root / "calibration.npz",
        **{
            k: v
            for k, v in prepared["calibration"].items()
            if isinstance(v, np.ndarray)
        },
    )
    (root / "calibration.npz").chmod(0o600)
    rows, failures = [], []
    for identifier in [reference_id, *inputs["current_capture_ids"]]:
        try:
            data, _ = load(query_source, identifier)
            image_target(observation, identifier, data, context)
            pixels = rgb_width(
                context,
                identifier,
                data,
                fits[identifier],
                masked["calibration"],
                masked["camera_geometry_offset"],
                root / (identifier + "-rgb.npz"),
            )
            if scale_camera:
                from .capture_scale_width import measure_scale_capture
                from .depth_row_observation import resample

                cues, evidence = observe_cues(
                    context,
                    identifier,
                    data,
                    detector,
                    root / (identifier + "-scale-cues.npz"),
                )
                if evidence["decoded_bgr_sha256"] != pixels["decoded_bgr_sha256"]:
                    raise InputError(
                        "Stable cues and torso widths used different RGB frames"
                    )
                scale_cue_evidence[identifier] = evidence
                video = (
                    context.store
                    / context.rows[identifier]["session_id"]
                    / "capture.bin"
                )

                def reread(sample_data, rows, centers):
                    sample, decoded = resample(
                        video, sample_data, rows, centers, pixels["decoded_bgr_sha256"]
                    )
                    file = root / (identifier + "-corrected-rgb.npz")
                    np.savez_compressed(file, **sample)
                    file.chmod(0o600)
                    pixels["corrected_pixel_cache_sha256"] = sha256(file)
                    if decoded != pixels["decoded_bgr_sha256"]:
                        raise InputError("Corrected torso RGB frames changed")
                    return sample

                measured = measure_scale_capture(
                    prepared,
                    data,
                    fits[identifier],
                    pose_prior(context, identifier, data, strength),
                    reference_pose,
                    cues,
                    reobserve=reread,
                )
            else:
                measured = measure_capture(
                    prepared,
                    data,
                    fits[identifier],
                    pose_prior(context, identifier, data, strength),
                    reference_pose,
                )
            rows.append({"capture_id": identifier, **measured, "rgb_evidence": pixels})
            print(
                "capture_measured",
                identifier,
                measured["perimeter_model_mm"],
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "capture_id": identifier,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "not_qc_refusal": True,
                }
            )
    result = {
        "source_kind": inputs["source_kind"],
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
        "not_adoption_evidence": True,
        "reference_capture_id": reference_id,
        "current_capture_ids": inputs["current_capture_ids"],
        "rows": rows,
        "failures": failures,
        "capture_inputs_sha256": sha256(observation / "capture-inputs.json"),
        "query_prediction_sha256": sha256(observation / "query-fits/predictions.json"),
        "reference_prediction_sha256": sha256(
            observation / "reference-fits/predictions.json"
        ),
        "calibration_sha256": sha256(root / "calibration.npz"),
        "sigma_interpretation": "Heuristic smoothed native-pose residual covariance; systematic pose/image bias unquantified",
        "renderer_report_access": False,
    }
    if scale_camera:
        result["measurement_candidate"] = "head_calf_joint_camera.v1"
        result["scale_cue_evidence"] = scale_cue_evidence
        result["anatomy_runtime_config_sha256"] = sha256(Path(anatomy_runtime_config))
        result["anatomy_model_sha256"] = anatomy_config["pose_model_sha256"]
        result["single_joint_camera_proposal"] = True
        result["iterative_camera_optimum"] = False
    write_json(root / "predictions.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale-camera", action="store_true")
    parser.add_argument("--anatomy-runtime-config")
    args = parser.parse_args()
    result = run(
        args.observations,
        args.out,
        scale_camera=args.scale_camera,
        anatomy_runtime_config=args.anatomy_runtime_config,
    )
    raise SystemExit(bool(result["failures"]))
