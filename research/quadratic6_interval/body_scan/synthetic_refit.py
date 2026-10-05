"""Synthetic-only, frozen SAM observation replay. Truth enters evaluation only."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256, write_json


def cache(config_path, source, output, names, observation_options=None):
    import fcntl
    import threading

    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO

    from .fit_strategy import build_observations
    from .research import options
    from .synthetic_lab import snapshot_sources

    root, source = new_run(output), Path(source).resolve()
    config = read_json(config_path)
    observation_options = options(observation_options) if observation_options else None
    pose_mode = (observation_options or {}).get("fit", {}).get("pose_mode", "frame")
    reference_pose, reference_initial, root_reference = None, None, None
    report = read_json(source / "report.json")
    profile = read_json(source / "store" / "profile.json")
    if (
        profile.get("source_kind") != "synthetic"
        or read_json(source / "store" / "store.json").get("synthetic_experiment")
        is not True
    ):
        raise InputError("Frozen refit requires synthetic observations")
    rows = [r for r in report["results"] if r["name"] in names]
    if set(names) != {r["name"] for r in rows} or any(
        r["state"] != "complete" for r in rows
    ):
        raise InputError("Refit requires completed source cases")
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > config["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < config["min_available_mb"] * 1024**2
            ):
                os._exit(75)
            time.sleep(0.1)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    for key in ("checkpoint", "mhr_model", "segmentation_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("Model checksum mismatch")
    snapshot_sources(root, config_path)
    state = torch.load(
        config["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    state = state.get("state_dict", state)
    mapping = state["head_pose.keypoint_mapping"].numpy()[:11]
    faces = state["head_pose.faces"].numpy()
    model = torch.jit.load(config["mhr_model"], map_location="cpu")
    detector = YOLO(config["segmentation_model"])

    def generate(shape, params):
        with torch.no_grad():
            v, s = model(
                torch.as_tensor(shape, dtype=torch.float32)[None],
                torch.as_tensor(params, dtype=torch.float32)[None],
                torch.zeros(1, 72),
            )
        v, s = v[0].numpy() / 100, s[0, :, :3].numpy() / 100
        return v, mapping @ np.concatenate([v, s])

    def generate_many(shapes, params):
        with torch.no_grad():
            vertices, skeleton = model(
                torch.as_tensor(shapes, dtype=torch.float32),
                torch.as_tensor(params, dtype=torch.float32)[None].repeat(
                    len(shapes), 1
                ),
                torch.zeros(len(shapes), 72),
            )
        vertices, skeleton = vertices.numpy() / 100, skeleton[:, :, :3].numpy() / 100
        return vertices, mapping @ np.concatenate([vertices, skeleton], 1)

    generate.batch = generate_many

    manifest = {
        "source_kind": "synthetic",
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
        "source": str(source),
        "source_report_sha256": sha256(source / "report.json"),
        "profile_sha256": sha256(source / "store" / "profile.json"),
        "cases": [],
        "observation_options": observation_options,
    }
    scales = np.asarray(profile["skeleton_scales"], np.float32)
    indices = profile["fit_vertex_indices"]
    for row in rows:
        started = time.monotonic()
        session = source / "store" / row["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Capture changed")
        attempts = list(session.glob("attempt_*/fit.json"))
        if len(attempts) != 1:
            raise InputError("Expected one source fit")
        attempt = attempts[0].parent
        original_fit = read_json(attempt / "fit.json")
        frame_rows = read_json(attempt / "frames.json")["frames"]
        wanted = {r["frame"]: r for r in frame_rows if r["state"] == "estimated"}
        capture = cv2.VideoCapture(str(session / "capture.bin"))
        estimates, frame_hashes, qualities = [], {}, []
        try:
            for index in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok, rgb = capture.read()
                if not ok:
                    raise InputError("Capture decode failed")
                if index not in wanted:
                    continue
                frame = wanted[index]
                path = attempt / frame["output"]
                frame_hashes[str(index)] = sha256(path)
                with np.load(path, allow_pickle=False) as data:
                    value = {k: data[k] for k in data.files}
                if (
                    (observation_options or {})
                    .get("fit", {})
                    .get("heading_smoothness", 0)
                ) or (observation_options or {}).get("fit", {}).get(
                    "heading_model", "acceleration"
                ) != "acceleration":
                    value["frame_index"] = np.asarray(index, np.int64)
                prediction = detector.predict(
                    rgb,
                    classes=[0],
                    device="cpu",
                    retina_masks=True,
                    verbose=False,
                    save=False,
                )[0]
                masks = prediction.masks.data
                from .quality import select_primary_person

                primary = select_primary_person([float(m.sum()) for m in masks])
                mask = masks[primary].numpy() > 0.5
                if (observation_options or {}).get("fit", {}).get(
                    "mask_refinement"
                ) == "grabcut":
                    from .mask_refinement import refine_person_mask

                    mask, _ = refine_person_mask(rgb, mask)
                estimates.append((frame["yaw_rad"], value, mask))
                # Normalized edge spread is measured without renderer masks.
                from .quality import image_detail

                qualities.append(image_detail(rgb, mask))
        finally:
            capture.release()
        # Ultralytics changes Torch's global CPU pool when selecting its device.
        # Restore the configured budget before building the MHR bases.
        torch.set_num_threads(config["cpu_threads"])
        initial = np.asarray(original_fit["initial_shape_params"], np.float32)
        if (observation_options or {}).get("fit", {}).get(
            "initialization"
        ) == "profile":
            if reference_initial is None:
                if row["spec"]["group"] != "baseline":
                    raise InputError(
                        "Shared initialization requires an observed baseline first"
                    )
                reference_initial = initial.copy()
            initial = reference_initial.copy()
        shapes = np.stack([v["shape_params"] for _, v, _ in estimates])
        arrays = {
            "initial": initial,
            "shape_scale": np.maximum(shapes.std(0), 0.1),
            "shapes": shapes,
            "yaw": np.array([y for y, _, _ in estimates]),
            "frames": np.array(sorted(wanted)),
            "faces": faces,
            "section_y": np.array(profile["sections"][0]["y"]),
            "section_center": np.array(profile["sections"][0]["center_xz"]),
            "quality": np.array([q["edge_energy"] for q in qualities]),
        }
        params = np.zeros(204, np.float32)
        params[136:] = scales
        rest_shapes = np.vstack([initial, initial + np.eye(45, dtype=np.float32)])
        rest_batch, rest_joints = generate_many(rest_shapes, params)
        rest = rest_batch[0]
        arrays["rest_base"] = rest
        arrays["rest_basis"] = np.moveaxis(rest_batch[1:] - rest, 0, -1)
        arrays["rest_anchors"] = rest_joints[0, [5, 6, 9, 10]]
        arrays["rest_anchor_basis"] = np.moveaxis(
            rest_joints[1:, [5, 6, 9, 10]] - arrays["rest_anchors"], 0, -1
        )
        if pose_mode != "frame":
            if pose_mode == "shared_recording" or reference_pose is None:
                reference_pose = np.median(
                    [v["mhr_model_params"][6:136] for _, v, _ in estimates], axis=0
                )
            arrays["pose_reference"] = reference_pose
        if (observation_options or {}).get("fit", {}).get(
            "root_mode"
        ) == "profile_tilt_xyz":
            from .pose_constraints import root_tilt_reference

            if root_reference is None:
                root_reference = root_tilt_reference(
                    [v["mhr_model_params"] for _, v, _ in estimates]
                )
            arrays["root_tilt_reference"] = root_reference
        variants = (
            (("shared", True),)
            if observation_options
            else (("native", False), ("shared", True))
        )
        for label, shared in variants:
            research = {
                **(observation_options or read_json(source / "research-options.json")),
                "fit": {
                    **(observation_options or {}).get("fit", {}),
                    "anchor_weight": 1,
                    "shared_skeleton": shared,
                },
            }
            observations = build_observations(
                estimates,
                initial,
                scales,
                faces,
                generate,
                indices,
                research,
                pose_reference=reference_pose,
                root_reference=root_reference,
            )
            for key in observations[0]:
                arrays[label + "_" + key] = np.array([o[key] for o in observations])
        file = root / (row["name"] + ".npz")
        np.savez_compressed(file, **arrays)
        file.chmod(0o600)
        manifest["cases"].append(
            {
                "name": row["name"],
                "cache": file.name,
                "cache_sha256": sha256(file),
                "video_sha256": row["video_sha256"],
                "frame_estimate_sha256": frame_hashes,
                "old_shape_params": original_fit["shape_params"],
                "old_perimeter_model_m": row["estimated_perimeter_model_m"],
                "elapsed_s": time.monotonic() - started,
                "torch_threads": torch.get_num_threads(),
            }
        )
        replace_json(root / "cache-manifest.json", manifest)
        print("cached", row["name"], round(time.monotonic() - started, 1), flush=True)


def run(cache_root, output, research, selection=None):
    import numpy as np
    import torch

    from .capture_source import resolve_source, supported_observations
    from .fit_strategy import solve
    from .mesh import section_perimeter
    from .research import options

    torch.set_num_threads(2)
    root, cache_root = new_run(output), Path(cache_root).resolve()
    manifest, research = (
        read_json(cache_root / "cache-manifest.json"),
        options(research),
    )
    if not supported_observations(manifest):
        raise InputError("Refit requires a synthetic cache")
    write_json(root / "research-options.json", research)
    observed = manifest.get("observation_options") or {}
    if observed.get("fit", {}).get("contour_primitive", "vertex") != research.get(
        "fit", {}
    ).get("contour_primitive", "vertex"):
        raise InputError("Contour primitive differs from frozen observations")
    if observed.get("fit", {}).get("heading_model", "acceleration") != research.get(
        "fit", {}
    ).get("heading_model", "acceleration"):
        raise InputError("Heading model does not match the frozen geometry")
    if observed.get("fit", {}).get("heading_smoothness", 0) != research.get(
        "fit", {}
    ).get("heading_smoothness", 0):
        raise InputError("Heading smoothness does not match the frozen geometry")
    if (research.get("fit", {}).get("distance_mode") == "segments") != (
        observed.get("fit", {}).get("distance_mode") == "segments"
    ):
        raise InputError("Segment distances require a matching ordered contour cache")
    if observed.get("fit", {}).get("contour_samples", 256) != research.get(
        "fit", {}
    ).get("contour_samples", 256):
        raise InputError("Contour sample count differs from frozen observations")
    for key, default in (
        ("pose_mode", "frame"),
        ("mask_refinement", "none"),
        ("root_mode", "frame"),
        ("anchor_source", "sam_projection"),
    ):
        if observed.get("fit", {}).get(key, default) != research.get("fit", {}).get(
            key, default
        ):
            # Older RGB caches record their mask variant separately.
            if (
                key == "mask_refinement"
                and manifest.get("observation_variant") == "observed_rgb_grabcut.v1"
                and research.get("fit", {}).get(key) == "grabcut"
            ):
                continue
            raise InputError("Observation cache does not match requested " + key)
    if (
        research.get("fit", {}).get("anchor_source") == "yolo17"
        and manifest.get("anchor_observer_model_sha256")
        != research["fit"]["anchor_model_sha256"]
    ):
        raise InputError("Independent anatomy observation model changed")
    results = {
        "source_kind": manifest["source_kind"],
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
        "cache_manifest_sha256": sha256(cache_root / "cache-manifest.json"),
        "cases": [],
    }
    context = (
        resolve_source(manifest) if "capture_inputs_manifest" in manifest else None
    )
    reference_initial = None
    if research.get("fit", {}).get("initialization") == "profile":
        reference_row = manifest["cases"][0]
        source_context = context or resolve_source(manifest)
        if not source_context.is_reference(source_context.rows[reference_row["name"]]):
            raise InputError(
                "Shared initialization requires an observed baseline first"
            )
        reference_path = cache_root / reference_row["cache"]
        if sha256(reference_path) != reference_row["cache_sha256"]:
            raise InputError("Reference observation cache changed")
        with np.load(reference_path, allow_pickle=False) as n:
            reference_initial = n["initial"].copy()
    for row in manifest["cases"]:
        file = cache_root / row["cache"]
        if file.parent != cache_root or sha256(file) != row["cache_sha256"]:
            raise InputError("Frozen observation cache changed")
        with np.load(file, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files}
        label = "shared" if research.get("fit", {}).get("shared_skeleton") else "native"
        keys = [k.removeprefix(label + "_") for k in data if k.startswith(label + "_")]
        picked = (
            np.arange(len(data[label + "_base"]))
            if selection is None
            else selection(data)
        )
        observations = [{k: data[label + "_" + k][i] for k in keys} for i in picked]
        if reference_initial is not None:
            shift = reference_initial - data["initial"]
            for o in observations:
                o["base"] = o["base"] + o["basis"] @ shift
                if "anchor_base" in o:
                    o["anchor_base"] = o["anchor_base"] + o["anchor_basis"] @ shift
            data["rest_base"] = data["rest_base"] + data["rest_basis"] @ shift
            if "rest_anchor_basis" in data:
                data["rest_anchors"] = (
                    data["rest_anchors"] + data["rest_anchor_basis"] @ shift
                )
            data["initial"] = reference_initial
        started = time.monotonic()
        fit = solve(observations, data["shape_scale"], research)
        global_delta = fit["shape_delta"][: len(data["initial"])]
        local_coefficients = fit["shape_delta"][len(data["initial"]) :]
        vertices = data["rest_base"] + data["rest_basis"] @ global_delta
        if len(local_coefficients):
            from .local_shape import torso_basis

            if "rest_anchors" not in data:
                raise InputError("Local refit requires estimated rest anatomy")
            vertices += (
                torso_basis(
                    data["rest_base"],
                    data["rest_anchors"],
                    research["fit"]["local_mode"],
                )
                @ local_coefficients
            )
            fit["local_shape_limit"] = research["fit"].get("local_limit", 0.1)
            fit["local_shape_at_bound"] = bool(
                np.max(abs(local_coefficients)) >= 0.98 * fit["local_shape_limit"]
            )
        section = {
            "y": float(data["section_y"]),
            "center_xz": data["section_center"].tolist(),
        }
        perimeter = section_perimeter(
            {"vertices": vertices.tolist(), "faces": data["faces"].tolist()}, section
        )
        if perimeter is None:
            raise InputError("Ambiguous refitted section")
        measurement_diagnostics = {}
        if research.get("fit", {}).get("diagnostics"):
            initial_perimeter = section_perimeter(
                {
                    "vertices": data["rest_base"].tolist(),
                    "faces": data["faces"].tolist(),
                },
                section,
            )
            probes = []
            for offset in (-0.005, 0.0, 0.005):
                measured = section_perimeter(
                    {"vertices": vertices.tolist(), "faces": data["faces"].tolist()},
                    {**section, "y": section["y"] + offset},
                )
                probes.append(
                    {
                        "plane_offset_model_mm": offset * 1000,
                        "perimeter_model_m": measured,
                    }
                )
            measurement_diagnostics = {
                "initial_perimeter_model_m": initial_perimeter,
                "section_definition": section,
                "section_probes": probes,
                "diagnostic_only": True,
            }
        results["cases"].append(
            {
                "name": row["name"],
                "perimeter_model_m": perimeter,
                "shape_params": (data["initial"] + global_delta).tolist(),
                "fit": {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in fit.items()
                },
                "frame_indices": data["frames"][picked].tolist(),
                "edge_energy_median": float(np.median(data["quality"][picked])),
                "elapsed_s": time.monotonic() - started,
                **(
                    {"measurement_diagnostics": measurement_diagnostics}
                    if measurement_diagnostics
                    else {}
                ),
            }
        )
        replace_json(root / "predictions.json", results)
        print("refit", row["name"], round(perimeter * 1000, 3), flush=True)
    return results


def evaluate(cache_root, output):
    """Only this post-fit stage reads the source renderer's truth annotations."""
    manifest = read_json(Path(cache_root) / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or "capture_inputs_manifest" in manifest
    ):
        raise InputError("Renderer truth evaluation requires a synthetic lab report")
    source = Path(manifest["source"])
    if sha256(source / "report.json") != manifest["source_report_sha256"]:
        raise InputError("Source report changed")
    report = read_json(source / "report.json")
    specs = {r["name"]: r for r in report["results"]}
    predictions = read_json(Path(output) / "predictions.json")
    rows = {r["name"]: r for r in predictions["cases"]}
    base = [r for name, r in rows.items() if specs[name]["spec"]["group"] == "baseline"]
    null = max(
        (
            abs(a["perimeter_model_m"] - b["perimeter_model_m"]) * 1000
            for a, b in itertools.combinations(base, 2)
        ),
        default=None,
    )
    pairs = []
    for name, row in rows.items():
        if specs[name]["spec"]["group"] != "change":
            continue
        for reference in base:
            truth = specs[name]["truth"]["actual_delta_model_m"] * 1000
            predicted = (
                row["perimeter_model_m"] - reference["perimeter_model_m"]
            ) * 1000
            correct = predicted * truth > 0
            pairs.append(
                {
                    "name": name,
                    "reference": reference["name"],
                    "truth_model_mm": truth,
                    "prediction_model_mm": predicted,
                    "correct_direction": correct,
                    "margin": bool(
                        correct and null is not None and abs(predicted) > 3 * null
                    ),
                }
            )
    distances = [
        r for name, r in rows.items() if specs[name]["spec"]["group"] == "distance"
    ]
    distance = max(
        (
            200
            * abs(a["perimeter_model_m"] - b["perimeter_model_m"])
            / (a["perimeter_model_m"] + b["perimeter_model_m"])
            for a, b in itertools.combinations(distances, 2)
        ),
        default=None,
    )
    result = {
        "baseline_max_null_model_mm": null,
        "direction_pairs": sum(p["correct_direction"] for p in pairs),
        "margin_pairs": sum(p["margin"] for p in pairs),
        "planned_pairs": len(pairs),
        "distance_max_symmetric_percent": distance,
        "change_pairs": pairs,
        "stress": [
            {
                "name": name,
                "difference_percent": 200
                * abs(r["perimeter_model_m"] - base[0]["perimeter_model_m"])
                / (r["perimeter_model_m"] + base[0]["perimeter_model_m"]),
            }
            for name, r in rows.items()
            if specs[name]["spec"]["group"] == "stress" and base
        ],
    }
    replace_json(Path(output) / "evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cache", "run"))
    parser.add_argument("--runtime-config")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--options")
    args = parser.parse_args()
    if args.mode == "cache":
        cache(
            args.runtime_config,
            args.source,
            args.out,
            args.cases,
            read_json(args.options) if args.options else None,
        )
    else:
        run(args.source, args.out, read_json(args.options))
        print(evaluate(args.source, args.out))
