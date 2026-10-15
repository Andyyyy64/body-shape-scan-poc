"""Deterministic synthetic video experiments; never physical accuracy evidence.

The renderer's mesh/camera/masks are withheld from YOLO and SAM. Workers use
the production pipeline, its resource guard and its process network isolation.
MHR geometry is a favourable same-family control, not independent human truth.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256, write_json
from .network_sandbox import sandbox_command, verify_network_sandbox


def launch(config_path: str, command: list[str], log_path: Path) -> int:
    config = read_json(config_path)
    verify_network_sandbox(config["network_sandbox"], config["python"])
    with log_path.open("xb") as log:
        log_path.chmod(0o600)
        return subprocess.run(
            sandbox_command(
                config["network_sandbox"], [config["python"], "-P", *command]
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
                "OMP_NUM_THREADS": str(config["cpu_threads"]),
                "MKL_NUM_THREADS": str(config["cpu_threads"]),
                "OPENBLAS_NUM_THREADS": str(config["cpu_threads"]),
                "NUMEXPR_NUM_THREADS": str(config["cpu_threads"]),
            },
        ).returncode


def export_worker(config_path: str, output: str) -> None:
    import fcntl
    import threading

    import psutil

    config, root = read_json(config_path), Path(output)
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > config["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < config["min_available_mb"] * 1024**2
            ):
                write_json(root / "resource_failure.json", {"status": "resource_guard"})
                os._exit(75)
            time.sleep(0.1)

    threading.Thread(target=guard, daemon=True).start()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MOMENTUM_ENABLED="0",
        BODY_SCAN_DINO_SOURCE=config["dino_source"],
    )
    import numpy as np
    import torch

    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    for key in ("checkpoint", "mhr_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("Model checksum mismatch")
    sys.path.insert(0, config["sam_source"])
    from sam_3d_body import load_sam_3d_body

    model, _ = load_sam_3d_body(
        config["checkpoint"], device="cpu", mhr_path=config["mhr_model"]
    )
    head = model.head_pose
    with torch.no_grad():
        vertices, skeleton = head.mhr(
            torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72)
        )
    vertices = vertices[0].numpy() / 100
    skeleton = skeleton[0, :, :3].numpy() / 100
    joints = head.keypoint_mapping.numpy() @ np.concatenate([vertices, skeleton])
    faces = head.faces.numpy()
    hip, shoulder = (joints[9] + joints[10]) / 2, (joints[5] + joints[6]) / 2
    section_y = float(hip[1] + 0.32 * (shoulder[1] - hip[1]))
    np.savez_compressed(
        root / "body.npz", vertices=vertices, faces=faces, joints=joints
    )
    (root / "body.npz").chmod(0o600)
    write_json(
        root / "body.json",
        {
            "source_kind": "synthetic",
            "generator": "zero_shape_zero_pose_MHR",
            "units": "model_m",
            "section_y": section_y,
            "hip_y": float(hip[1]),
            "shoulder_y": float(shoulder[1]),
            "center_xz": [float(hip[0]), float(hip[2])],
            "body_sha256": sha256(root / "body.npz"),
            "engine_id": config["engine_id"],
            "limitations": [
                "Same MHR family as estimator; favourable inverse-problem control",
                "Declared lengths are renderer model units, not calibrated human metres",
                "Procedural appearance, no photorealism or population coverage claim",
            ],
        },
    )


def cases() -> list[dict]:
    rows = []

    def add(name, **values):
        rows.append({"name": name, "distance": 2.0, "seed": 17, **values})

    for seed in (17, 23, 41):
        add(f"baseline_{seed}", seed=seed, group="baseline", nuisance="mild")
    for delta in (-0.02, -0.01, 0.01, 0.02, 0.04):
        add(f"change_{delta:+.3f}", delta=delta, group="change", nuisance="mild")
    for distance in (1.5, 2.0, 2.5):
        for seed in (53, 67):
            add(
                f"distance_{distance}_{seed}",
                distance=distance,
                seed=seed,
                group="distance",
            )
    for nuisance in (
        "plain",
        "clutter",
        "dim",
        "backlight",
        "blur",
        "noise",
        "compression",
        "camera_jitter",
        "breathing",
        "cloth",
        "asymmetry",
        "partial_occlusion",
        "large_occlusion",
        "two_people",
        "cropped_head",
        "partial_rotation",
        "blank",
        "flicker",
        "combined",
    ):
        add(f"stress_{nuisance}", nuisance=nuisance, group="stress")
    return rows


def protocol() -> dict:
    return {
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "thresholds": {
            "distance_max_symmetric_percent": 1.0,
            "imaging_stress_false_change_percent": 1.0,
            "change_null_margin_multiplier": 3,
        },
        "expected_rejections": [
            "stress_blank",
            "stress_two_people",
            "stress_cropped_head",
            "stress_large_occlusion",
        ],
        "expected_warning_or_rejection": ["stress_partial_rotation"],
        "geometry_confounds": [
            "stress_cloth",
            "stress_asymmetry",
            "stress_breathing",
            "stress_combined",
        ],
        "imaging_stress": [
            "stress_plain",
            "stress_clutter",
            "stress_dim",
            "stress_backlight",
            "stress_blur",
            "stress_noise",
            "stress_compression",
            "stress_camera_jitter",
            "stress_partial_occlusion",
            "stress_flicker",
        ],
        "interpretation": "Exploratory synthetic thresholds, saved before execution. No human adoption decision follows.",
        "pose": "MHR rest pose; not a physical relaxed-arms-down human capture",
        "repeats": "Seeds vary initial yaw; not independent human recaptures",
    }


def snapshot_sources(root: Path, config_path: str):
    snapshot = new_run(root / "implementation")
    names = (
        "synthetic_lab.py",
        "coverage_renderer.py",
        "calibrated_width.py",
        "framewise_width.py",
        "phase_registration.py",
        "longitudinal_basis.py",
        "subpixel_edges.py",
        "contour_width.py",
        "silhouette_edges.py",
        "capture_manifest.py",
        "capture_width.py",
        "capture_source.py",
        "capture_observation.py",
        "capture_measurement.py",
        "capture_scale_cues.py",
        "capture_scale_width.py",
        "framewise_camera.py",
        "camera_prior_uncertainty.py",
        "camera_width_response.py",
        "depth_row_observation.py",
        "perspective_width.py",
        "head_scale.py",
        "head_anatomy_scale.py",
        "leg_scale.py",
        "pipeline.py",
        "contours.py",
        "shared_shape.py",
        "fit_strategy.py",
        "fit_diagnostics.py",
        "contour_distance.py",
        "mask_refinement.py",
        "local_shape.py",
        "pose_constraints.py",
        "image_anchors.py",
        "synthetic_local_transport.py",
        "synthetic_image_anchors.py",
        "synthetic_pose_cache.py",
        "synthetic_contour_refit.py",
        "synthetic_heading_refit.py",
        "synthetic_image_cache.py",
        "synthetic_identifiability.py",
        "synthetic_mask_refit.py",
        "sampling.py",
        "image_quality.py",
        "synthetic_refit.py",
        "quality.py",
        "devices.py",
        "network_sandbox.py",
        "research.py",
        "vision.py",
        "mesh.py",
        "geometry.py",
    )
    for name in names:
        target = snapshot / name
        shutil.copyfile(Path(__file__).with_name(name), target)
        target.chmod(0o600)
    write_json(
        snapshot / "hashes.json", {name: sha256(snapshot / name) for name in names}
    )
    shutil.copyfile(config_path, snapshot / "runtime-config.json")
    (snapshot / "runtime-config.json").chmod(0o600)


def deform(vertices, body: dict, delta: float = 0, variant: str = "plain"):
    import numpy as np

    from .mesh import section_perimeter

    section = {"y": body["section_y"], "center_xz": body["center_xz"]}
    faces = body["faces"]

    def perimeter(v):
        result = section_perimeter(
            {"vertices": v.tolist(), "faces": faces.tolist()}, section
        )
        if result is None:
            raise InputError("Synthetic torso intersection is ambiguous")
        return result

    reference = perimeter(vertices)
    center = np.asarray(body["center_xz"])
    radial = vertices[:, [0, 2]] - center
    # Local, smooth deformation excludes arms away from the torso. This is
    # deliberately outside the estimator's 45-dimensional shape parameterisation.
    weight = np.exp(-(((vertices[:, 1] - body["section_y"]) / 0.10) ** 2))
    weight *= np.exp(-((np.abs(radial[:, 0]) / 0.25) ** 8))

    def altered(scale):
        out = vertices.copy()
        out[:, [0, 2]] += radial * (weight * scale)[:, None]
        return out

    lo, hi = -0.5, 0.5
    for _ in range(25):
        middle = (lo + hi) / 2
        if perimeter(altered(middle)) < reference + delta:
            lo = middle
        else:
            hi = middle
    out = altered((lo + hi) / 2) if delta else vertices.copy()
    if variant in {"cloth", "asymmetry"}:
        azimuth = np.arctan2(radial[:, 1], radial[:, 0])
        wave = (
            0.012 * np.cos(azimuth * 7)
            if variant == "cloth"
            else 0.02 * (radial[:, 0] > 0)
        )
        length = np.maximum(np.linalg.norm(radial, axis=1), 0.001)
        out[:, [0, 2]] += radial / length[:, None] * (wave * weight)[:, None]
    return out, reference, perimeter(out)


def render_frame(vertices, faces, body: dict, spec: dict, index: int, count: int):
    import cv2
    import numpy as np

    h, w, focal = 640, 640, 430.0
    variant = spec.get("nuisance", "plain")
    rng = np.random.default_rng(spec["seed"] + index)
    clutter = variant in {"mild", "clutter", "combined", "backlight"}
    image = np.full((h, w, 3), (175, 180, 183), np.uint8)
    if clutter:
        cv2.rectangle(image, (15, 60), (135, 590), (90, 110, 130), -1)
        cv2.rectangle(image, (495, 370), (639, 570), (80, 105, 90), -1)
        for x in range(150, 640, 90):
            cv2.line(image, (x, 0), (x, 640), (160, 164, 169), 2)
    cv2.rectangle(image, (0, 565), (639, 639), (100, 105, 111), -1)
    coverage = math.pi / 2 if variant == "partial_rotation" else 2 * math.pi
    angle = index / count * coverage + (spec["seed"] % 11) * 0.025
    rot = np.array(
        [
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ]
    )
    v = vertices.copy()
    if variant in {"breathing", "combined"}:
        weight = np.exp(-(((v[:, 1] - body["section_y"]) / 0.12) ** 2))
        v[:, [0, 2]] *= (
            1 + (0.018 * math.sin(index * 2 * math.pi / 13) * weight)[:, None]
        )
    v = v @ rot.T
    y_center = float((vertices[:, 1].max() + vertices[:, 1].min()) / 2)
    positions = v * [1, -1, -1] + [0, y_center, spec["distance"]]
    if variant in {"camera_jitter", "combined"}:
        positions += [
            0.018 * math.sin(index),
            0.012 * math.cos(index * 1.7),
            0.025 * math.sin(index * 0.7),
        ]
    projected = positions[:, :2] / positions[:, 2:] * focal + [w / 2, h / 2]
    triangles = positions[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-9)
    light = np.array([-0.4, -0.5, -1.0])
    light /= np.linalg.norm(light)
    shade = np.clip(0.55 + 0.45 * np.abs(normals @ light), 0.25, 1)
    face_y = vertices[faces, 1].mean(axis=1)
    colours = np.full((len(faces), 3), (111, 161, 207), float)
    colours[(face_y > body["hip_y"]) & (face_y < body["shoulder_y"] + 0.06)] = [
        145,
        85,
        48,
    ]
    colours[face_y < body["hip_y"]] = [60, 65, 83]
    colours = np.clip(colours * shade[:, None], 0, 255).astype(np.uint8)
    renderer = spec.get("renderer", "legacy_integer.v1")
    if renderer == "continuous_coverage8.v1":
        from .coverage_renderer import paint_person

        image, mask = paint_person(
            image,
            projected,
            faces,
            colours,
            np.argsort(-triangles[:, :, 2].mean(axis=1)),
            scale=8,
        )
    elif renderer == "legacy_integer.v1":
        polygons = projected[faces].astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        for j in np.argsort(-triangles[:, :, 2].mean(axis=1)):
            cv2.fillConvexPoly(image, polygons[j], tuple(map(int, colours[j])))
            cv2.fillConvexPoly(mask, polygons[j], 255)
    else:
        raise InputError("Unknown synthetic renderer")
    if variant == "two_people":
        # Same rendered person twice, equal mask area: deliberately ambiguous.
        small = cv2.resize(image, (320, 640))
        image = np.concatenate([small, small], axis=1)
    if variant in {"partial_occlusion", "large_occlusion"}:
        top = 370 if variant == "partial_occlusion" else 180
        cv2.rectangle(image, (225, top), (415, 570), (105, 114, 112), -1)
    if variant == "cropped_head":
        image = cv2.warpAffine(
            image,
            np.float32([[1, 0, 0], [0, 1, -220]]),
            (w, h),
            borderValue=(175, 180, 183),
        )
    if variant == "blank":
        image[:] = 160
    if variant in {"dim", "backlight", "flicker", "combined"}:
        yy, xx = np.mgrid[:h, :w]
        lighting = 0.18 + 0.5 * xx / w
        if variant == "backlight":
            lighting = np.ones((h, w))
            lighting[mask > 0] = 0.22
        if variant == "flicker":
            lighting = np.full((h, w), 0.25 + 0.75 * ((index % 4) < 2))
        image = np.clip(image.astype(float) * lighting[:, :, None], 0, 255).astype(
            np.uint8
        )
    if variant in {"blur", "combined"}:
        image = cv2.GaussianBlur(image, (15, 15), 4)
    if variant in {"noise", "combined"}:
        image = np.clip(
            image.astype(float) + rng.normal(0, 18, image.shape), 0, 255
        ).astype(np.uint8)
    if variant in {"compression", "combined"}:
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 15])
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return image


def render_case(root: Path, spec: dict) -> str:
    import cv2
    import numpy as np

    body = read_json(root / "body.json")
    with np.load(root / "body.npz") as mesh:
        vertices, faces = mesh["vertices"], mesh["faces"]
    body["faces"] = faces
    v, before, after = deform(
        vertices, body, spec.get("delta", 0), spec.get("nuisance", "plain")
    )
    identifier = uuid.uuid4().hex
    path = new_run(root / "store" / identifier)
    video = path / "capture.bin"
    # Container detection is by content, so render with an AVI extension then
    # rename to the exact capture.bin contract used by actual recordings.
    temporary = path / "render.avi"
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), 2.4, (640, 640)
    )
    if not writer.isOpened():
        raise InputError("Synthetic video codec unavailable")
    started = time.monotonic()
    try:
        for i in range(48):
            frame = render_frame(v, faces, body, spec, i, 48)
            writer.write(frame)
            if i in (0, 12, 24, 36):
                cv2.imwrite(str(path / f"preview_{i}.jpg"), frame)
                (path / f"preview_{i}.jpg").chmod(0o600)
    finally:
        writer.release()
    temporary.rename(video)
    video.chmod(0o600)
    write_json(
        path / "truth.json",
        {
            "source_kind": "synthetic",
            "spec": spec,
            "baseline_perimeter_model_m": before,
            "input_perimeter_model_m": after,
            "actual_delta_model_m": after - before,
            "section_y": body["section_y"],
            "center_xz": body["center_xz"],
            "render_elapsed_s": round(time.monotonic() - started, 2),
            "camera_focal_px": 430,
            "frame_count": 48,
            "fps": 2.4,
            "physical_accuracy_validated": False,
        },
    )
    write_json(
        path / "session.json",
        {
            "source_kind": "synthetic",
            "label": spec["name"],
            "created_at": time.time(),
            "video_sha256": sha256(video),
            "state": "recorded_not_reconstructed",
        },
    )
    return identifier


def run_matrix(config_path: str, output: str, names: list[str] | None = None) -> dict:
    import fcntl

    root = Path(output).resolve()
    lock_path = root / "matrix.lock"
    with lock_path.open("a") as lock:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise InputError("Another orchestrator is using this experiment") from exc
        return _run_matrix(config_path, output, names)


def _run_matrix(config_path: str, output: str, names: list[str] | None = None) -> dict:
    root = Path(output).resolve()
    config = read_json(config_path)
    manifest = read_json(root / "matrix.json")
    if manifest.get("research_options") is not None:
        from .research import options

        if (
            options(read_json(root / "research-options.json"))
            != manifest["research_options"]
        ):
            raise InputError("Experiment research options changed")
    if manifest["engine_id"] != config["engine_id"]:
        raise InputError("Experiment backend changed")
    known = {case["name"] for case in manifest["cases"]}
    if names and not set(names).issubset(known):
        raise InputError("Unknown experiment case")
    pending = [
        case
        for case in manifest["cases"]
        if case["state"] not in {"complete", "failed"}
        and (not names or case["name"] in names)
    ]
    snapshot = root / "implementation" / "hashes.json"
    if pending and snapshot.exists():
        for name, digest in read_json(snapshot).items():
            if sha256(Path(__file__).with_name(name)) != digest:
                raise InputError(
                    "Experiment implementation changed; use a separate experiment"
                )
    for case in manifest["cases"]:
        if names and case["name"] not in names:
            continue
        if case.get("state") in {"complete", "failed"}:
            continue
        if not case.get("session_id"):
            case["session_id"] = render_case(root, case["spec"])
            case["state"] = "rendered"
            replace_json(root / "matrix.json", manifest)
        path = root / "store" / case["session_id"]
        # A parent interrupted after worker completion must not reprocess the
        # immutable successful mesh, or erase a terminal failed attempt.
        previous = (
            read_json(path / "reconstruction.json")
            if (path / "reconstruction.json").exists()
            else None
        )
        if previous and previous["state"] in {"complete", "failed"}:
            case.update(state=previous["state"], reconstruction=previous)
            replace_json(root / "matrix.json", manifest)
            continue
        case["state"] = "running"
        replace_json(root / "matrix.json", manifest)
        print("Processing " + case["name"], flush=True)
        code = launch(
            config_path,
            [
                "-u",
                "-m",
                "body_scan.pipeline",
                "--config",
                str(Path(config_path).resolve()),
                "--store",
                str(root / "store"),
                "--session",
                case["session_id"],
                "--allow-synthetic",
                *(
                    ["--research-options", str(root / "research-options.json")]
                    if manifest.get("research_options") is not None
                    else []
                ),
            ],
            path / ("worker_" + uuid.uuid4().hex + ".log"),
        )
        status = (
            read_json(path / "reconstruction.json")
            if (path / "reconstruction.json").exists()
            else {"state": "failed", "error": "worker ended without status"}
        )
        if status.get("state") not in {"complete", "failed"}:
            status = {
                "state": "failed",
                "stage": "interrupted",
                "error": "Worker exited without a terminal reconstruction state",
                "exit_code": code,
            }
            replace_json(path / "reconstruction.json", status)
        case.update(state=status["state"], exit_code=code, reconstruction=status)
        replace_json(root / "matrix.json", manifest)
        print(case["name"] + ": " + status["state"], flush=True)
        report(root)
    return report(root)


def report(root: Path) -> dict:
    from .mesh import section_perimeter

    manifest = read_json(root / "matrix.json")
    results = []
    for row in manifest["cases"]:
        entry = {
            k: row.get(k)
            for k in ("name", "state", "session_id", "exit_code", "reconstruction")
        }
        entry["spec"] = row["spec"]
        if row.get("session_id"):
            path = root / "store" / row["session_id"]
            entry["truth"] = read_json(path / "truth.json")
            entry["video_sha256"] = read_json(path / "session.json")["video_sha256"]
            if (path / "mesh.json").exists():
                mesh = read_json(path / "mesh.json")
                entry["estimated_perimeter_model_m"] = section_perimeter(
                    mesh, mesh["sections"][0]
                )
                entry["quality"] = mesh["quality"]
                entry["fit_implementation_sha256"] = [
                    read_json(fit)["implementation_sha256"]
                    for fit in path.glob("attempt_*/fit.json")
                ]
        results.append(entry)
    successful = [
        r
        for r in results
        if r["state"] == "complete" and r.get("estimated_perimeter_model_m") is not None
    ]
    baseline = [r for r in successful if r["spec"]["group"] == "baseline"]
    nulls = [
        abs(a["estimated_perimeter_model_m"] - b["estimated_perimeter_model_m"])
        for a, b in itertools.combinations(baseline, 2)
    ]
    changes = []
    for changed in successful:
        if changed["spec"]["group"] != "change":
            continue
        for base in baseline:
            prediction = (
                changed["estimated_perimeter_model_m"]
                - base["estimated_perimeter_model_m"]
            )
            truth = changed["truth"]["actual_delta_model_m"]
            changes.append(
                {
                    "before": base["name"],
                    "after": changed["name"],
                    "truth_delta_model_m": truth,
                    "predicted_delta_model_m": prediction,
                    "sign_correct": prediction * truth > 0,
                    "absolute_delta_error_model_m": abs(prediction - truth),
                    "constant_shape_negative_control_sign_correct": False,
                    "exceeds_3x_max_null": abs(prediction) >= 3 * max(nulls)
                    if nulls
                    else None,
                }
            )
    distances = [r for r in successful if r["spec"]["group"] == "distance"]
    distance_pairs = [
        {
            "before": a["name"],
            "after": b["name"],
            "symmetric_percent": 200
            * abs(a["estimated_perimeter_model_m"] - b["estimated_perimeter_model_m"])
            / (a["estimated_perimeter_model_m"] + b["estimated_perimeter_model_m"]),
        }
        for a, b in itertools.combinations(distances, 2)
    ]
    result = {
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "real_video_pipeline_verified": False,
        "engine_id": manifest["engine_id"],
        "analysis_current_source_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "synthetic_lab.py",
                "pipeline.py",
                "contours.py",
                "shared_shape.py",
                "fit_strategy.py",
                "quality.py",
            )
        },
        "generation_snapshot_sha256": (
            read_json(root / "implementation" / "hashes.json")
            if (root / "implementation" / "hashes.json").exists()
            else None
        ),
        "case_count": len(results),
        "unique_video_count": len(
            {r["video_sha256"] for r in results if "video_sha256" in r}
        ),
        "duplicate_video_groups": [
            [r["name"] for r in results if r.get("video_sha256") == digest]
            for digest in sorted(
                {r["video_sha256"] for r in results if "video_sha256" in r}
            )
            if sum(r.get("video_sha256") == digest for r in results) > 1
        ],
        "completed": len(successful),
        "failed": sum(r["state"] == "failed" for r in results),
        "pending": sum(r["state"] not in {"complete", "failed"} for r in results),
        "baseline_max_null_model_m": max(nulls) if nulls else None,
        "change_pairs": changes,
        "distance_pairs": distance_pairs,
        "results": results,
        "limitations": read_json(root / "body.json")["limitations"]
        + [
            "Geometry oracle is not supplied to detection, pose estimation or fitting",
            "Only one generated body; stress results do not establish human reliability",
            "Renderer truth and estimator canonical section may differ; no absolute calibration",
            "Failed and rejected cases remain in denominator; no automatic acceptance gate",
        ],
    }
    replace_json(root / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "worker-export", "run", "report"))
    parser.add_argument("--runtime-config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--renderer",
        choices=("legacy_integer.v1", "continuous_coverage8.v1"),
        help="Freeze the renderer when initializing a separate experiment",
    )
    args = parser.parse_args()
    if args.renderer is not None and args.mode != "init":
        parser.error("--renderer is only valid for init; run uses its frozen matrix")
    root = Path(args.out).resolve()
    if args.mode != "report" and not args.runtime_config:
        parser.error("--runtime-config is required")
    if args.mode == "worker-export":
        export_worker(args.runtime_config, args.out)
    elif args.mode == "init":
        planned = cases()
        if args.cases:
            if not set(args.cases).issubset({c["name"] for c in planned}):
                raise InputError("Unknown experiment case")
            planned = [c for c in planned if c["name"] in args.cases]
        if args.renderer is not None:
            planned = [{**case, "renderer": args.renderer} for case in planned]
        root = new_run(root)
        snapshot_sources(root, args.runtime_config)
        saved_protocol = protocol()
        if args.renderer is not None:
            saved_protocol["renderer"] = args.renderer
        write_json(root / "protocol.json", saved_protocol)
        new_run(root / "store")
        write_json(
            root / "store" / "store.json",
            {"schema_version": "private_scan_store.v1", "synthetic_experiment": True},
        )
        code = launch(
            args.runtime_config,
            [
                "-m",
                "body_scan.synthetic_lab",
                "worker-export",
                "--runtime-config",
                args.runtime_config,
                "--out",
                str(root),
            ],
            root / "export.log",
        )
        if code:
            raise InputError(
                f"Synthetic body export failed ({code}); see private export.log"
            )
        config = read_json(args.runtime_config)
        write_json(
            root / "matrix.json",
            {
                "source_kind": "synthetic",
                "engine_id": config["engine_id"],
                "cases": [
                    {"name": c["name"], "spec": c, "state": "pending"} for c in planned
                ],
            },
        )
        print("Synthetic experiment initialised", flush=True)
    elif args.mode == "run":
        run_matrix(args.runtime_config, args.out, args.cases)
    else:
        print(report(root)["completed"])


if __name__ == "__main__":
    main()
