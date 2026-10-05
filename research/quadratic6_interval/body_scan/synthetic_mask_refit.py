"""Freeze RGB-refined observations while keeping earlier SAM estimates fixed."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(config_path, source_cache, output):
    import fcntl
    import threading

    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO

    from .contours import CONTOUR_SAMPLES, _outer_contour
    from .mask_refinement import refine_person_mask
    from .quality import select_primary_person
    from .synthetic_lab import snapshot_sources

    config = read_json(config_path)
    source_cache = Path(source_cache).resolve()
    manifest = read_json(source_cache / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or manifest.get("diagnostic_only") is not True
    ):
        raise InputError("RGB refit requires frozen synthetic observations")
    source = Path(manifest["source"])
    if (
        sha256(source / "report.json") != manifest["source_report_sha256"]
        or sha256(source / "store/profile.json") != manifest["profile_sha256"]
    ):
        raise InputError("Frozen reconstruction changed")
    profile = read_json(source / "store/profile.json")
    root = new_run(output)
    snapshot_sources(root, config_path)
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
            time.sleep(0.25)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    for key in ("mhr_model", "segmentation_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("Model changed")
    model = torch.jit.load(config["mhr_model"], map_location="cpu")
    detector = YOLO(config["segmentation_model"])
    source_rows = {r["name"]: r for r in read_json(source / "report.json")["results"]}
    result = {
        **manifest,
        "cases": [],
        "observation_variant": "observed_rgb_grabcut.v1",
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
    }
    for row in manifest["cases"]:
        original_file = source_cache / row["cache"]
        if (
            original_file.parent != source_cache
            or sha256(original_file) != row["cache_sha256"]
        ):
            raise InputError("Observation cache changed")
        with np.load(original_file, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files if not k.startswith("native_")}
        session = source / "store" / source_rows[row["name"]]["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Input video changed")
        attempts = list(session.glob("attempt_*/fit.json"))
        if len(attempts) != 1:
            raise InputError("Expected one source reconstruction")
        frames = {
            r["frame"]: r
            for r in read_json(attempts[0].parent / "frames.json")["frames"]
        }
        wanted = {int(index): order for order, index in enumerate(data["frames"])}
        capture = cv2.VideoCapture(str(session / "capture.bin"))
        diagnostics = []
        try:
            for frame_index in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok, image = capture.read()
                if not ok:
                    raise InputError("Input decode failed")
                if frame_index not in wanted:
                    continue
                order = wanted[frame_index]
                prediction = detector.predict(
                    image,
                    classes=[0],
                    device="cpu",
                    retina_masks=True,
                    verbose=False,
                    save=False,
                )[0]
                torch.set_num_threads(config["cpu_threads"])
                if prediction.masks is None:
                    raise InputError("Frozen source lost person detection")
                masks = prediction.masks.data
                mask = (
                    masks[
                        select_primary_person([float(m.sum()) for m in masks])
                    ].numpy()
                    > 0.5
                )
                refined, detail = refine_person_mask(image, mask)
                height, width = mask.shape
                target = _outer_contour(refined.astype("uint8"), width, height)
                target = target[
                    np.linspace(0, len(target) - 1, CONTOUR_SAMPLES).astype(int)
                ]
                data["shared_target"][order] = target
                data["shared_mask_height"][order] = np.ptp(np.where(refined)[0]) + 1
                if "shared_target_weight" in data:
                    estimate_file = attempts[0].parent / frames[frame_index]["output"]
                    if (
                        sha256(estimate_file)
                        != row["frame_estimate_sha256"][str(frame_index)]
                    ):
                        raise InputError("Frozen SAM estimate changed")
                    with np.load(estimate_file, allow_pickle=False) as n:
                        params = n["mhr_model_params"].copy()
                    params[136:] = profile["skeleton_scales"]
                    with torch.no_grad():
                        vertices, _ = model(
                            torch.tensor(data["initial"])[None],
                            torch.tensor(params)[None],
                            torch.zeros(1, 72),
                        )
                    vertices = vertices[0].numpy() / 100 * [1, -1, -1]
                    p = vertices + data["shared_camera"][order]
                    projection = (
                        p[:, :2] / p[:, 2:] * data["shared_focal"][order]
                        + data["shared_center"][order]
                    )
                    box = projection[np.asarray(profile["fit_vertex_indices"])]
                    low, high = box.min(0), box.max(0)
                    margin = np.maximum((high - low) * 0.05, 2)
                    inside = ((target >= low - margin) & (target <= high + margin)).all(
                        1
                    )
                    weight = float(data["shared_target_weight"][order].max())
                    data["shared_target_weight"][order] = np.where(inside, weight, 1)
                diagnostics.append({"frame": frame_index, **detail})
        finally:
            capture.release()
        if len(diagnostics) != len(wanted):
            raise InputError("Refined observations are incomplete")
        target_file = root / row["cache"]
        np.savez_compressed(target_file, **data)
        target_file.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(target_file),
                "source_cache_sha256": row["cache_sha256"],
                "refinement_diagnostics": diagnostics,
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print("refined_cache", row["name"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache(args.runtime_config, args.source, args.out)
