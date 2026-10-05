"""Freeze independent image landmarks beside unchanged synthetic SAM estimates."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(config_path, source_cache, output):
    import fcntl

    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO

    from .image_anchors import select_anchors
    from .synthetic_lab import snapshot_sources

    config = read_json(config_path)
    source_cache = Path(source_cache).resolve()
    manifest = read_json(source_cache / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or manifest.get("diagnostic_only") is not True
    ):
        raise InputError("Independent anchor cache requires synthetic observations")
    source = Path(manifest["source"])
    if (
        sha256(source / "report.json") != manifest["source_report_sha256"]
        or sha256(source / "store/profile.json") != manifest["profile_sha256"]
    ):
        raise InputError("Frozen reconstruction changed")
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
    modelpath = Path(config["pose_model"])
    if sha256(modelpath) != config["pose_model_sha256"]:
        raise InputError("Independent anatomy model changed")
    detector = YOLO(modelpath)
    if detector.task != "pose":
        raise InputError("Expected a pose model")
    root = new_run(output)
    snapshot_sources(root, config_path)
    options = manifest["observation_options"].copy()
    options["fit"] = {
        **options["fit"],
        "anchor_source": "yolo17",
        "anchor_model_sha256": config["pose_model_sha256"],
    }
    result = {
        **manifest,
        "cases": [],
        "observation_options": options,
        "anchor_observer_model_sha256": config["pose_model_sha256"],
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
    }
    rows = {r["name"]: r for r in read_json(source / "report.json")["results"]}
    for row in manifest["cases"]:
        original = source_cache / row["cache"]
        if original.parent != source_cache or sha256(original) != row["cache_sha256"]:
            raise InputError("Frozen observation cache changed")
        with np.load(original, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files}
        session = source / "store" / rows[row["name"]]["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Source capture changed")
        attempts = list(session.glob("attempt_*/frames.json"))
        if len(attempts) != 1:
            raise InputError("Expected one source reconstruction")
        frame_rows = {r["frame"]: r for r in read_json(attempts[0])["frames"]}
        wanted = {
            int(data["frames"][int(order)]): i
            for i, order in enumerate(data["shared_source_order"])
        }
        target = np.zeros((len(wanted), 4, 2), np.float32)
        confidence = np.zeros((len(wanted), 4), np.float32)
        capture = cv2.VideoCapture(str(session / "capture.bin"))
        seen = 0
        try:
            for frame in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok, image = capture.read()
                if not ok:
                    raise InputError("Capture decode failed")
                if frame not in wanted:
                    continue
                order = wanted[frame]
                estimate = attempts[0].parent / frame_rows[frame]["output"]
                if sha256(estimate) != row["frame_estimate_sha256"][str(frame)]:
                    raise InputError("SAM estimate changed")
                with np.load(estimate, allow_pickle=False) as n:
                    reference = n["pred_keypoints_2d"][[5, 6, 9, 10]]
                prediction = detector.predict(
                    image, classes=[0], device="cpu", verbose=False, save=False
                )[0]
                torch.set_num_threads(config["cpu_threads"])
                if prediction.keypoints is None:
                    raise InputError("Independent anatomy was not observed")
                mask = np.zeros(image.shape[:2], bool)
                lo = np.floor(data["shared_target"][order].min(0)).astype(int)
                hi = np.ceil(data["shared_target"][order].max(0)).astype(int) + 1
                mask[lo[1] : hi[1], lo[0] : hi[0]] = True
                target[order], confidence[order] = select_anchors(
                    prediction.keypoints.data.numpy(),
                    prediction.boxes.xyxy.numpy(),
                    mask,
                    reference,
                )
                seen += 1
        finally:
            capture.release()
        if seen != len(wanted):
            raise InputError("Independent observations incomplete")
        data["shared_anchor_target"] = target
        data["shared_anchor_weights"] = confidence
        dest = root / row["cache"]
        np.savez_compressed(dest, **data)
        dest.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(dest),
                "source_cache_sha256": row["cache_sha256"],
                "independent_confident_anchor_count": int((confidence > 0).sum()),
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print(
            "image_anchors",
            row["name"],
            "confident",
            (confidence > 0).sum(),
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache(args.runtime_config, args.source, args.out)
