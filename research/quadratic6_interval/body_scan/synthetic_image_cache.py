"""Cache observed synthetic RGB boundaries and frozen SAM estimates, without 3D fitting."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(
    config_path, source, output, names, contour_samples=256, *, capture_inputs=None
):
    import cv2
    import psutil
    import torch
    from ultralytics import YOLO

    from .capture_source import from_capture_inputs
    from .contours import _outer_contour
    from .mask_refinement import refine_person_mask
    from .quality import image_detail, select_primary_person
    from .synthetic_lab import snapshot_sources

    if contour_samples not in (256, 512, 1024):
        raise InputError("Invalid observed contour sample count")
    cfg = read_json(config_path)
    source = Path(source).resolve()
    context = (
        from_capture_inputs(capture_inputs, config_path) if capture_inputs else None
    )
    if context:
        source = context.store
        store = context.store
        profile = context.profile
        rows = [r for r in context.rows.values() if r["name"] in names]
        provenance = context.metadata
    else:
        report = read_json(source / "report.json")
        store = source / "store"
        profile = read_json(store / "profile.json")
        if (
            report.get("source_kind") != "synthetic"
            or profile.get("source_kind") != "synthetic"
            or read_json(store / "store.json").get("synthetic_experiment") is not True
            or report["engine_id"] != cfg["engine_id"]
        ):
            raise InputError(
                "Image cache requires synthetic SAM estimates from the declared runtime"
            )
        rows = [r for r in report["results"] if r["name"] in names]
        provenance = {
            "source_report_sha256": sha256(source / "report.json"),
            "profile_sha256": sha256(store / "profile.json"),
        }
    if (
        not rows
        or set(names) != {r["name"] for r in rows}
        or any(r["state"] != "complete" for r in rows)
        or not (
            context.is_reference(rows[0])
            if context
            else rows[0]["spec"]["group"] == "baseline"
        )
    ):
        raise InputError("Image cache requires completed captures and a baseline first")
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > cfg["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available < cfg["min_available_mb"] * 1024**2
            ):
                os._exit(75)
            time.sleep(0.25)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if sha256(Path(cfg["segmentation_model"])) != cfg["segmentation_model_sha256"]:
        raise InputError("Person detector changed")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "1"
    root = new_run(output)
    snapshot_sources(root, config_path)
    os.environ["YOLO_CONFIG_DIR"] = str(root / "upstream-config")
    detector = YOLO(cfg["segmentation_model"])
    m = {
        "source_kind": profile["source_kind"],
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
        "source": str(source),
        **provenance,
        "observation_variant": "observed_rgb_grabcut.v1",
        "cache_role": "observed_images_and_SAM_only",
        "source_pipeline_metrics_fixed_version_verified": False,
        "cases": [],
    }
    for row in rows:
        session = store / row["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Frozen capture changed")
        attempts = list(session.glob("attempt_*/frames.json"))
        if len(attempts) != 1:
            raise InputError("Expected one source inference")
        frame_rows = {
            r["frame"]: r
            for r in read_json(attempts[0])["frames"]
            if r["state"] == "estimated"
        }
        estimates = []
        targets = []
        legacy_targets = []
        mask_hashes = {}
        heights = []
        centers = []
        quality = []
        hashes = {}
        capture = cv2.VideoCapture(str(session / "capture.bin"))
        try:
            for frame in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
                ok, bgr = capture.read()
                if not ok:
                    raise InputError("Capture decoding failed")
                if frame not in frame_rows:
                    continue
                path = attempts[0].parent / frame_rows[frame]["output"]
                hashes[str(frame)] = sha256(path)
                with np.load(path, allow_pickle=False) as n:
                    value = {k: n[k] for k in n.files}
                prediction = detector.predict(
                    bgr,
                    classes=[0],
                    device="cpu",
                    retina_masks=True,
                    verbose=False,
                    save=False,
                )[0]
                if prediction.masks is None:
                    raise InputError(
                        "Frozen good image no longer has a detected person"
                    )
                masks = prediction.masks.data.cpu().numpy()
                primary = select_primary_person(masks.sum(axis=(1, 2)))
                mask = masks[primary] > 0.5
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask, _ = refine_person_mask(bgr, mask)
                boundary = _outer_contour(
                    (mask != 0).astype(np.uint8), bgr.shape[1], bgr.shape[0]
                )
                legacy_targets.append(
                    boundary[np.linspace(0, len(boundary) - 1, 256).astype(int)]
                )
                mask_hashes[str(frame)] = hashlib.sha256(mask.tobytes()).hexdigest()
                targets.append(
                    boundary[
                        np.linspace(0, len(boundary) - 1, contour_samples).astype(int)
                    ]
                )
                heights.append(float(np.ptp(np.where(mask)[0]) + 1))
                centers.append([bgr.shape[1] / 2, bgr.shape[0] / 2])
                quality.append(image_detail(bgr, mask)["edge_energy"])
                estimates.append(value)
        finally:
            capture.release()
        frames = np.array(sorted(frame_rows))
        shapes = np.array([v["shape_params"] for v in estimates])
        mesh = read_json(session / "mesh.json")
        data = {
            "frames": frames,
            "shapes": shapes,
            "initial": np.median(shapes, axis=0).astype(np.float32),
            "shape_scale": np.maximum(shapes.std(0), 0.1),
            "yaw": np.array([frame_rows[int(f)]["yaw_rad"] for f in frames]),
            "faces": np.asarray(mesh["faces"], np.int64),
            "section_y": np.array(profile["sections"][0]["y"]),
            "section_center": np.array(profile["sections"][0]["center_xz"]),
            "quality": np.array(quality),
            "shared_source_order": np.arange(len(frames)),
            "shared_target": np.asarray(targets, np.float32),
            "legacy_target256": np.asarray(legacy_targets, np.float32),
            "shared_mask_height": np.asarray(heights, np.float32),
            "shared_center": np.asarray(centers, np.float32),
        }
        file = root / (row["name"] + ".npz")
        np.savez_compressed(file, **data)
        file.chmod(0o600)
        m["cases"].append(
            {
                "name": row["name"],
                "cache": file.name,
                "cache_sha256": sha256(file),
                "video_sha256": row["video_sha256"],
                "frame_estimate_sha256": hashes,
                "frame_person_mask_sha256": mask_hashes,
                "observed_contour_samples": contour_samples,
                "estimated_mesh_topology_source_sha256": sha256(session / "mesh.json"),
            }
        )
        replace_json(root / "cache-manifest.json", m)
        print("observed_cached", row["name"], len(frames), flush=True)
    return m


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--contour-samples", type=int, default=256)
    a = p.parse_args()
    cache(a.runtime_config, a.source, a.out, a.cases, a.contour_samples)
