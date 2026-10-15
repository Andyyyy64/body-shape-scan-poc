"""Offline SAM/MHR runtime check using explicitly synthetic pixels.

This tests real model/device execution, never physical measurement accuracy.
All artifacts stay outside Git, just like actual runtime jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .io import InputError, new_run, read_json, sha256, write_json
from .network_sandbox import sandbox_command, verify_network_sandbox


def smoke(config_path: str, output: str) -> dict:
    config = read_json(config_path)
    verify_network_sandbox(config["network_sandbox"], config["python"])
    root = new_run(output)
    command = sandbox_command(
        config["network_sandbox"],
        [
            config["python"],
            "-m",
            "body_scan.runtime_smoke",
            "--config",
            str(Path(config_path).resolve()),
            "--out",
            str(root),
        ],
    )
    with (root / "runtime.log").open("xb") as log:
        (root / "runtime.log").chmod(0o600)
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            },
        )
    if result.returncode:
        raise InputError(
            f"Offline model smoke failed (exit {result.returncode}); details are in the private output runtime.log"
        )
    return read_json(root / "report.json")


def run(config_path: str, output: str) -> None:
    import psutil

    config, root = read_json(config_path), Path(output)
    lock_path = Path(config_path).resolve().parent / "inference.lock"
    inference_lock = lock_path.open("a")
    lock_path.chmod(0o600)
    try:
        fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        inference_lock.close()
        raise InputError("Another inference job is using this runtime") from exc
    started = time.monotonic()
    peak = {"rss": 0}

    def guard():
        while True:
            rss = psutil.Process().memory_info().rss
            peak["rss"] = max(peak["rss"], rss)
            if (
                rss > config["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < config["min_available_mb"] * 1024**2
            ):
                write_json(
                    root / "resource_failure.json",
                    {
                        "status": "resource_guard",
                        "rss_mb": round(rss / 1024**2),
                        "max_rss_mb": config["max_rss_mb"],
                        "available_mb": round(
                            psutil.virtual_memory().available / 1024**2
                        ),
                        "min_available_mb": config["min_available_mb"],
                    },
                )
                os._exit(75)
            time.sleep(0.1)

    threading.Thread(target=guard, daemon=True).start()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MOMENTUM_ENABLED="0",
        YOLO_AUTOINSTALL="false",
        YOLO_OFFLINE="1",
        BODY_SCAN_DINO_SOURCE=config["dino_source"],
        YOLO_CONFIG_DIR=str(root / "upstream-config"),
    )
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO, settings

    from .devices import configure_device

    settings.update({"sync": False})
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    cv2.setNumThreads(1)
    device = configure_device(torch, config)
    for key in ("checkpoint", "mhr_model", "segmentation_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("Model checksum mismatch")
    sys.path.insert(0, config["sam_source"])
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, model_config = load_sam_3d_body(
        config["checkpoint"], device="cpu", mhr_path=config["mhr_model"]
    )
    if device in ("cuda", "mps"):
        model.backbone.encoder.to(device=device, dtype=torch.float16)
    for name in list(model._modules):
        if name.endswith("_hand"):
            delattr(model, name)
    estimator = SAM3DBodyEstimator(model, model_config)
    image = np.full((384, 256, 3), 230, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask, (128, 55), 25, 1, -1)
    cv2.rectangle(mask, (95, 85), (161, 265), 1, -1)
    cv2.rectangle(mask, (65, 85), (85, 265), 1, -1)
    cv2.rectangle(mask, (171, 85), (191, 265), 1, -1)
    cv2.rectangle(mask, (97, 265), (121, 360), 1, -1)
    cv2.rectangle(mask, (135, 265), (159, 360), 1, -1)
    image[mask.astype(bool)] = [50, 80, 120]
    detector = YOLO(config["segmentation_model"])
    if detector.task != "segment":
        raise InputError("Segmentation checkpoint is invalid")
    detector.predict(image, classes=[0], device="cpu", verbose=False, save=False)
    result = estimator.process_one_image(
        img=image,
        bboxes=np.array([[60, 20, 196, 365]], np.float32),
        masks=mask[None],
        inference_type="body",
    )
    if len(result) != 1:
        raise InputError("Synthetic SAM smoke returned no prediction")
    value = result[0]
    for item in value.values():
        if item is not None and not np.isfinite(np.asarray(item)).all():
            raise InputError("SAM smoke returned nonfinite values")
    params = torch.as_tensor(value["mhr_model_params"], dtype=torch.float32)
    shape = torch.as_tensor(value["shape_params"], dtype=torch.float32)
    with torch.no_grad():
        vertices, skeleton = model.head_pose.mhr(
            shape[None], params[None], torch.zeros(1, 72)
        )
    if not torch.isfinite(vertices).all() or not torch.isfinite(skeleton).all():
        raise InputError("MHR smoke returned nonfinite values")
    report = {
        "status": "offline_sam_mhr_smoke_passed",
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "full_video_pipeline_verified": False,
        "sample_frames": 1,
        "encoder_device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "elapsed_s": round(time.monotonic() - started, 2),
        "peak_rss_mb": round(peak["rss"] / 1024**2),
        "peak_cuda_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2)
        if device == "cuda"
        else None,
        "peak_cuda_reserved_mb": round(torch.cuda.max_memory_reserved(0) / 1024**2)
        if device == "cuda"
        else None,
        "mesh_vertex_count": int(vertices.shape[1]),
        "network_sandbox": config["network_sandbox"],
        "max_rss_mb": config["max_rss_mb"],
        "min_available_mb": config["min_available_mb"],
        "cuda_memory_fraction": config.get("cuda_memory_fraction")
        if device == "cuda"
        else None,
        "implementation_sha256": sha256(Path(__file__)),
        "engine_id": config["engine_id"],
    }
    write_json(root / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.config, args.out)
