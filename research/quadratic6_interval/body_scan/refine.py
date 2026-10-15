"""Offline contour refinement of a legacy ``sam_parameter_ensemble`` recording.

Writes a new UI-readable session using the same shared-shape silhouette fit
that the default pipeline now applies. Kept for re-processing recordings made
before the fit became the default; it does not modify the source session.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from .contours import contour_observation
from .io import (
    InputError,
    new_run,
    read_json,
    relative_asset,
    replace_json,
    sha256,
    write_json,
)
from .mesh import validate_mesh
from .quality import (
    CAPTURE_PROTOCOL_ID,
    FrameRejected,
    frame_protocol_violations,
    minimum_usable_frames,
    select_primary_person,
    summarize_outcomes,
)
from .shared_shape import fit_observations

METHOD = "fixed_pose_silhouette_fit.v1"


def run(config_path, store, source_id):
    from .web import session_path, store_root

    root = store_root(store)
    source = session_path(root, source_id)
    config = read_json(config_path)
    original = validate_mesh(read_json(source / "mesh.json"))
    meta = read_json(source / "session.json")
    if (
        original.get("method") != "sam_parameter_ensemble"
        or meta.get("source_kind") != "real"
    ):
        raise InputError("Refinement requires an original real SAM recording")
    if (
        read_json(source / "reconstruction.json").get("engine_id")
        != config["engine_id"]
    ):
        raise InputError("Source reconstruction and runtime engine differ")
    manifests = list(source.glob("attempt_*/frames.json"))
    if len(manifests) != 1:
        raise InputError("Exactly one successful inference manifest is required")
    if sha256(source / "capture.bin") != meta["video_sha256"]:
        raise InputError("Capture hash changed")
    for key in ("mhr_model", "segmentation_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("Runtime model hash changed")
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise InputError("Another model worker is active") from None
    destination = new_run(root / uuid.uuid4().hex)
    status = destination / "reconstruction.json"
    stop = threading.Event()
    started = time.monotonic()

    def progress(stage, **extra):
        replace_json(
            status,
            {
                "state": "running",
                "stage": stage,
                "elapsed_s": round(time.monotonic() - started, 1),
                **extra,
            },
        )

    try:
        import cv2
        import numpy as np
        import psutil
        import torch

        os.environ["YOLO_CONFIG_DIR"] = str(destination / "detector_settings")
        from ultralytics import YOLO

        def guard():
            while not stop.wait(1):
                if (
                    psutil.Process().memory_info().rss > config["max_rss_mb"] * 1024**2
                    or psutil.virtual_memory().available
                    < config["min_available_mb"] * 1024**2
                ):
                    replace_json(
                        status,
                        {
                            "state": "failed",
                            "error": "Mac memory guard stopped refinement",
                        },
                    )
                    os._exit(75)

        monitor = threading.Thread(target=guard, daemon=True)
        monitor.start()
        write_json(
            destination / "session.json",
            {
                **meta,
                "label": "輪郭フィット（再処理）",
                "derived_from": source_id,
                "state": "refining",
            },
        )
        shutil.copyfile(source / "capture.bin", destination / "capture.bin")
        (destination / "capture.bin").chmod(0o600)
        progress("preparing_contours")
        torch.set_num_threads(config["cpu_threads"])
        cv2.setNumThreads(1)
        model = torch.jit.load(config["mhr_model"], map_location="cpu").eval()
        detector = YOLO(config["segmentation_model"])
        manifest = read_json(manifests[0])["frames"]
        rows = [r for r in manifest if r["state"] == "estimated"]
        if not 3 <= len(rows) <= 32:
            raise InputError("Refinement needs 3–32 saved frame estimates")
        faces = np.asarray(original["faces"])
        profile = read_json(root / "profile.json")
        if original["scale_reference_id"] != "model_skeleton_" + profile["id"]:
            raise InputError("Source and stored skeleton profile differ")

        def vertices(shape, params):
            with torch.no_grad():
                vv, _ = model(
                    torch.tensor(shape, dtype=torch.float32)[None],
                    torch.tensor(params, dtype=torch.float32)[None],
                    torch.zeros(1, 72),
                )
            vv = vv[0].numpy() / 100
            vv[:, [1, 2]] *= -1
            return vv

        capture = cv2.VideoCapture(str(source / "capture.bin"))
        selected = {row["frame"]: i for i, row in enumerate(rows)}
        frames = {}
        try:
            for index in range(3600):
                ok, frame = capture.read()
                if not ok:
                    break
                if index in selected:
                    frames[index] = frame
        finally:
            capture.release()
        if set(frames) != set(selected):
            raise InputError("Saved frame indices could not be decoded")
        outcomes = [dict(r) for r in manifest if r["state"] != "estimated"]
        kept = []
        for row in rows:
            image = frames.pop(row["frame"])
            height, width = image.shape[:2]
            with np.load(
                relative_asset(manifests[0].parent, row["output"]), allow_pickle=False
            ) as value:
                value = {key: value[key] for key in value.files}
            prediction = detector.predict(
                image,
                classes=[0],
                device="cpu",
                retina_masks=True,
                verbose=False,
                save=False,
            )[0]
            masks = [] if prediction.masks is None else list(prediction.masks.data)
            try:
                primary = select_primary_person([float(m.sum()) for m in masks])
            except FrameRejected as rejected:
                outcomes.append({**row, "state": "failed", "reason": rejected.reason})
                continue
            mask = masks[primary].numpy().astype("uint8")
            if mask.shape != (height, width):
                raise InputError("Mask coordinates differ from the image")
            violations = frame_protocol_violations(
                value["pred_keypoints_2d"], width, height
            )
            if violations:
                outcomes.append(
                    {
                        **row,
                        "state": "excluded",
                        "reason": violations[0],
                        "reasons": violations,
                    }
                )
                continue
            kept.append((row, value, mask))
            outcomes.append(row)
        summary = summarize_outcomes(outcomes)
        required = minimum_usable_frames(len(manifest))
        if len(kept) < required:
            raise InputError(
                f"Only {len(kept)}/{len(manifest)} frames satisfy the capture protocol; {required} required"
            )
        shapes = np.stack([value["shape_params"] for _, value, _ in kept])
        initial = np.median(shapes, axis=0).astype("float32")
        observations = []
        for j, (row, value, mask) in enumerate(kept):

            def posed(shape, params=value["mhr_model_params"]):
                return vertices(shape, params)

            observations.append(
                contour_observation(
                    mask,
                    faces,
                    posed,
                    initial,
                    value["pred_cam_t"],
                    float(value["focal_length"]),
                )
            )
            progress("preparing_contours", completed=j + 1, total=len(kept))
        progress("fitting_shared_shape", completed=len(kept), total=len(kept))
        fit = fit_observations(
            np.array([o["base"] for o in observations]),
            np.array([o["basis"] for o in observations]),
            np.array([o["target"] for o in observations]),
            np.array([o["camera"] for o in observations]),
            np.array([o["focal"] for o in observations]),
            np.array([o["center"] for o in observations]),
            np.maximum(np.std(shapes, axis=0), 0.1),
        )
        shape = initial + fit.pop("shape_delta")
        fit["camera_delta"] = fit["camera_delta"].tolist()
        params = np.zeros(204, np.float32)
        params[136:] = profile["skeleton_scales"]
        canonical = vertices(shape, params)
        canonical[:, [1, 2]] *= -1
        result = {
            **original,
            "vertices": canonical.tolist(),
            "method": METHOD,
            "producer": "SAM/MHR initialization + shared-shape silhouette fit",
            "physical_accuracy_validated": False,
            "quality": {
                **original["quality"],
                "usable_frames": len(kept),
                **summary,
                "warning": "Fixed per-frame poses; accuracy unvalidated",
            },
        }
        validate_mesh(result)
        write_json(destination / "frames.json", {"frames": outcomes})
        write_json(
            destination / "fit.json",
            {
                **fit,
                "method": METHOD,
                "capture_protocol_id": CAPTURE_PROTOCOL_ID,
                "source_mesh_sha256": sha256(source / "mesh.json"),
                "implementation_sha256": {
                    name: sha256(Path(__file__).with_name(name))
                    for name in (
                        "refine.py",
                        "contours.py",
                        "shared_shape.py",
                        "quality.py",
                    )
                },
                "engine_id": config["engine_id"],
                "initial_shape_params": initial.tolist(),
                "shape_params": shape.tolist(),
            },
        )
        write_json(destination / "mesh.json", result)
        replace_json(
            status,
            {
                "state": "complete",
                "stage": "complete",
                "elapsed_s": round(time.monotonic() - started, 1),
                "method": METHOD,
                "usable_frames": len(kept),
                "total_frames": len(manifest),
                "excluded_frames": summary["excluded_frames"],
                "exclusion_reasons": summary["exclusion_reasons"],
            },
        )
        return destination
    except Exception as exc:
        replace_json(
            status,
            {
                "state": "failed",
                "error": str(exc)
                if isinstance(exc, InputError)
                else "Refinement failed; inspect the private runtime log",
            },
        )
        raise
    finally:
        stop.set()
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(run(args.config, args.store, args.source))
