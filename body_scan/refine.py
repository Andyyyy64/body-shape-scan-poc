"""Offline refinement of a saved recording; writes a new UI-readable session."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

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
from .shared_shape import fit_observations


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
        from scipy.spatial import cKDTree

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
                "label": "輪郭フィット（試験）",
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
        rows = [
            r for r in read_json(manifests[0])["frames"] if r["state"] == "estimated"
        ]
        if not 3 <= len(rows) <= 32:
            raise InputError("Refinement needs 3–32 saved frame estimates")
        values = []
        for row in rows:
            with np.load(
                relative_asset(manifests[0].parent, row["output"]), allow_pickle=False
            ) as value:
                values.append({key: value[key] for key in value.files})
        shapes = np.stack([v["shape_params"] for v in values])
        initial = np.median(shapes, axis=0).astype("float32")
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
        bases, bases_delta, targets, cameras, focals, centers = [], [], [], [], [], []
        for j, (row, value) in enumerate(zip(rows, values)):
            image = frames.pop(row["frame"])
            height, width = image.shape[:2]
            prediction = detector.predict(
                image,
                classes=[0],
                device="cpu",
                retina_masks=True,
                verbose=False,
                save=False,
            )[0]
            if prediction.masks is None or len(prediction.masks.data) != 1:
                raise InputError("Refinement needs one person per saved frame")
            mask = prediction.masks.data[0].numpy().astype("uint8")
            if mask.shape != (height, width):
                raise InputError("Mask coordinates differ from the image")
            posed = vertices(initial, value["mhr_model_params"])
            camera = value["pred_cam_t"]
            focal = float(value["focal_length"])
            position = posed + camera
            if (position[:, 2] <= 0.1).any():
                raise InputError("Invalid initial camera geometry")
            projected = position[:, :2] / position[:, 2:] * focal + [
                width / 2,
                height / 2,
            ]
            raster = np.zeros((height, width), np.uint8)
            for face in faces:
                cv2.fillConvexPoly(
                    raster, np.clip(projected[face], -10000, 10000).astype("int32"), 1
                )

            def contour(binary):
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                if not contours:
                    raise InputError("No silhouette boundary")
                points = max(contours, key=len)[:, 0, :]
                points = points[
                    (points[:, 0] > 2)
                    & (points[:, 0] < width - 3)
                    & (points[:, 1] > 2)
                    & (points[:, 1] < height - 3)
                ]
                if len(points) < 30:
                    raise InputError("Insufficient untruncated silhouette boundary")
                return points

            predicted, observed = contour(raster), contour(mask)
            ids = np.unique(cKDTree(projected).query(predicted)[1])
            ids = ids[np.linspace(0, len(ids) - 1, 256).astype(int)]
            observed = observed[np.linspace(0, len(observed) - 1, 256).astype(int)]
            base = posed[ids]
            basis = []
            for dim in range(len(initial)):
                changed = initial.copy()
                changed[dim] += 1
                basis.append(vertices(changed, value["mhr_model_params"])[ids] - base)
            basis = np.stack(basis, axis=-1)
            perturbation = np.linspace(-0.1, 0.1, len(initial)).astype("float32")
            residual = np.max(
                abs(
                    vertices(initial + perturbation, value["mhr_model_params"])[ids]
                    - (base + basis @ perturbation)
                )
            )
            if residual > 1e-4:
                raise InputError("MHR local shape basis failed the linearity check")
            bases.append(base)
            bases_delta.append(basis)
            targets.append(observed)
            cameras.append(camera)
            focals.append(focal)
            centers.append([width / 2, height / 2])
            progress("preparing_contours", completed=j + 1, total=len(rows))
        progress("fitting_shared_shape", completed=len(rows), total=len(rows))
        fit = fit_observations(
            np.array(bases),
            np.array(bases_delta),
            np.array(targets),
            np.array(cameras),
            np.array(focals),
            np.array(centers),
            np.maximum(np.std(shapes, axis=0), 0.1),
        )
        shape = initial + fit.pop("shape_delta")
        fit["camera_delta"] = fit["camera_delta"].tolist()
        params = np.zeros(204, np.float32)
        params[136:] = profile["skeleton_scales"]
        canonical = vertices(shape, params)
        canonical[:, [1, 2]] *= -1
        method = "fixed_pose_silhouette_fit.v1"
        result = {
            **original,
            "vertices": canonical.tolist(),
            "method": method,
            "producer": "SAM/MHR initialization + shared-shape silhouette fit",
            "physical_accuracy_validated": False,
            "quality": {
                **original["quality"],
                "warning": "Experimental fixed-pose contour fit; accuracy unvalidated",
            },
        }
        validate_mesh(result)
        write_json(
            destination / "fit.json",
            {
                **fit,
                "method": method,
                "source_mesh_sha256": sha256(source / "mesh.json"),
                "implementation_sha256": sha256(Path(__file__)),
                "optimizer_sha256": sha256(Path(__file__).with_name("shared_shape.py")),
                "engine_id": config["engine_id"],
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
