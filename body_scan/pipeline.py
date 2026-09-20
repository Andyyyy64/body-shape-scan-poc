"""Local rotation video -> per-frame SAM estimates -> shared-shape contour fit.

Default reconstruction (``fixed_pose_silhouette_fit.v1``): sample frames,
select the primary person, estimate pose/shape per frame with SAM 3D Body,
exclude frames that violate the capture protocol, initialise one shared shape
from the per-frame median, then refine it against every frame's observed
silhouette with the per-frame poses held fixed. The canonical MHR mesh uses
the store's first-scan skeleton scales. This is not multi-view pose fitting
and carries no physical-unit calibration.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256, write_json
from .mesh import validate_mesh
from .quality import (
    CAPTURE_PROTOCOL_ID,
    FrameRejected,
    frame_protocol_violations,
    minimum_usable_frames,
    select_primary_person,
    summarize_outcomes,
)

METHOD = "fixed_pose_silhouette_fit.v1"
PRODUCER = (
    "SAM 3D Body + MHR; per-frame median initialisation + shared-shape silhouette fit"
)
REASON_TEXT = {
    "no_person": "人物未検出",
    "multiple_people": "複数の人物",
    "core_out_of_frame": "頭〜腰が画面外",
    "arm_not_lowered": "腕が下がっていない",
}


def run(config_path: str, store: str, identifier: str):
    from .web import session_path

    root = Path(store).resolve()
    session = session_path(root, identifier)
    config = read_json(config_path)
    status_file = session / "reconstruction.json"
    started = time.monotonic()

    def progress(stage, **extra):
        replace_json(
            status_file,
            {
                "state": "running",
                "stage": stage,
                "elapsed_s": round(time.monotonic() - started, 1),
                **extra,
            },
        )

    try:
        if (session / "mesh.json").exists():
            raise InputError("This recording already has an immutable reconstruction")
        meta = read_json(session / "session.json")
        if meta.get("source_kind") != "real" or not (session / "capture.bin").is_file():
            raise InputError("A saved real video is required")
        lock_path = Path(config_path).resolve().parent / "inference.lock"
        inference_lock = lock_path.open("a")
        lock_path.chmod(0o600)
        try:
            fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            inference_lock.close()
            raise InputError("別のSAM実行が同じランタイムを使用中です。")
        # Held for this worker's lifetime; process exit releases the OS lock.
        if sha256(session / "capture.bin") != meta["video_sha256"]:
            raise InputError(
                "Saved video changed after capture; its provenance no longer matches"
            )
        progress("runtime_start")
        import psutil

        peak = {"rss_bytes": 0}

        def guard():
            while True:
                rss = psutil.Process().memory_info().rss
                peak["rss_bytes"] = max(peak["rss_bytes"], rss)
                if (
                    rss > config["max_rss_mb"] * 1024**2
                    or psutil.virtual_memory().available
                    < config["min_available_mb"] * 1024**2
                ):
                    replace_json(
                        status_file,
                        {
                            "state": "failed",
                            "stage": "resource_guard",
                            "error": "Macのメモリ保護のため停止しました。原動画は保存されています。",
                        },
                    )
                    os._exit(75)
                time.sleep(1)

        threading.Thread(target=guard, daemon=True).start()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["MOMENTUM_ENABLED"] = "0"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        os.environ["YOLO_OFFLINE"] = "1"
        os.environ["BODY_SCAN_DINO_SOURCE"] = config["dino_source"]
        attempt = new_run(session / ("attempt_" + uuid.uuid4().hex))
        os.environ["YOLO_CONFIG_DIR"] = str(attempt / "upstream-config")
        import cv2
        import numpy as np
        import roma
        import torch
        from ultralytics import YOLO, settings

        from .contours import contour_observation
        from .shared_shape import fit_observations

        settings.update({"sync": False})

        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        cv2.setNumThreads(1)
        if config["encoder_device"] == "mps":
            if not torch.backends.mps.is_available():
                raise InputError(
                    "Configured MPS device is unavailable; no fallback was used"
                )
            torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
        elif config["encoder_device"] != "cpu":
            raise InputError("Unknown encoder device")
        progress("verifying_models")
        for key in ("checkpoint", "mhr_model", "segmentation_model"):
            if sha256(Path(config[key])) != config[key + "_sha256"]:
                raise InputError(
                    "A model file changed; regenerate the runtime configuration"
                )
        sys.path.insert(0, config["sam_source"])
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        progress("loading_sam")
        model, model_config = load_sam_3d_body(
            config["checkpoint"], device="cpu", mhr_path=config["mhr_model"]
        )
        if config["encoder_device"] == "mps":
            model.backbone.encoder.to(device="mps", dtype=torch.float16)
        for name in list(model._modules):
            if name.endswith("_hand"):
                delattr(model, name)
        estimator = SAM3DBodyEstimator(model, model_config)
        detector = YOLO(config["segmentation_model"])
        if detector.task != "segment" or detector.names.get(0) != "person":
            raise InputError("A COCO person segmentation model is required")
        from .vision import sampled_video_frames

        progress("reading_video")
        frames, source_frame_count = sampled_video_frames(
            str(session / "capture.bin"), config["sample_frames"]
        )
        indices = [index for index, _ in frames]
        estimates = []
        outcomes = []
        try:
            for i, (frame_index, bgr) in enumerate(frames):
                progress("estimating_frames", completed=i, total=len(indices))
                height, width = bgr.shape[:2]
                prediction = detector.predict(
                    bgr,
                    classes=[0],
                    device="cpu",
                    retina_masks=True,
                    verbose=False,
                    save=False,
                    project=str(attempt / "segmentation"),
                )[0]
                masks = [] if prediction.masks is None else list(prediction.masks.data)
                try:
                    primary = select_primary_person([float(m.sum()) for m in masks])
                except FrameRejected as rejected:
                    outcomes.append(
                        {
                            "frame": int(frame_index),
                            "state": "failed",
                            "reason": rejected.reason,
                        }
                    )
                    continue
                mask = masks[primary].cpu().numpy() > 0.5
                if mask.shape != (height, width):
                    raise InputError("Segmentation and RGB coordinates differ")
                yy, xx = np.where(mask)
                bbox = np.array(
                    [[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], np.float32
                )
                result = estimator.process_one_image(
                    img=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    bboxes=bbox,
                    masks=mask.astype("uint8")[None],
                    inference_type="body",
                )
                if len(result) != 1:
                    outcomes.append(
                        {
                            "frame": int(frame_index),
                            "state": "failed",
                            "reason": "sam_no_prediction",
                        }
                    )
                    continue
                value = {
                    k: np.asarray(v) for k, v in result[0].items() if v is not None
                }
                if any(
                    v.dtype.kind not in "biuf" or not np.isfinite(v).all()
                    for v in value.values()
                ):
                    raise InputError("SAM returned a nonfinite or unsupported output")
                name = f"frame_{int(frame_index):06d}.npz"
                np.savez_compressed(
                    attempt / name, **{k: v for k, v in value.items() if k != "mask"}
                )
                (attempt / name).chmod(0o600)
                rotation = roma.euler_to_rotmat(
                    "xyz", torch.from_numpy(value["global_rot"])
                ).numpy()
                yaw = math.atan2(float(rotation[0, 2]), float(rotation[2, 2])) % (
                    2 * math.pi
                )
                violations = frame_protocol_violations(
                    value["pred_keypoints_2d"], width, height
                )
                if violations:
                    outcomes.append(
                        {
                            "frame": int(frame_index),
                            "state": "excluded",
                            "reason": violations[0],
                            "reasons": violations,
                            "yaw_rad": yaw,
                            "output": name,
                        }
                    )
                    continue
                estimates.append((yaw, value, mask))
                outcomes.append(
                    {
                        "frame": int(frame_index),
                        "state": "estimated",
                        "yaw_rad": yaw,
                        "output": name,
                    }
                )
        finally:
            frames.clear()
        write_json(attempt / "frames.json", {"frames": outcomes})
        summary = summarize_outcomes(outcomes)
        required = minimum_usable_frames(len(indices))
        if len(estimates) < required:
            detail = "、".join(
                f"{REASON_TEXT.get(reason, reason)} {count}枚"
                for reason, count in summary["exclusion_reasons"].items()
            )
            raise InputError(
                f"使用できるフレームが{len(estimates)}/{len(indices)}枚で、必要な{required}枚に届きません"
                f"（{detail}）。脱力して腕を下げ、約2mで頭から腰まで画面に入れて撮り直してください。"
            )
        progress("fitting_contours", completed=0, total=len(estimates))
        shapes = np.stack([value["shape_params"] for _, value, _ in estimates])
        initial = np.median(shapes, axis=0).astype("float32")
        frame_scales = np.stack(
            [value["mhr_model_params"][136:] for _, value, _ in estimates]
        )
        if initial.shape != (45,) or frame_scales.shape[1:] != (68,):
            raise InputError("Unexpected SAM/MHR parameter dimensions")
        profile_path = root / "profile.json"
        if profile_path.exists():
            profile = read_json(profile_path)
            if profile["engine_id"] != config["engine_id"]:
                raise InputError(
                    "The model/backend changed; use a separate store or explicitly bridge versions"
                )
            scales = np.asarray(profile["skeleton_scales"], np.float32)
        else:
            scales = np.median(frame_scales, axis=0).astype("float32")

        def mhr_vertices(shape, params):
            with torch.no_grad():
                verts, skel = model.head_pose.mhr(
                    torch.as_tensor(shape, dtype=torch.float32)[None],
                    torch.as_tensor(params, dtype=torch.float32)[None],
                    torch.zeros(1, 72),
                )
            return verts[0].numpy() / 100.0, skel[0, :, :3].numpy() / 100.0

        faces = estimator.faces.astype(int)
        observations = []
        for j, (_, value, mask) in enumerate(estimates):

            def posed(shape, params=value["mhr_model_params"]):
                vertices = mhr_vertices(shape, params)[0]
                vertices[:, [1, 2]] *= -1  # MHR y-up/z-back -> camera y-down/z-forward
                return vertices

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
            progress("fitting_contours", completed=j + 1, total=len(estimates))
        progress("fitting_shared_shape", completed=len(estimates), total=len(estimates))
        fit = fit_observations(
            np.array([o["base"] for o in observations]),
            np.array([o["basis"] for o in observations]),
            np.array([o["target"] for o in observations]),
            np.array([o["camera"] for o in observations]),
            np.array([o["focal"] for o in observations]),
            np.array([o["center"] for o in observations]),
            np.maximum(np.std(shapes, axis=0), 0.1),
        )
        shape = (initial + fit.pop("shape_delta")).astype("float32")
        params = np.zeros(204, np.float32)
        params[136:] = scales
        vertices, skeleton = mhr_vertices(shape, params)
        mapping = model.head_pose.keypoint_mapping.detach().cpu().numpy()
        joints = mapping @ np.concatenate([vertices, skeleton])
        if not profile_path.exists():
            hip = (joints[9] + joints[10]) / 2
            shoulder = (joints[5] + joints[6]) / 2
            if shoulder[1] <= hip[1]:
                raise InputError("Canonical body axes are inconsistent")
            half_width = max(
                abs(joints[5, 0] - shoulder[0]), abs(joints[6, 0] - shoulder[0])
            )
            region = np.where(
                (vertices[:, 1] >= hip[1])
                & (vertices[:, 1] <= shoulder[1])
                & (np.abs(vertices[:, 0] - hip[0]) <= half_width)
            )[0].tolist()
            if len(region) < 100:
                raise InputError("Canonical torso region could not be defined")
            profile = {
                "id": uuid.uuid4().hex,
                "engine_id": config["engine_id"],
                "skeleton_scales": scales.tolist(),
                "sections": [
                    {
                        "id": "model_abdomen_torso32.v1",
                        "y": float(hip[1] + 0.32 * (shoulder[1] - hip[1])),
                        "center_xz": [float(hip[0]), float(hip[2])],
                    }
                ],
                "comparison_vertex_indices": region,
                "source_capture_sha256": meta["video_sha256"],
            }

        angles = sorted(y for y, _, _ in estimates)
        gap = max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]))
        bins = {int(y / (2 * math.pi) * 8) % 8 for y in angles}
        complete = len(bins) >= 6 and gap <= math.pi / 2
        depth = float(
            np.median([float(value["pred_cam_t"][2]) for _, value, _ in estimates])
        )
        mesh = {
            "schema_version": "canonical_mesh.v1",
            "source_kind": "real",
            "units": "model_m",
            "reference_frame_id": "mhr_rest_" + profile["id"],
            "canonical_pose_id": "mhr_zero_pose_136.v1",
            "scale_reference_id": "model_skeleton_" + profile["id"],
            "producer": PRODUCER,
            "method": METHOD,
            "capture_sha256": meta["video_sha256"],
            "sections": profile["sections"],
            "comparison_vertex_indices": profile["comparison_vertex_indices"],
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "quality": {
                "rotation_confirmed": complete,
                "orientation_bins": len(bins),
                "max_yaw_gap_deg": math.degrees(gap),
                "usable_frames": len(estimates),
                "attempted_frames": len(indices),
                "camera_depth_median_model_m": depth,
                **summary,
                "warning": "Model-estimated dimensions; unobserved body parts may be imputed. Fixed per-frame poses; accuracy unvalidated.",
            },
            "physical_accuracy_validated": False,
        }
        validate_mesh(mesh)
        if not profile_path.exists():
            write_json(profile_path, profile)
        write_json(
            attempt / "fit.json",
            {
                **{
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in fit.items()
                },
                "method": METHOD,
                "capture_protocol_id": CAPTURE_PROTOCOL_ID,
                "initial_shape_params": initial.tolist(),
                "shape_params": shape.tolist(),
                "implementation_sha256": {
                    name: sha256(Path(__file__).with_name(name))
                    for name in (
                        "pipeline.py",
                        "contours.py",
                        "shared_shape.py",
                        "quality.py",
                    )
                },
                "engine_id": config["engine_id"],
            },
        )
        write_json(session / "mesh.json", mesh)
        replace_json(
            status_file,
            {
                "state": "complete",
                "stage": "complete",
                "elapsed_s": round(time.monotonic() - started, 1),
                "method": METHOD,
                "usable_frames": len(estimates),
                "total_frames": len(indices),
                "excluded_frames": summary["excluded_frames"],
                "exclusion_reasons": summary["exclusion_reasons"],
                "rotation_confirmed": complete,
                "peak_rss_mb": round(peak["rss_bytes"] / 1024**2),
                "engine_id": config["engine_id"],
            },
        )
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, InputError)
            else "SAMの処理に失敗しました。詳細は非公開の実行ログに保存しています。"
        )
        replace_json(
            status_file,
            {
                "state": "failed",
                "stage": "failed",
                "error": message,
                "elapsed_s": round(time.monotonic() - started, 1),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    run(args.config, args.store, args.session)
