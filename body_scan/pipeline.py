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
    "insufficient_image_detail": "ぼけ・画像細部不足",
    "insufficient_mask_refinement_seeds": "人物の輪郭情報不足",
    "excessive_mask_refinement": "人物輪郭が不安定",
    "independent_anatomy_unobserved": "画像上の関節を確認できません",
    "no_person": "人物未検出",
    "multiple_people": "複数の人物",
    "core_out_of_frame": "頭〜腰が画面外",
    "arm_not_lowered": "腕が下がっていない",
    "torso_not_observed": "胴体中央の人物マスク不足",
}


def validate_video_source(meta: dict, marker: dict, allow_synthetic: bool) -> str:
    """Synthetic videos require an explicit worker flag and a separate lab store."""
    kind = meta.get("source_kind")
    lab = marker.get("synthetic_experiment") is True
    if kind == "real" and not lab and not allow_synthetic:
        return kind
    if kind == "synthetic" and lab and allow_synthetic:
        return kind
    raise InputError("Video source and experiment store do not match the worker mode")


def run(
    config_path: str,
    store: str,
    identifier: str,
    allow_synthetic: bool = False,
    research_options: str | None = None,
    diagnostic_real: bool = False,
):
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
        source_kind = validate_video_source(
            meta, read_json(root / "store.json"), allow_synthetic
        )
        from .research import (
            fit_region,
            method_id,
            options,
            relative_fit_error,
            torso_mask_support,
        )

        if research_options and source_kind != "synthetic" and not diagnostic_real:
            raise InputError("Research variants require a synthetic experiment")
        if diagnostic_real and (source_kind != "real" or not research_options):
            raise InputError(
                "Diagnostic real-video mode requires real recordings and explicit research options"
            )
        research = options(read_json(research_options)) if research_options else {}
        if diagnostic_real and "camera" in research:
            raise InputError(
                "Synthetic renderer intrinsics cannot be used for a real recording"
            )
        method = method_id(research) if research_options else METHOD
        if not (session / "capture.bin").is_file():
            raise InputError("A saved video is required")
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
                            "error": "メモリ保護のため停止しました。原動画は保存されています。",
                            "peak_rss_mb": round(peak["rss_bytes"] / 1024**2),
                            "max_rss_mb": config["max_rss_mb"],
                            "rss_limit_exceeded": rss > config["max_rss_mb"] * 1024**2,
                            "min_available_mb": config["min_available_mb"],
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

        from .fit_strategy import build_observations, solve

        settings.update({"sync": False})

        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        cv2.setNumThreads(1)
        from .devices import configure_device

        encoder_device = configure_device(torch, config)
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
        if encoder_device in ("mps", "cuda"):
            model.backbone.encoder.to(device=encoder_device, dtype=torch.float16)
        for name in list(model._modules):
            if name.endswith("_hand"):
                delattr(model, name)
        estimator = SAM3DBodyEstimator(model, model_config)
        detector = YOLO(config["segmentation_model"])
        anchor_detector = None
        if research.get("fit", {}).get("anchor_source") == "yolo17":
            digest = research["fit"]["anchor_model_sha256"]
            if (
                not config.get("pose_model")
                or config.get("pose_model_sha256") != digest
                or sha256(Path(config["pose_model"])) != digest
            ):
                raise InputError(
                    "Independent anatomy model checksum/configuration mismatch"
                )
            anchor_detector = YOLO(config["pose_model"])
            if (
                anchor_detector.task != "pose"
                or anchor_detector.names.get(0) != "person"
            ):
                raise InputError("A COCO person pose model is required")
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
                cam_int = None
                if research.get("camera"):
                    camera = research["camera"]
                    if (width, height) != (camera["width"], camera["height"]):
                        raise InputError(
                            "Video dimensions do not match declared intrinsics"
                        )
                    cam_int = torch.tensor(
                        [
                            [
                                [camera["fx"], 0, camera["cx"]],
                                [0, camera["fy"], camera["cy"]],
                                [0, 0, 1],
                            ]
                        ],
                        dtype=torch.float32,
                    )
                prediction = detector.predict(
                    bgr,
                    classes=[0],
                    device="cpu",
                    retina_masks=True,
                    verbose=False,
                    save=False,
                    project=str(attempt / "segmentation"),
                )[0]
                # Restore the budget even when detection fails or is ambiguous.
                torch.set_num_threads(config["cpu_threads"])
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
                detail = {}
                detail_limit = research.get("qc", {}).get("min_image_detail")
                boundary_limit = research.get("qc", {}).get("min_boundary_detail")
                if (
                    detail_limit is not None
                    or boundary_limit is not None
                    or research.get("fit", {}).get("max_frames")
                ):
                    from .quality import image_detail

                    detail = image_detail(bgr, mask)
                    if (
                        detail_limit is not None
                        and detail["edge_energy"] < detail_limit
                    ):
                        outcomes.append(
                            {
                                "frame": int(frame_index),
                                "state": "excluded",
                                "reason": "insufficient_image_detail",
                                "image_detail": detail,
                            }
                        )
                        continue
                if boundary_limit is not None:
                    from .image_quality import boundary_detail

                    boundary = boundary_detail(bgr, mask)
                    detail["boundary"] = boundary
                    if boundary["boundary_detail"] < boundary_limit:
                        outcomes.append(
                            {
                                "frame": int(frame_index),
                                "state": "excluded",
                                "reason": "insufficient_image_detail",
                                "image_detail": detail,
                            }
                        )
                        continue
                yy, xx = np.where(mask)
                bbox = np.array(
                    [[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], np.float32
                )
                result = estimator.process_one_image(
                    img=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                    bboxes=bbox,
                    masks=mask.astype("uint8")[None],
                    inference_type="body",
                    **({"cam_int": cam_int} if cam_int is not None else {}),
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
                if detail:
                    value["image_detail_score"] = np.asarray(detail["edge_energy"])
                if anchor_detector is not None:
                    from .image_anchors import select_anchors

                    prediction = anchor_detector.predict(
                        bgr, classes=[0], device="cpu", verbose=False, save=False
                    )[0]
                    torch.set_num_threads(config["cpu_threads"])
                    try:
                        if prediction.keypoints is None:
                            raise FrameRejected("independent_anatomy_unobserved")
                        target, confidence = select_anchors(
                            prediction.keypoints.data.cpu().numpy(),
                            prediction.boxes.xyxy.cpu().numpy(),
                            mask,
                            value["pred_keypoints_2d"][[5, 6, 9, 10]],
                        )
                    except FrameRejected as rejected:
                        outcomes.append(
                            {
                                "frame": int(frame_index),
                                "state": "excluded",
                                "reason": rejected.reason,
                            }
                        )
                        continue
                    value["image_anchor_target"] = target
                    value["image_anchor_weights"] = confidence
                if (
                    research.get("fit", {}).get("heading_smoothness", 0)
                    or research.get("fit", {}).get("heading_model", "acceleration")
                    != "acceleration"
                ):
                    value["frame_index"] = np.asarray(frame_index, np.int64)
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
                refinement = {}
                if research.get("fit", {}).get("mask_refinement") == "grabcut":
                    from .mask_refinement import refine_person_mask

                    try:
                        mask, refinement = refine_person_mask(bgr, mask)
                    except FrameRejected as rejected:
                        outcomes.append(
                            {
                                "frame": int(frame_index),
                                "state": "excluded",
                                "reason": rejected.reason,
                                "output": name,
                            }
                        )
                        continue
                violations = frame_protocol_violations(
                    value["pred_keypoints_2d"], width, height
                )
                support = {}
                if research.get("qc", {}).get("mask_support"):
                    support = torso_mask_support(value["pred_keypoints_2d"], mask)
                    if not support["supported"]:
                        violations.append("torso_not_observed")
                if violations:
                    outcomes.append(
                        {
                            "frame": int(frame_index),
                            "state": "excluded",
                            "reason": violations[0],
                            "reasons": violations,
                            "yaw_rad": yaw,
                            "output": name,
                            "torso_mask_support": support,
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
                        "torso_mask_support": support,
                        "image_detail": detail,
                        **({"mask_refinement": refinement} if refinement else {}),
                    }
                )
        finally:
            frames.clear()
        write_json(attempt / "frames.json", {"frames": outcomes})
        summary = summarize_outcomes(outcomes)
        required = minimum_usable_frames(len(indices))
        write_json(
            attempt / "qc.json",
            {
                **summary,
                "usable_frames": len(estimates),
                "required_frames": required,
                "research_options": research,
                "method": method,
                "source_kind": source_kind,
                "torch_threads_at_qc": torch.get_num_threads(),
            },
        )
        if len(estimates) < required:
            detail = "、".join(
                f"{REASON_TEXT.get(reason, reason)} {count}枚"
                for reason, count in summary["exclusion_reasons"].items()
            )
            raise InputError(
                f"使用できるフレームが{len(estimates)}/{len(indices)}枚で、必要な{required}枚に届きません"
                f"（{detail}）。脱力して腕を下げ、約2mで頭から腰まで画面に入れて撮り直してください。"
            )
        angles = sorted(y for y, _, _ in estimates)
        gap = max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]))
        bins = {int(y / (2 * math.pi) * 8) % 8 for y in angles}
        complete = len(bins) >= 6 and gap <= math.pi / 2
        replace_json(
            attempt / "qc.json",
            {
                **summary,
                "usable_frames": len(estimates),
                "required_frames": required,
                "rotation_confirmed": complete,
                "orientation_bins": len(bins),
                "max_yaw_gap_deg": math.degrees(gap),
                "torch_threads_at_qc": torch.get_num_threads(),
                "research_options": research,
                "method": method,
                "source_kind": source_kind,
            },
        )
        if research.get("qc", {}).get("require_rotation") and not complete:
            raise InputError(
                "回転範囲が不足しています。全周が観測できるよう撮り直してください。"
            )
        progress("fitting_contours", completed=0, total=len(estimates))
        shapes = np.stack([value["shape_params"] for _, value, _ in estimates])
        initial = np.median(shapes, axis=0).astype("float32")
        pose_mode = research.get("fit", {}).get("pose_mode", "frame")
        root_reference = None
        if research.get("fit", {}).get("root_mode") == "profile_tilt":
            raise InputError(
                "Historical root convention; regenerate using profile_tilt_xyz"
            )
        if research.get("fit", {}).get("root_mode") == "profile_tilt_xyz":
            from .pose_constraints import root_tilt_reference

            root_reference = root_tilt_reference(
                [v["mhr_model_params"] for _, v, _ in estimates]
            )
        pose_reference = (
            np.median([v["mhr_model_params"][6:136] for _, v, _ in estimates], axis=0)
            if pose_mode != "frame"
            else None
        )
        frame_scales = np.stack(
            [value["mhr_model_params"][136:] for _, value, _ in estimates]
        )
        if initial.shape != (45,) or frame_scales.shape[1:] != (68,):
            raise InputError("Unexpected SAM/MHR parameter dimensions")
        profile_path = root / "profile.json"
        if profile_path.exists():
            profile = read_json(profile_path)
            if profile.get("source_kind", "real") != source_kind:
                raise InputError("A profile cannot bridge real and synthetic captures")
            if profile.get("method", METHOD) != method:
                raise InputError("Research method changed; use a separate store")
            if profile["engine_id"] != config["engine_id"]:
                raise InputError(
                    "The model/backend changed; use a separate store or explicitly bridge versions"
                )
            scales = np.asarray(profile["skeleton_scales"], np.float32)
            if root_reference is not None:
                if profile.get("root_tilt_reference") is None:
                    raise InputError("Profile has no shared root tilt")
                root_reference = np.asarray(profile["root_tilt_reference"], np.float32)
            if pose_mode == "profile":
                reference_pose = profile.get("pose_reference_params")
                if reference_pose is None:
                    raise InputError("Profile has no shared local pose")
                pose_reference = np.asarray(reference_pose, np.float32)
                if (
                    pose_reference.shape != (130,)
                    or not np.isfinite(pose_reference).all()
                ):
                    raise InputError("Invalid profile local pose")
            if research.get("fit", {}).get("initialization") == "profile":
                reference = profile.get("shape_initialization_params")
                if reference is None:
                    raise InputError("Profile has no shared shape initialization")
                initial = np.asarray(reference, np.float32)
                if initial.shape != (45,) or not np.isfinite(initial).all():
                    raise InputError("Invalid profile shape initialization")
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
        mapping = model.head_pose.keypoint_mapping.detach().cpu().numpy()
        fit_indices = None
        if research.get("region", "full") != "full":
            if profile_path.exists():
                fit_indices = profile["fit_vertex_indices"]
            else:
                rest_params = np.zeros(204, np.float32)
                rest_params[136:] = scales
                rest_vertices, rest_skeleton = mhr_vertices(initial, rest_params)
                rest_joints = mapping @ np.concatenate([rest_vertices, rest_skeleton])
                fit_indices = fit_region(rest_vertices, rest_joints, research["region"])

        def anatomy(shape, params):
            vertices, skeleton = mhr_vertices(shape, params)
            joints = (
                mapping[:11] @ np.concatenate([vertices, skeleton])
                if research.get("fit", {}).get("anchor_weight")
                else np.empty((11, 3))
            )
            return vertices, joints

        observations = build_observations(
            estimates,
            initial,
            scales,
            faces,
            anatomy,
            fit_indices,
            research,
            pose_reference=pose_reference,
            root_reference=root_reference,
        )
        progress("fitting_shared_shape", completed=len(estimates), total=len(estimates))
        shape_scale = np.maximum(np.std(shapes, axis=0), 0.1)
        fit = solve(observations, shape_scale, research)
        if research.get("fit", {}).get("refresh_contour"):
            first = fit
            selection = (initial + fit["shape_delta"][: len(initial)]).astype("float32")
            observations = build_observations(
                estimates,
                initial,
                scales,
                faces,
                anatomy,
                fit_indices,
                research,
                selection_shape=selection,
                selection_fit=first,
                pose_reference=pose_reference,
                root_reference=root_reference,
            )
            fit = solve(observations, shape_scale, research)
            fit["contour_refresh"] = {
                "first_final_loss_px": first["final_loss_px"],
                "first_shape_delta": first["shape_delta"].tolist(),
                "selection_only": True,
            }
        fitted_delta = fit.pop("shape_delta")
        shape = (initial + fitted_delta[: len(initial)]).astype("float32")
        local_coefficients = fitted_delta[len(initial) :]
        if len(local_coefficients):
            fit["local_shape_coefficients"] = local_coefficients.tolist()
            fit["local_shape_mode"] = research["fit"]["local_mode"]
            fit["local_shape_limit"] = research["fit"].get("local_limit", 0.1)
            fit["local_shape_at_bound"] = bool(
                np.max(abs(local_coefficients)) >= 0.98 * fit["local_shape_limit"]
            )
        relative_error = relative_fit_error(
            fit["final_unweighted_loss_px"],
            [o["mask_height"] for o in observations],
        )
        estimated_indices = [o["frame"] for o in outcomes if o["state"] == "estimated"]
        fit["relative_fit_error"] = relative_error
        write_json(
            attempt / "fit.json",
            {
                **{
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in fit.items()
                },
                "method": method,
                "research_options": research,
                "capture_protocol_id": CAPTURE_PROTOCOL_ID,
                "initial_shape_params": initial.tolist(),
                "shape_params": shape.tolist(),
                "fit_vertex_count": len(fit_indices)
                if fit_indices is not None
                else None,
                "fit_source_frame_indices": [
                    int(estimated_indices[o["source_order"]]) for o in observations
                ],
                **(
                    {"shape_scale": shape_scale.tolist()}
                    if research.get("fit", {}).get("diagnostics")
                    else {}
                ),
                "torch_threads_at_fit": torch.get_num_threads(),
                "implementation_sha256": {
                    name: sha256(Path(__file__).with_name(name))
                    for name in (
                        "pipeline.py",
                        "contours.py",
                        "silhouette_edges.py",
                        "shared_shape.py",
                        "fit_strategy.py",
                        "fit_diagnostics.py",
                        "contour_distance.py",
                        "mask_refinement.py",
                        "local_shape.py",
                        "pose_constraints.py",
                        "image_anchors.py",
                        "sampling.py",
                        "image_quality.py",
                        "quality.py",
                        "devices.py",
                        "network_sandbox.py",
                        "research.py",
                        "vision.py",
                        "mesh.py",
                        "geometry.py",
                    )
                },
                "engine_id": config["engine_id"],
                "source_kind": source_kind,
                "physical_accuracy_validated": False,
            },
        )
        limit = research.get("qc", {}).get("max_relative_fit_error")
        if limit is not None and relative_error > limit:
            raise InputError(
                "観測輪郭と推定形状の残差が大きすぎます。遮蔽や撮影条件を見直してください。"
            )
        params = np.zeros(204, np.float32)
        params[136:] = scales
        vertices, skeleton = mhr_vertices(shape, params)
        joints = mapping @ np.concatenate([vertices, skeleton])
        if len(local_coefficients):
            from .local_shape import torso_basis

            initial_vertices, initial_skeleton = mhr_vertices(initial, params)
            initial_joints = mapping[:11] @ np.concatenate(
                [initial_vertices, initial_skeleton]
            )
            vertices = (
                vertices
                + torso_basis(
                    initial_vertices,
                    initial_joints[[5, 6, 9, 10]],
                    research["fit"]["local_mode"],
                )
                @ local_coefficients
            )
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
                "source_kind": source_kind,
                "method": method,
                "engine_id": config["engine_id"],
                "skeleton_scales": scales.tolist(),
                **(
                    {"root_tilt_reference": root_reference.tolist()}
                    if root_reference is not None
                    else {}
                ),
                **(
                    {"pose_reference_params": pose_reference.tolist()}
                    if pose_mode == "profile"
                    else {}
                ),
                **(
                    {"shape_initialization_params": initial.tolist()}
                    if research.get("fit", {}).get("initialization") == "profile"
                    else {}
                ),
                "sections": [
                    {
                        "id": "model_abdomen_torso32.v1",
                        "y": float(hip[1] + 0.32 * (shoulder[1] - hip[1])),
                        "center_xz": [float(hip[0]), float(hip[2])],
                    }
                ],
                "comparison_vertex_indices": region,
                "source_capture_sha256": meta["video_sha256"],
                "fit_vertex_indices": fit_indices,
            }
        depth = float(
            np.median([float(value["pred_cam_t"][2]) for _, value, _ in estimates])
        )
        mesh = {
            "schema_version": "canonical_mesh.v1",
            "source_kind": source_kind,
            "units": "model_m",
            "reference_frame_id": "mhr_rest_" + profile["id"],
            "canonical_pose_id": "mhr_zero_pose_136.v1",
            "scale_reference_id": "model_skeleton_" + profile["id"],
            "producer": PRODUCER
            + (
                " + bounded anatomical torso residual"
                if len(local_coefficients)
                else ""
            ),
            "method": method,
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
                "relative_fit_error": relative_error,
                **summary,
                "warning": "Model-estimated dimensions; unobserved body parts may be imputed. Fixed per-frame poses; accuracy unvalidated.",
            },
            "physical_accuracy_validated": False,
        }
        if research.get("fit", {}).get("diagnostics"):
            from .mesh import section_perimeter

            initial_vertices, _ = mhr_vertices(initial, params)
            initial_mesh = {
                "vertices": initial_vertices.tolist(),
                "faces": faces.tolist(),
            }
            write_json(
                attempt / "measurement-diagnostics.json",
                {
                    "source_kind": source_kind,
                    "diagnostic_only": True,
                    "physical_accuracy_validated": False,
                    "sections": [
                        {
                            "definition": section,
                            "initial_perimeter_model_m": section_perimeter(
                                initial_mesh, section
                            ),
                            "fitted_perimeter_model_m": section_perimeter(
                                mesh, section
                            ),
                        }
                        for section in profile["sections"]
                    ],
                },
            )
        if diagnostic_real:
            mesh.update(
                diagnostic_only=True,
                diagnostic_label="診断用",
                adoption_validated=False,
            )
        validate_mesh(mesh)
        if not profile_path.exists():
            write_json(profile_path, profile)
        write_json(session / "mesh.json", mesh)
        replace_json(
            status_file,
            {
                "state": "complete",
                "stage": "complete",
                "elapsed_s": round(time.monotonic() - started, 1),
                "method": method,
                "usable_frames": len(estimates),
                "total_frames": len(indices),
                "excluded_frames": summary["excluded_frames"],
                "exclusion_reasons": summary["exclusion_reasons"],
                "rotation_confirmed": complete,
                "peak_rss_mb": round(peak["rss_bytes"] / 1024**2),
                "encoder_device": encoder_device,
                "peak_cuda_allocated_mb": (
                    round(torch.cuda.max_memory_allocated(0) / 1024**2)
                    if encoder_device == "cuda"
                    else None
                ),
                "peak_cuda_reserved_mb": (
                    round(torch.cuda.max_memory_reserved(0) / 1024**2)
                    if encoder_device == "cuda"
                    else None
                ),
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
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--research-options")
    parser.add_argument(
        "--diagnostic-real",
        action="store_true",
        help="Explicit diagnostic-only research execution on real videos; requires --research-options",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.store,
        args.session,
        args.allow_synthetic,
        args.research_options,
        args.diagnostic_real,
    )
