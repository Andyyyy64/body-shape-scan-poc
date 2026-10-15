"""Observe all native RGB head/calf scale cues using an independent pose model."""

from __future__ import annotations

import hashlib

import numpy as np

from .head_anatomy_scale import scan_head_anatomy
from .head_scale import representative_width
from .io import InputError, sha256
from .leg_scale import calf_widths, scan_leg_anatomy
from .subpixel_edges import refine_positions


def create_detector(config):
    from pathlib import Path

    from ultralytics import YOLO

    model = Path(config["pose_model"])
    if sha256(model) != config["pose_model_sha256"]:
        raise InputError("Independent anatomy model changed")
    detector = YOLO(str(model))
    if detector.task != "pose":
        raise InputError("Independent anatomy requires a pose model")
    return detector


def observe_cues(context, identifier, data, detector, output):
    import cv2
    import torch

    frames = data["frames"][data["shared_source_order"].astype(int)]
    if len(frames) != 48 or len(set(frames)) != 48:
        raise InputError("Stable scale cues require all48 independent native frames")
    target = data["width_target"]
    if target.shape != (48, 1024, 2):
        raise InputError("Stable cues require native dense image contours")
    video = context.store / context.rows[identifier]["session_id"] / "capture.bin"
    if sha256(video) != context.rows[identifier]["video_sha256"]:
        raise InputError("Scale-cue capture video changed")
    lookup = {int(f): i for i, f in enumerate(frames)}
    head = [None] * 48
    leg = [None] * 48
    points = [None] * 48
    decoded = {}
    evidence = []
    capture = cv2.VideoCapture(str(video))
    try:
        for frame in range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, image = capture.read()
            if not ok:
                raise InputError("Scale-cue RGB decoding failed")
            if frame not in lookup:
                continue
            i = lookup[frame]
            prediction = detector.predict(
                image, classes=[0], device="cpu", verbose=False, save=False
            )[0]
            torch.set_num_threads(1)
            if prediction.keypoints is None or not len(prediction.keypoints):
                raise InputError(
                    "Independent anatomy missing on a required scale-cue frame"
                )
            boxes = prediction.boxes.xyxy.cpu().numpy()
            low = target[i].min(0)
            high = target[i].max(0)
            intersection = np.maximum(
                np.minimum(boxes[:, 2:], high) - np.maximum(boxes[:, :2], low), 0
            ).prod(1)
            area = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
            overlap = intersection / np.maximum(
                area + (high - low).prod() - intersection, 1
            )
            index = int(overlap.argmax())
            if overlap[index] < 0.5:
                raise InputError(
                    "Independent anatomy does not match the observed person"
                )
            keypoints = prediction.keypoints.data.cpu().numpy()[index]
            hr, hp, _ = scan_head_anatomy(target[i], keypoints)
            refined_head = refine_positions(image, hr, hp)
            head[i] = representative_width(refined_head["positions"][None])[0]
            lr, spans = scan_leg_anatomy(target[i], keypoints)
            counts = np.array([len(s) for s in spans])
            refined_leg = refine_positions(
                image, np.repeat(lr, counts), np.concatenate(spans)
            )
            leg[i] = float(
                np.median(
                    calf_widths(
                        np.split(refined_leg["positions"], np.cumsum(counts)[:-1])
                    )
                )
            )
            points[i] = keypoints
            decoded[str(frame)] = hashlib.sha256(image.tobytes()).hexdigest()
            evidence.append(
                {
                    "frame": frame,
                    "head_rows": 64,
                    "calf_rows": 32,
                    "calf_intervals": int(counts.sum()),
                    "head_refined_fraction": float(refined_head["refined"].mean()),
                    "calf_refined_fraction": float(refined_leg["refined"].mean()),
                }
            )
    finally:
        capture.release()
    if any(x is None for x in head + leg + points):
        raise InputError("Scale-cue native frames are incomplete")
    values = {
        "observed_frames": frames,
        "rgb_head_width_px": np.array(head),
        "rgb_leg_width_px": np.array(leg),
        "keypoints": np.array(points),
    }
    np.savez_compressed(output, **values)
    output.chmod(0o600)
    return values, {
        "cache_sha256": sha256(output),
        "video_sha256": context.rows[identifier]["video_sha256"],
        "decoded_bgr_sha256": decoded,
        "all48_views_retained": True,
        "frame_observation_evidence": evidence,
    }
