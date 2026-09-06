"""Optional local OpenCV operations. All poses and metric calibration are explicit inputs."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

from .geometry import number
from .io import (
    InputError,
    new_run,
    private_path,
    read_json,
    relative_asset,
    sha256,
    write_json,
)


def libraries():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise InputError("Install the vision extra: uv sync --extra vision") from exc
    cv2.setNumThreads(1)
    return cv2, np


def save_image(path, image):
    cv2, _ = libraries()
    path = private_path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise InputError("Image could not be encoded")
    with path.open("xb") as handle:
        path.chmod(0o600)
        handle.write(encoded.tobytes())


def record(
    source: str,
    output: str,
    seconds: float,
    preview: bool,
    delay_s: float = 0.0,
    source_kind: str = "real",
) -> dict:
    cv2, _ = libraries()
    seconds = number(seconds, "seconds", positive=True)
    delay_s = number(delay_s, "delay_s")
    if not 0 <= delay_s <= 30 or source_kind not in {"synthetic", "real"}:
        raise InputError("Invalid delay or source kind")
    if seconds > 300:
        raise InputError("Capture is bounded to 300 seconds per run")
    if not source.isdecimal() and not Path(source).is_file():
        raise InputError("Source must be a local camera index or existing local video")
    root = new_run(output)
    cap = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    writer = None
    times = []
    try:
        if not cap.isOpened():
            raise InputError(
                "Camera/video unavailable; check local camera permission or source"
            )
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise InputError("Source did not provide a valid frame rate")
        time.sleep(delay_s)
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(root / "capture.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise InputError("Video encoder unavailable")
                (root / "capture.avi").chmod(0o600)
            if frame.shape[:2] != (height, width):
                raise InputError("Capture dimensions changed within the recording")
            writer.write(frame)
            times.append(time.monotonic() - start)
            if preview:
                cv2.imshow("Local capture - Q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if not times:
            raise InputError("No frames captured")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
    result = {
        "schema_version": "capture.v1",
        "status": "recorded_not_measured",
        "source_kind": source_kind,
        "video": "capture.avi",
        "frames": len(times),
        "fps_container": fps,
        "capture_elapsed_s": times,
        "width": width,
        "height": height,
        "video_sha256": sha256(root / "capture.avi"),
    }
    write_json(root / "capture.json", result)
    return {"status": result["status"], "frames": len(times)}


def extract(video: str, output: str, stride: int, max_frames: int) -> dict:
    cv2, _ = libraries()
    if stride < 1 or not 1 <= max_frames <= 1000:
        raise InputError("stride must be positive and max_frames within 1..1000")
    if not Path(video).is_file():
        raise InputError("Video must be an existing local file")
    root = new_run(output)
    cap = cv2.VideoCapture(str(video))
    rows = []
    index = 0
    try:
        if not cap.isOpened():
            raise InputError("Video unavailable")
        while len(rows) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                name = f"frame_{index:06d}.png"
                save_image(root / name, frame)
                rows.append(
                    {
                        "image": name,
                        "frame_index": index,
                        "video_time_ms": cap.get(cv2.CAP_PROP_POS_MSEC),
                        "yaw_rad": None,
                    }
                )
            index += 1
    finally:
        cap.release()
    if not rows:
        raise InputError("No frames extracted")
    write_json(
        root / "frames.json",
        {
            "schema_version": "frames.v1",
            "frames": rows,
            "note": "Temporal sampling only. Annotate/estimate angles before angle selection.",
        },
    )
    return {"status": "frames_extracted_angles_unknown", "frames": len(rows)}


def calibrate(
    images: str, output: str, columns: int, rows: int, square_m: float
) -> dict:
    cv2, np = libraries()
    if not 3 <= columns <= 30 or not 3 <= rows <= 30:
        raise InputError("Inner corner counts must be within 3..30")
    square_m = number(square_m, "square_m", positive=True)
    object_points = np.zeros((columns * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    points, objects, names, rejected, sizes = [], [], [], 0, set()
    for path in sorted(Path(images).glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise InputError("Unreadable calibration image")
        sizes.add((image.shape[1], image.shape[0]))
        ok, corners = cv2.findChessboardCornersSB(image, (columns, rows))
        if ok:
            points.append(corners)
            objects.append(object_points)
            names.append(sha256(path))
        else:
            rejected += 1
    if len(sizes) != 1 or len(points) < 8:
        raise InputError(
            "Need at least 8 detected boards with identical image dimensions"
        )
    if len(set(names)) != len(names):
        raise InputError("Duplicate calibration images are not independent board poses")
    size = sizes.pop()
    rms, K, distortion, rotations, translations = cv2.calibrateCamera(
        objects, points, size, None, None
    )
    per_image = []
    for obj, obs, r, t in zip(objects, points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, K, distortion)
        per_image.append(
            float(np.sqrt(np.mean(np.sum((projected - obs) ** 2, axis=2))))
        )
    if not np.isfinite(K).all() or not np.isfinite(distortion).all():
        raise InputError("Nonfinite calibration result")
    result = {
        "schema_version": "camera_calibration.v1",
        "image_size": list(size),
        "K": K.tolist(),
        "distortion": distortion.ravel().tolist(),
        "rms_px": float(rms),
        "per_image_rms_px": per_image,
        "accepted": len(points),
        "rejected": rejected,
        "square_m": square_m,
        "input_sha256": names,
        "status": "needs_physical_validation",
        "metric_body_pose_established": False,
    }
    root = new_run(output)
    write_json(root / "calibration.json", result)
    return {"status": result["status"], "rms_px": float(rms), "accepted": len(points)}


def camera(data: dict):
    _, np = libraries()
    K = np.asarray(data["K"], dtype=float)
    dist = np.asarray(data["distortion"], dtype=float)
    size = data["image_size"]
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise InputError("Invalid camera matrix")
    if (
        not np.allclose(K[2], [0, 0, 1])
        or not np.isclose(K[0, 1], 0)
        or not np.isclose(K[1, 0], 0)
    ):
        raise InputError("Expected OpenCV pinhole camera matrix")
    if (
        dist.ndim != 1
        or len(dist) not in (4, 5, 8, 12, 14)
        or not np.isfinite(dist).all()
    ):
        raise InputError("Invalid distortion coefficients")
    if len(size) != 2 or any(
        isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in size
    ):
        raise InputError("Invalid camera image dimensions")
    return K, dist, tuple(size)


def reconstruct(manifest_path: str, output: str) -> dict:
    """Intersect observed silhouette cones on metric planes; poses are supplied, not inferred."""
    cv2, np = libraries()
    data = read_json(manifest_path)
    if data.get("schema_version") != "silhouette_scan.v1" or data.get(
        "source_kind"
    ) not in {"real", "synthetic"}:
        raise InputError("Expected silhouette_scan.v1 and a source_kind")
    for key in ("measurement_definition_id", "pose_evidence_id"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise InputError("Measurement and pose evidence identifiers required")
    result = {
        "schema_version": "section_measurement.v1",
        "source_kind": data["source_kind"],
        "method": "calibrated_visual_hull_convex_section",
        "status": "needs_validation",
        "measurement_definition_id": data["measurement_definition_id"],
        "physical_accuracy_validated": False,
        "uncertainty_m": None,
        "input_sha256": sha256(Path(manifest_path)),
        "sections": [],
    }
    if not data.get("scale_evidence_id"):
        result["status"] = "unscaled"
        root = new_run(output)
        write_json(root / "measurement.json", result)
        return result
    K, distortion, (width, height) = camera(data["camera"])
    bounds = data["bounds_xz_m"]
    if len(bounds) != 4:
        raise InputError("bounds_xz_m must be [xmin, xmax, zmin, zmax]")
    xmin, xmax, zmin, zmax = [number(x, "bounds") for x in bounds]
    step = number(data["grid_step_m"], "grid_step_m", positive=True)
    if xmax <= xmin or zmax <= zmin:
        raise InputError("Invalid reconstruction bounds")
    nx, nz = math.floor((xmax - xmin) / step) + 1, math.floor((zmax - zmin) / step) + 1
    if nx < 3 or nz < 3 or nx * nz > 1_000_000:
        raise InputError(
            "Cross-section grid must have 3+ cells per side and at most 1 million cells"
        )
    sections = data["sections"]
    if not 1 <= len(sections) <= 20 or len({s["id"] for s in sections}) != len(
        sections
    ):
        raise InputError("Need 1..20 distinct section identifiers")
    views = data["views"]
    if not 4 <= len(views) <= 72:
        raise InputError("Need 4..72 calibrated views")
    allowed = number(data["max_yaw_gap_deg"], "max_yaw_gap_deg", positive=True)
    if allowed > 90:
        raise InputError("Maximum angular gap cannot exceed 90 degrees")
    loaded = []
    angles = []
    parent = Path(manifest_path).resolve().parent
    for view in views:
        mask_path = relative_asset(parent, view["mask"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if (
            mask is None
            or mask.shape != (height, width)
            or not set(np.unique(mask)).issubset({0, 1, 255})
        ):
            raise InputError("Masks must be binary and match the calibrated full image")
        if not mask.any():
            raise InputError("Empty mask")
        R, t = np.asarray(view["R"], float), np.asarray(view["t_m"], float)
        if (
            R.shape != (3, 3)
            or t.shape != (3,)
            or not np.isfinite(R).all()
            or not np.isfinite(t).all()
        ):
            raise InputError("Invalid body-to-camera pose")
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(R), 1.0, atol=1e-5
        ):
            raise InputError("Pose must be a proper rigid rotation; no hidden scaling")
        center = -R.T @ t
        if math.hypot(center[0], center[2]) <= 1e-8:
            raise InputError("Cannot determine horizontal camera coverage")
        angles.append(math.atan2(center[0], center[2]) % (2 * math.pi))
        loaded.append((mask > 0, cv2.Rodrigues(R)[0], R, t, sha256(mask_path)))
    angles = sorted(set(round(a, 9) for a in angles))
    gap = max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]))
    if math.degrees(gap) > allowed + 1e-6:
        raise InputError("Incomplete 360-degree coverage in actual metric poses")
    xx, zz = np.meshgrid(xmin + np.arange(nx) * step, zmin + np.arange(nz) * step)
    for section in sections:
        y = number(section["y_m"], "section y_m")
        seed = [number(v, "section center") for v in section["center_xz_m"]]
        if len(seed) != 2 or not xmin < seed[0] < xmax or not zmin < seed[1] < zmax:
            raise InputError("Section center must lie within metric bounds")
        points = np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()])
        inside = np.ones(len(points), dtype=bool)
        for mask, rvec, R, t, _ in loaded:
            cam_points = points @ R.T + t
            if np.any(cam_points[:, 2] <= 0):
                raise InputError("Reconstruction volume crosses the camera plane")
            projected, _ = cv2.projectPoints(points, rvec, t, K, distortion)
            coords = projected.reshape(-1, 2)
            pix = np.rint(coords).astype(int)
            visible = (
                (pix[:, 0] >= 0)
                & (pix[:, 0] < width)
                & (pix[:, 1] >= 0)
                & (pix[:, 1] < height)
            )
            samples = np.zeros(len(points), bool)
            samples[visible] = mask[pix[visible, 1], pix[visible, 0]]
            inside &= samples
        grid = inside.reshape(nz, nx).astype("uint8")
        _, labels = cv2.connectedComponents(grid)
        sx, sz = round((seed[0] - xmin) / step), round((seed[1] - zmin) / step)
        if not (0 <= sx < nx and 0 <= sz < nz):
            raise InputError("Section center is outside the discrete grid")
        label = labels[sz, sx]
        measured = {"id": section["id"], "y_m": y, "value_m": None}
        if label == 0:
            measured.update(
                status="missing", reason="no_consistent_silhouette_at_section_center"
            )
        else:
            component = labels == label
            if (
                component[0].any()
                or component[-1].any()
                or component[:, 0].any()
                or component[:, -1].any()
            ):
                measured.update(
                    status="missing", reason="section_touches_reconstruction_bounds"
                )
            else:
                xz = np.column_stack([xx[component], zz[component]]).astype("float32")
                hull = cv2.convexHull(xz).reshape(-1, 2)
                if len(hull) < 3:
                    measured.update(status="missing", reason="degenerate_section")
                else:
                    boundary_points = np.column_stack(
                        [hull[:, 0], np.full(len(hull), y), hull[:, 1]]
                    )
                    clipped = False
                    for _, rvec, _, t, _ in loaded:
                        uv, _ = cv2.projectPoints(
                            boundary_points, rvec, t, K, distortion
                        )
                        uv = uv.reshape(-1, 2)
                        clipped |= bool(
                            (uv[:, 0] < 2).any()
                            or (uv[:, 0] > width - 3).any()
                            or (uv[:, 1] < 2).any()
                            or (uv[:, 1] > height - 3).any()
                        )
                    if clipped:
                        measured.update(status="missing", reason="section_out_of_frame")
                    else:
                        measured.update(
                            status="needs_validation",
                            value_m=float(cv2.arcLength(hull, True)),
                            polygon_xz_m=hull.tolist(),
                            perimeter_definition="convex_hull",
                        )
        result["sections"].append(measured)
    result.update(
        grid_step_m=step,
        max_yaw_gap_deg=math.degrees(gap),
        scale_evidence_id=data["scale_evidence_id"],
        pose_evidence_id=data["pose_evidence_id"],
        mask_sha256=[v[-1] for v in loaded],
    )
    if all(s["value_m"] is None for s in result["sections"]):
        result["status"] = "missing"
    root = new_run(output)
    write_json(root / "measurement.json", result)
    return result


def segment(images: str, model: str, output: str) -> dict:
    cv2, np = libraries()
    model_path = Path(model)
    if not model_path.is_file():
        raise InputError("Supply a local segmentation checkpoint; no implicit download")
    root = new_run(output)
    os.environ["YOLO_CONFIG_DIR"] = str(root / "upstream-config")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO

    predictor = YOLO(str(model_path))
    if predictor.task != "segment" or predictor.names.get(0) != "person":
        raise InputError("Expected a COCO person segmentation checkpoint")
    import torch

    torch.set_num_threads(1)
    frames = read_json(Path(images) / "frames.json")["frames"]
    records = []
    for frame in frames:
        image_path = relative_asset(Path(images), frame["image"])
        image = cv2.imread(str(image_path))
        predictions = predictor.predict(
            image,
            classes=[0],
            retina_masks=True,
            verbose=False,
            save=False,
            device="cpu",
            project=str(root / "upstream"),
        )
        pred = predictions[0]
        if pred.masks is None or len(pred.masks.data) != 1:
            records.append(
                {
                    "image": frame["image"],
                    "status": "missing",
                    "reason": "expected_exactly_one_person",
                }
            )
            continue
        mask = (pred.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        if mask.shape != image.shape[:2]:
            raise InputError("Segmentation output is not in source-image coordinates")
        name = Path(frame["image"]).stem + ".png"
        save_image(root / name, mask)
        records.append(
            {"image": frame["image"], "mask": name, "status": "segmented_pose_unknown"}
        )
    write_json(
        root / "masks.json",
        {
            "model_sha256": sha256(model_path),
            "records": records,
            "note": "Masks alone do not establish metric scale or body pose.",
        },
    )
    return {
        "status": "masks_generated_not_measured",
        "usable": sum(r.get("mask") is not None for r in records),
        "attempts": len(records),
    }


def undistort(images: str, calibration: str, output: str) -> dict:
    """One coordinate space for SAM RGB/masks. Segment the output images, not the originals."""
    cv2, np = libraries()
    info = read_json(calibration)
    K, dist, (width, height) = camera(info)
    frames = read_json(Path(images) / "frames.json")["frames"]
    root = new_run(output)
    records = []
    for i, frame in enumerate(frames):
        path = relative_asset(Path(images), frame["image"])
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (height, width):
            raise InputError(
                "Calibration and frame sizes differ; do not reuse K after changing capture mode"
            )
        name = f"frame_{i:06d}.png"
        save_image(root / name, cv2.undistort(image, K, dist, None, K))
        records.append({**frame, "image": name, "source_sha256": sha256(path)})
    write_json(
        root / "frames.json",
        {
            "schema_version": "frames.v1",
            "frames": records,
            "coordinate_space": "undistorted_full_image",
        },
    )
    write_json(
        root / "calibration.json",
        {
            **info,
            "K": K.tolist(),
            "distortion": [0.0] * 5,
            "source_calibration_sha256": sha256(Path(calibration)),
            "coordinate_space": "undistorted_full_image",
        },
    )
    return {"status": "undistorted_not_measured", "frames": len(records)}


def sampled_video_frames(video_path: str, samples: int, max_frames: int = 3600):
    """Count by decoding, then sample sequentially; MediaRecorder files may lack seek indexes."""
    cv2, np = libraries()
    if not Path(video_path).is_file() or not 3 <= samples <= 64:
        raise InputError("Use a local video and 3..64 samples")
    cap = cv2.VideoCapture(video_path)
    count = 0
    try:
        if not cap.isOpened():
            raise InputError("Saved video could not be decoded")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            if count > max_frames:
                raise InputError("Video exceeds the bounded scan length")
    finally:
        cap.release()
    if count < 3:
        raise InputError("At least three video frames are required")
    selected = set(np.linspace(0, count - 1, min(samples, count), dtype=int).tolist())
    cap = cv2.VideoCapture(video_path)
    result = []
    try:
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise InputError(
                    "Video changed or became unreadable between decoding passes"
                )
            if index in selected:
                result.append((index, frame))
    finally:
        cap.release()
    return result, count
