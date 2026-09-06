"""Explicit official SAM initialization; never a substitute for metric joint fitting."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .io import InputError, new_run, read_json, relative_asset, sha256, write_json
from .vision import camera, libraries


def initialize(
    scan: str, source: str, checkpoint: str, mhr_model: str, output: str
) -> dict:
    cv2, np = libraries()
    import torch

    if not torch.cuda.is_available():
        raise InputError("Official SAM inference requires an available CUDA device")
    source_path = Path(source).resolve()
    if not (source_path / "sam_3d_body" / "__init__.py").is_file():
        raise InputError("Supply the official SAM source checkout")
    for p in (checkpoint, mhr_model):
        if not Path(p).is_file():
            raise InputError("Local official weights required; no implicit download")
    commit = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(source_path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip():
        raise InputError("SAM tracked source changes must be committed for provenance")
    data = read_json(scan)
    if data.get("schema_version") != "silhouette_scan.v1" or data.get(
        "source_kind"
    ) not in {"synthetic", "real"}:
        raise InputError("Expected silhouette_scan.v1")
    if not 1 <= len(data["views"]) <= 72:
        raise InputError("Use 1..72 selected frames, one person per frame")
    K, dist, (width, height) = camera(data["camera"])
    if np.any(dist != 0):
        raise InputError(
            "SAM input must already be undistorted, with matching masks and updated K"
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(source_path))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    root = new_run(output)
    model, cfg = load_sam_3d_body(
        checkpoint_path=checkpoint, device="cuda", mhr_path=mhr_model
    )
    estimator = SAM3DBodyEstimator(model, cfg)
    parent = Path(scan).resolve().parent
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for i, view in enumerate(data["views"]):
        image_path, mask_path = (
            relative_asset(parent, view["image"]),
            relative_asset(parent, view["mask"]),
        )
        bgr = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if (
            bgr is None
            or mask is None
            or bgr.shape[:2] != (height, width)
            or mask.shape != (height, width)
        ):
            raise InputError("Image/mask dimensions do not match calibrated K")
        if not set(np.unique(mask)).issubset({0, 1, 255}):
            raise InputError("SAM prompts must be binary observed masks")
        yy, xx = np.where(mask > 0)
        if len(xx) == 0:
            raise InputError("Empty observed mask")
        bbox = np.array(
            [[xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]], dtype=np.float32
        )
        result = estimator.process_one_image(
            img=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            bboxes=bbox,
            masks=(mask > 0).astype("uint8")[None],
            cam_int=torch.tensor(K, dtype=torch.float32)[None],
            inference_type="body",
        )
        if len(result) != 1:
            raise InputError("Expected exactly one body prediction")
        prediction = {k: np.asarray(v) for k, v in result[0].items() if v is not None}
        if any(
            v.dtype.kind not in "biuf" or not np.isfinite(v).all()
            for v in prediction.values()
        ):
            raise InputError("Non-numeric or nonfinite SAM prediction")
        for required in [
            "pred_vertices",
            "mhr_model_params",
            "shape_params",
            "scale_params",
            "pred_cam_t",
        ]:
            if required not in prediction:
                raise InputError("Required official SAM output is missing")
        name = f"prediction_{i:03d}.npz"
        np.savez_compressed(root / name, **prediction)
        (root / name).chmod(0o600)
        outputs.append(
            {
                "file": name,
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
            }
        )
    np.save(root / "faces.npy", estimator.faces, allow_pickle=False)
    (root / "faces.npy").chmod(0o600)
    torch.cuda.synchronize()
    summary = {
        "schema_version": "sam_initialization.v1",
        "source_kind": data["source_kind"],
        "status": "initialized_not_metric_validated",
        "sam_commit": commit,
        "checkpoint_sha256": sha256(Path(checkpoint)),
        "mhr_sha256": sha256(Path(mhr_model)),
        "outputs": outputs,
        "elapsed_s": time.monotonic() - start,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "units": "upstream native output; MHR metric round trip not established by this command",
        "physical_accuracy_validated": False,
    }
    write_json(root / "initialization.json", summary)
    return {"status": summary["status"], "frames": len(outputs)}
