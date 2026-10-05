"""Post-hoc generator projection diagnostics, never supplied to reconstruction."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .io import InputError, read_json, replace_json, sha256


def diagnostics(root: Path) -> dict:
    import numpy as np

    from .synthetic_lab import deform

    if read_json(root / "store" / "store.json").get("synthetic_experiment") is not True:
        raise InputError("Projection diagnostics requires a synthetic lab")
    manifest, body = read_json(root / "matrix.json"), read_json(root / "body.json")
    if body.get("generator") not in {
        "zero_shape_zero_pose_MHR",
        "heldout_axis_deformation_of_MHR",
    }:
        raise InputError("Unsupported generator projection")
    if sha256(root / "body.npz") != body["body_sha256"]:
        raise InputError("Generator body changed")
    with np.load(root / "body.npz") as mesh:
        vertices, joints, faces = mesh["vertices"], mesh["joints"], mesh["faces"]
    body["faces"] = faces
    y_center = (vertices[:, 1].max() + vertices[:, 1].min()) / 2

    def project(points, angle, distance, focal):
        rotation = np.array(
            [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ]
        )
        position = points @ rotation.T * [1, -1, -1] + [0, y_center, distance]
        return position[:, :2] / position[:, 2:] * focal + [320, 320]

    keypoints = []
    baselines = [
        c
        for c in manifest["cases"]
        if c["spec"]["group"] == "baseline" and c["state"] == "complete"
    ]
    for case in baselines:
        path = root / "store" / case["session_id"]
        truth = read_json(path / "truth.json")
        spec = case["spec"]
        errors, yaws, angles = [], [], []
        attempts = list(path.glob("attempt_*/fit.json"))
        if len(attempts) != 1:
            raise InputError("Expected one fitted baseline attempt")
        attempt = attempts[0].parent
        focal, depth = [], []
        for frame in read_json(attempt / "frames.json")["frames"]:
            if frame["state"] != "estimated":
                continue
            angle = (
                frame["frame"] / truth["frame_count"] * 2 * math.pi
                + (spec["seed"] % 11) * 0.025
            )
            expected = project(
                joints, angle, spec["distance"], truth["camera_focal_px"]
            )
            with np.load(attempt / frame["output"]) as estimate:
                errors.extend(
                    np.linalg.norm(
                        estimate["pred_keypoints_2d"][[5, 6, 9, 10]]
                        - expected[[5, 6, 9, 10]],
                        axis=1,
                    ).tolist()
                )
                focal.append(float(estimate["focal_length"]))
                depth.append(float(estimate["pred_cam_t"][2]))
            yaws.append(frame["yaw_rad"])
            angles.append(angle)
        choices = []
        for sign in (-1, 1):
            delta = np.asarray(yaws) - sign * np.asarray(angles)
            offset = np.angle(np.mean(np.exp(1j * delta)))
            error = np.angle(np.exp(1j * (delta - offset)))
            choices.append(float(np.mean(error**2)))
        keypoints.append(
            {
                "name": case["name"],
                "core_keypoint_mean_error_px": float(np.mean(errors)),
                "core_keypoint_p95_error_px": float(np.percentile(errors, 95)),
                "relative_yaw_rms_deg_after_offset_alignment": math.sqrt(min(choices))
                * 180
                / math.pi,
                "estimated_focal_px": sorted(set(focal)),
                "camera_depth_median_model_m": float(np.median(depth)),
            }
        )
    displacement = []
    if baselines:
        # Same known local deformation as the declared experiment. Projection is
        # diagnostic only; reconstruction never receives these vertices/angles.
        spec = baselines[0]["spec"]
        truth = read_json(root / "store" / baselines[0]["session_id"] / "truth.json")
        selected = (abs(vertices[:, 1] - body["section_y"]) < 0.04) & (
            abs(vertices[:, 0]) < 0.25
        )
        for delta in (-0.02, -0.01, 0.01, 0.02, 0.04):
            changed, before, after = deform(vertices, body, delta)
            pixels = []
            for frame in np.linspace(0, truth["frame_count"] - 1, 16).astype(int):
                angle = (
                    frame / truth["frame_count"] * 2 * math.pi
                    + (spec["seed"] % 11) * 0.025
                )
                a = project(vertices, angle, spec["distance"], truth["camera_focal_px"])
                b = project(changed, angle, spec["distance"], truth["camera_focal_px"])
                pixels.extend(
                    np.linalg.norm(a[selected] - b[selected], axis=1).tolist()
                )
            displacement.append(
                {
                    "input_delta_model_mm": (after - before) * 1000,
                    "max_projected_displacement_px": float(np.max(pixels)),
                    "p95_projected_displacement_px": float(np.percentile(pixels, 95)),
                }
            )
    result = {
        "source_kind": "synthetic",
        "diagnostic_only_oracle_projection": True,
        "physical_accuracy_validated": False,
        "keypoints": keypoints,
        "displacements": displacement,
        "analysis_source_sha256": sha256(Path(__file__)),
        "interpretation": "Generator landmarks/camera/angles used only for post-hoc error analysis. Relative yaw aligns a constant offset/direction per capture. Pixel magnitude comparisons are not causal proofs or human accuracy evidence.",
    }
    replace_json(root / "projection-diagnostics.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    diagnostics(Path(args.out).resolve())
    print("Saved post-hoc projection diagnostics")
