"""Synthetic controls, always labeled as non-physical evidence."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .geometry import (
    convex_hull,
    ellipse_perimeter,
    polygon_perimeter,
    support_width,
    width_integral,
)
from .io import InputError, new_run, private_path, write_json
from .statistics import analyze_pairs, detection_power


def html_report(root: Path, data: dict) -> None:
    root = private_path(root)
    title = "Exploratory report: physical accuracy not validated"
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with (root / "report.html").open("x", encoding="utf-8") as f:
        f.write(
            '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            "<title>"
            + title
            + "</title><style>body{font:16px system-ui;max-width:70rem;margin:3rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:1rem}</style>"
            "<h1>" + title + "</h1><p>No adoption decision follows from this report. "
            "Participant-derived reports must remain private.</p><pre>"
            + html.escape(text)
            + "</pre>"
        )
    (root / "report.html").chmod(0o600)


def run_demo(output: str) -> dict:
    root = new_run(output)
    points = [
        (
            0.18 * math.cos(i * 2 * math.pi / 4096),
            0.12 * math.sin(i * 2 * math.pi / 4096),
        )
        for i in range(4096)
    ]
    # Localized angular deformation, not a change to the same ellipse parameters being fitted.
    deformed = [
        (
            x,
            z
            - 0.006
            * math.exp(-(((math.atan2(z / 0.12, x / 0.18) - math.pi / 2) / 0.55) ** 2)),
        )
        for x, z in points
    ]
    cases = {
        "ellipse": points,
        "local_deformation": convex_hull(deformed),
        "scale_error_1pct": [(x * 1.01, z * 1.01) for x, z in points],
        "nonellipse": [
            (-0.18, -0.06),
            (-0.12, -0.12),
            (0.10, -0.12),
            (0.19, -0.02),
            (0.13, 0.13),
            (-0.12, 0.12),
        ],
    }
    results = []
    for name, polygon in cases.items():
        truth = polygon_perimeter(polygon)
        observations = [
            {
                "angle_rad": i * math.pi / 360,
                "width_m": support_width(polygon, i * math.pi / 360),
            }
            for i in range(360)
        ]
        measured = width_integral(observations, 5)["value_m"]
        results.append(
            {
                "case": name,
                "reference_perimeter_m": truth,
                "width_integral_m": measured,
                "error_m": measured - truth,
                "ellipse_baseline_m": ellipse_perimeter(
                    support_width(polygon, 0) / 2,
                    support_width(polygon, math.pi / 2) / 2,
                ),
            }
        )
    baseline = results[0]["reference_perimeter_m"]
    pairs = []
    for i, delta in enumerate([0.0, -0.005, -0.01, -0.02, 0.005, 0.01, 0.02]):
        changed = [
            (x * (baseline + delta) / baseline, z * (baseline + delta) / baseline)
            for x, z in points
        ]
        estimate = (
            width_integral(
                [
                    {
                        "angle_rad": j * math.pi / 72,
                        "width_m": support_width(changed, j * math.pi / 72),
                    }
                    for j in range(72)
                ],
                5,
            )["value_m"]
            - baseline
        )
        for method, prediction in [
            ("width_integral", estimate),
            ("constant_shape", 0.0),
        ]:
            pairs.append(
                {
                    "pair_id": f"synthetic_pair_{i}",
                    "subject_id": f"synthetic_subject_{i}",
                    "baseline_session_id": f"synthetic_before_{i}",
                    "followup_session_id": f"synthetic_after_{i}",
                    "device_class": "synthetic_camera",
                    "part": "synthetic_section",
                    "method": method,
                    "measurement_definition_id": "synthetic_plane.v1",
                    "split": "evaluation",
                    "status": "ok",
                    "predicted_delta_m": prediction,
                    "reference_delta_m": delta,
                    "reference_uncertainty_m": 0.0,
                }
            )
    dataset = {
        "schema_version": "paired_validation.v1",
        "source_kind": "synthetic",
        "pairs": pairs,
    }
    write_json(root / "pairs.json", dataset)
    report = {
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "cases": results,
        "power_examples": [detection_power(s, 0.01) for s in [0.005, 0.0036, 0.0025]],
        "paired_analysis": analyze_pairs(dataset, 0.003, 0.01),
        "known_scale_false_delta_m": results[2]["width_integral_m"]
        - results[0]["width_integral_m"],
    }
    write_json(root / "report.json", report)
    html_report(root, report)
    return report


def render_demo(output: str, delta_m: float = 0.0) -> dict:
    from .vision import libraries, reconstruct, save_image

    cv2, np = libraries()
    root = new_run(output)
    K = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])
    polygon = np.array(
        [
            [
                0.16 * math.cos(i * 2 * math.pi / 720),
                0.11 * math.sin(i * 2 * math.pi / 720),
            ]
            for i in range(720)
        ]
    )
    original_perimeter = polygon_perimeter([tuple(p) for p in polygon])
    from .geometry import number

    delta_m = number(delta_m, "delta_m")
    if original_perimeter + delta_m <= 0:
        raise InputError("Requested synthetic circumference must be positive")
    polygon *= (original_perimeter + delta_m) / original_perimeter
    vertices = np.vstack(
        [
            np.column_stack([polygon[:, 0], np.full(len(polygon), y), polygon[:, 1]])
            for y in [-0.30, 0.30]
        ]
    )
    views = []
    for i in range(16):
        yaw = i * 2 * math.pi / 16
        R = np.diag([1.0, -1.0, -1.0]) @ np.array(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ]
        )
        t = np.array([0.0, 0.0, 2.0])
        uv, _ = cv2.projectPoints(vertices, cv2.Rodrigues(R)[0], t, K, np.zeros(5))
        contour = cv2.convexHull(np.rint(uv).astype("int32"))
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(mask, contour, 255)
        name = f"mask_{i:03d}.png"
        save_image(root / name, mask)
        views.append({"mask": name, "yaw_rad": yaw, "R": R.tolist(), "t_m": t.tolist()})
    manifest = {
        "schema_version": "silhouette_scan.v1",
        "source_kind": "synthetic",
        "measurement_definition_id": "synthetic_prism_plane.v1",
        "scale_evidence_id": "synthetic_metric_geometry",
        "pose_evidence_id": "synthetic_known_poses",
        "camera": {"K": K.tolist(), "distortion": [0.0] * 5, "image_size": [640, 480]},
        "bounds_xz_m": [-0.23, 0.23, -0.2, 0.2],
        "grid_step_m": 0.0015,
        "max_yaw_gap_deg": 30,
        "sections": [{"id": "synthetic_center", "y_m": 0.0, "center_xz_m": [0.0, 0.0]}],
        "views": views,
    }
    write_json(root / "scan.json", manifest)
    reconstructed = reconstruct(str(root / "scan.json"), str(root / "reconstructed"))
    truth = polygon_perimeter([tuple(p) for p in polygon])
    value = reconstructed["sections"][0]["value_m"]
    result = {
        "source_kind": "synthetic",
        "known_delta_m": delta_m,
        "reference_perimeter_m": truth,
        "reconstructed_m": value,
        "error_m": value - truth if value is not None else None,
        "physical_accuracy_validated": False,
        "note": "Known rigid poses and metric scale; not a test of human pose or camera calibration estimation.",
    }
    write_json(root / "report.json", result)
    html_report(root, result)
    return result
