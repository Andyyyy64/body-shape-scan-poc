"""Oracle geometry diagnostics in an isolated synthetic-only analysis path."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import InputError, new_run, read_json, sha256, write_json


def capacity(cache_root, case, output, align_anatomy=False):
    """Fit true geometry to the estimated fixed-skeleton basis, outside inference."""
    import numpy as np

    from .mesh import section_perimeter
    from .synthetic_lab import deform

    cache_root = Path(cache_root).resolve()
    manifest = read_json(cache_root / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or manifest.get("diagnostic_only") is not True
    ):
        raise InputError("Capacity diagnostics requires a synthetic observation cache")
    source = Path(manifest["source"])
    if sha256(source / "report.json") != manifest["source_report_sha256"]:
        raise InputError("Source report changed")
    if sha256(source / "store" / "profile.json") != manifest["profile_sha256"]:
        raise InputError("Source measurement definition changed")
    body = read_json(source / "body.json")
    if (
        body.get("source_kind") != "synthetic"
        or sha256(source / "body.npz") != body["body_sha256"]
    ):
        raise InputError("Oracle source provenance failed")
    row = next((r for r in manifest["cases"] if r["name"] == case), None)
    if row is None:
        raise InputError("Unknown diagnostic case")
    file = cache_root / row["cache"]
    if file.parent != cache_root or sha256(file) != row["cache_sha256"]:
        raise InputError("Observation cache changed")
    with np.load(file, allow_pickle=False) as n:
        data = {k: n[k] for k in n.files}
    with np.load(source / "body.npz", allow_pickle=False) as n:
        truth_vertices, faces = n["vertices"], n["faces"]
    if len(truth_vertices) != len(data["rest_base"]) or not np.array_equal(
        faces, data["faces"]
    ):
        raise InputError("Oracle and estimator need corresponding topology")
    body["faces"] = faces
    section = {
        "y": float(data["section_y"]),
        "center_xz": data["section_center"].tolist(),
    }

    def register(vertices):
        if not align_anatomy:
            return vertices
        if "rest_anchors" not in data:
            raise InputError("Anatomical registration requires estimated rest anchors")
        anchors = data["rest_anchors"]
        hip, shoulder = anchors[2:].mean(0), anchors[:2].mean(0)
        result = vertices.copy()
        result[:, 1] = hip[1] + (vertices[:, 1] - body["hip_y"]) * (
            shoulder[1] - hip[1]
        ) / (body["shoulder_y"] - body["hip_y"])
        result[:, [0, 2]] += hip[[0, 2]] - np.asarray(body["center_xz"])
        return result

    def perimeter(vertices):
        result = section_perimeter(
            {"vertices": vertices.tolist(), "faces": faces.tolist()}, section
        )
        if result is None:
            raise InputError("Ambiguous diagnostic measurement section")
        return result

    root = new_run(output)
    base, basis = data["rest_base"], data["rest_basis"]
    registered = register(truth_vertices)
    torso = (
        (truth_vertices[:, 1] >= body["hip_y"])
        & (truth_vertices[:, 1] <= body["shoulder_y"])
        & (abs(truth_vertices[:, 0]) < 0.25)
    )
    regions = {
        "whole_mesh": np.ones(len(base), bool),
        "torso": torso,
        "abdomen_band": torso & (abs(registered[:, 1] - section["y"]) < 0.12),
    }
    rows = []
    for region, selected in regions.items():
        matrix = basis[selected].reshape(-1, basis.shape[-1]).astype(float)
        previous = None
        for delta in (0, -0.01, 0.01, -0.02, 0.02, 0.04):
            target, _, _ = deform(truth_vertices, body, delta)
            target = register(target)
            coefficient, _, rank, singular = np.linalg.lstsq(
                matrix, (target - base)[selected].reshape(-1), rcond=1e-6
            )
            fitted = base + basis @ coefficient
            measured = perimeter(fitted)
            truth = perimeter(target)
            if delta == 0:
                previous = measured, truth
            predicted_delta, truth_delta = measured - previous[0], truth - previous[1]
            rows.append(
                {
                    "region": region,
                    "declared_generator_delta_model_mm": delta * 1000,
                    "truth_delta_at_profile_plane_model_mm": truth_delta * 1000,
                    "fitted_delta_model_mm": predicted_delta * 1000,
                    "response_gain": predicted_delta / truth_delta
                    if abs(truth_delta) > 1e-9
                    else None,
                    "surface_rms_model_mm": float(
                        np.sqrt(np.mean((fitted[selected] - target[selected]) ** 2))
                        * 1000
                    ),
                    "coefficient_norm": float(np.linalg.norm(coefficient)),
                    "max_abs_coefficient": float(np.max(abs(coefficient))),
                    "basis_rank": int(rank),
                    "smallest_retained_singular_value": float(singular[int(rank) - 1])
                    if rank
                    else None,
                }
            )
    result = {
        "source_kind": "synthetic",
        "diagnostic_only_oracle": True,
        "physical_accuracy_validated": False,
        "cache_manifest_sha256": sha256(cache_root / "cache-manifest.json"),
        "cache_sha256": row["cache_sha256"],
        "body_sha256": body["body_sha256"],
        "analysis_implementation_sha256": sha256(Path(__file__)),
        "section_definition": section,
        "anatomical_y_registration": align_anatomy,
        "registration_is_diagnostic_only": True,
        "rows": rows,
        "interpretation": "True vertices used only in this diagnostic. Region-dependent least squares with estimated fixed skeleton; coefficient inflation and skeleton errors can affect results. Not a global capacity ceiling or image-attainable precision proof.",
    }
    write_json(root / "capacity.json", result)
    return result


def model_recovery(cache_root, case, output, research):
    """Exercise the actual solver on known in-model changes and ideal observations."""
    import numpy as np
    import torch

    from .fit_strategy import solve
    from .mesh import section_perimeter
    from .research import options

    torch.set_num_threads(2)
    research = options(research)
    cache_root = Path(cache_root).resolve()
    manifest = read_json(cache_root / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or manifest.get("diagnostic_only") is not True
    ):
        raise InputError("Model recovery requires a synthetic observation cache")
    row = next((r for r in manifest["cases"] if r["name"] == case), None)
    if row is None:
        raise InputError("Unknown diagnostic case")
    file = cache_root / row["cache"]
    if file.parent != cache_root or sha256(file) != row["cache_sha256"]:
        raise InputError("Observation cache changed")
    with np.load(file, allow_pickle=False) as n:
        data = {k: n[k] for k in n.files}
    section = {
        "y": float(data["section_y"]),
        "center_xz": data["section_center"].tolist(),
    }

    def perimeter(delta):
        v = data["rest_base"] + data["rest_basis"] @ delta
        value = section_perimeter(
            {"vertices": v.tolist(), "faces": data["faces"].tolist()}, section
        )
        if value is None:
            raise InputError("Ambiguous in-model section")
        return value

    # Choose a visible radial direction from the estimated basis, never truth.
    vertices = data["rest_base"]
    band = abs(vertices[:, 1] - section["y"]) < 0.04
    radial = vertices[:, [0, 2]] - np.asarray(section["center_xz"])
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-8)
    scores = np.einsum(
        "nc,ncd->d", radial[band], data["rest_basis"][band][:, [0, 2], :]
    )
    dimension = int(abs(scores).argmax())
    zero = np.zeros(data["rest_basis"].shape[-1], np.float32)
    original = perimeter(zero)
    probe = zero.copy()
    probe[dimension] = 0.01
    slope = (perimeter(probe) - original) / 0.01
    if abs(slope) < 1e-6:
        raise InputError("Selected basis has no measurable section response")
    label = "shared" if research.get("fit", {}).get("shared_skeleton") else "native"
    keys = [k.removeprefix(label + "_") for k in data if k.startswith(label + "_")]

    def project(v, index):
        p = v + data[label + "_camera"][index]
        return (
            p[:, :2] / p[:, 2:] * data[label + "_focal"][index]
            + data[label + "_center"][index]
        )

    root = new_run(output)
    results = []
    for millimetres in (0, 10, -10):
        known = zero.copy()
        known[dimension] = millimetres / 1000 / slope
        observations = []
        for index in range(len(data[label + "_base"])):
            o = {key: data[label + "_" + key][index] for key in keys}
            o["target"] = project(o["base"] + o["basis"] @ known, index)
            if "base_weight" in o:
                o["target_weight"] = o["base_weight"].copy()
            if "anchor_base" in o:
                o["anchor_target"] = project(
                    o["anchor_base"] + o["anchor_basis"] @ known, index
                )
            observations.append(o)
        fit = solve(observations, data["shape_scale"], research)
        truth_delta = perimeter(known) - original
        predicted = perimeter(fit["shape_delta"]) - original
        results.append(
            {
                "target_delta_model_mm": truth_delta * 1000,
                "predicted_delta_model_mm": predicted * 1000,
                "gain": predicted / truth_delta if abs(truth_delta) > 1e-9 else None,
                "known_shape_coefficient": float(known[dimension]),
                "fit": {
                    k: v.tolist() if hasattr(v, "tolist") else v for k, v in fit.items()
                },
            }
        )
        print("ideal_model", millimetres, predicted * 1000, flush=True)
    result = {
        "source_kind": "synthetic",
        "diagnostic_only_oracle": True,
        "physical_accuracy_validated": False,
        "ideal_float_boundary_and_landmarks": True,
        "selected_shape_dimension": dimension,
        "cache_sha256": row["cache_sha256"],
        "research_options": research,
        "rows": results,
        "interpretation": "In-model recovery through the actual solver; ideal correspondences, fixed contour vertices and estimated camera. No real-mask or population precision proof.",
    }
    write_json(root / "model-recovery.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode", choices=("capacity", "model_recovery"), default="capacity"
    )
    parser.add_argument("--options")
    parser.add_argument("--align-anatomy", action="store_true")
    args = parser.parse_args()
    if args.mode == "model_recovery":
        if not args.options:
            parser.error("--options is required for model recovery")
        model_recovery(args.cache, args.case, args.out, read_json(args.options))
    else:
        result = capacity(args.cache, args.case, args.out, args.align_anatomy)
        for row in result["rows"]:
            print(
                row["region"],
                row["declared_generator_delta_model_mm"],
                row["response_gain"],
            )
