"""Rebuild research contour topology from model-validated synthetic pose caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(
    source_cache,
    output,
    research,
    selection_predictions=None,
    *,
    reference_fit=None,
    case_names=None,
):
    from .capture_source import resolve_source, supported_observations
    from .contours import contour_observation
    from .local_shape import torso_basis, transport_basis
    from .research import options

    source_cache = Path(source_cache).resolve()
    manifest = read_json(source_cache / "cache-manifest.json")
    if selection_predictions is not None and reference_fit is not None:
        raise InputError("Reference geometry and iterative fit refresh are distinct")
    chosen = set(case_names) if case_names is not None else None
    if chosen is not None and (
        not chosen or not chosen <= {r["name"] for r in manifest["cases"]}
    ):
        raise InputError("Unknown reference cache cases")
    research = options(research)
    if (
        not supported_observations(manifest)
        or manifest.get("native_mhr_rotation_convention") != "xyz"
    ):
        raise InputError(
            "Contour rebuilding requires model-validated native pose caches"
        )
    old_fit = manifest["observation_options"]["fit"]
    for key in (
        "pose_mode",
        "root_mode",
        "initialization",
        "shared_skeleton",
        "mask_refinement",
        "anchor_source",
        "heading_smoothness",
        "heading_model",
    ):
        if research["fit"].get(key) != old_fit.get(key):
            raise InputError(
                "Contour rebuilding cannot alter frozen pose or observed image anchors"
            )
    context = resolve_source(manifest)
    profile = context.profile
    root = new_run(output)
    selections = {}
    if selection_predictions is not None:
        selection_root = Path(selection_predictions).resolve()
        predictions = read_json(selection_root / "predictions.json")
        previous_options = options(read_json(selection_root / "research-options.json"))
        expected_options = {**research, "fit": dict(research["fit"])}
        expected_options["fit"].pop("refresh_contour", None)
        if (
            not research["fit"].get("refresh_contour")
            or previous_options != expected_options
            or predictions.get("source_kind") != manifest["source_kind"]
            or predictions["cache_manifest_sha256"]
            != sha256(source_cache / "cache-manifest.json")
        ):
            raise InputError("Contour refresh requires the matching prior frozen fit")
        selections = {r["name"]: r for r in predictions["cases"]}
    if reference_fit is not None:
        reference_root = Path(reference_fit).resolve()
        predictions = read_json(reference_root / "predictions.json")
        previous_options = options(read_json(reference_root / "research-options.json"))
        if (
            previous_options != research
            or predictions.get("source_kind") != manifest["source_kind"]
            or predictions.get("cache_manifest_sha256")
            != sha256(source_cache / "cache-manifest.json")
        ):
            raise InputError("Reference template requires its original frozen fit")
        selections = {r["name"]: r for r in predictions["cases"]}
        if not (chosen or {r["name"] for r in manifest["cases"]}) <= selections.keys():
            raise InputError("Reference fit is missing required cases")
    result = {
        **manifest,
        "cases": [],
        "observation_options": research,
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
        "canonical_transport_validated": True,
    }
    if selection_predictions is not None:
        result["selection_predictions_sha256"] = sha256(
            selection_root / "predictions.json"
        )
    if reference_fit is not None:
        result["reference_geometry"] = {
            "purpose": "reference_template_geometry",
            "prediction_sha256": sha256(reference_root / "predictions.json"),
            "parent_cache_manifest_sha256": sha256(
                source_cache / "cache-manifest.json"
            ),
            "observed_images_unchanged": True,
        }
    for row in manifest["cases"]:
        if chosen is not None and row["name"] not in chosen:
            continue
        src = source_cache / row["cache"]
        if (
            src.parent != source_cache
            or sha256(src) != row["cache_sha256"]
            or max(
                row["rigid_transport_max_vertex_model_mm"],
                row["posed_shape_probe_max_model_mm"],
            )
            > 0.005
        ):
            raise InputError("Model validation or frozen cache changed")
        with np.load(src, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files}
        canonical = {
            mode: torso_basis(data["rest_base"], data["rest_anchors"], mode)
            for mode in ("radial", "independent")
        }
        selected = None
        if selection_predictions is not None or reference_fit is not None:
            selected = selections[row["name"]]
            if selected["frame_indices"] != data["frames"].tolist():
                raise InputError("Refreshed fit has a different frame sequence")
            selected = selected["fit"]
        template_local = None
        if selected is not None and research["fit"].get("local_mode", "none") != "none":
            if research["fit"].get("local_transport") != "canonical":
                raise InputError("Cached local refresh requires canonical transport")
            template_local = transport_basis(
                data["rest_basis"],
                data["template_basis"],
                canonical[research["fit"]["local_mode"]],
            )
        obs = []
        for i in range(len(data["shared_base"])):
            source_order = int(data["shared_source_order"][i])
            rotation = data["frame_rotations"][source_order]
            base = (
                (data["template_base"] - data["template_root"]) @ rotation.T
                + data["frame_root_positions"][source_order]
            ) * [1, -1, -1]
            basis = (
                np.einsum("ij,njd->nid", rotation, data["template_basis"])
                * np.array([1, -1, -1])[None, :, None]
            )

            def generate(shape):
                delta = np.asarray(shape) - data["initial"]
                nonzero = np.flatnonzero(delta)
                if not len(nonzero):
                    return base
                if len(nonzero) == 1:
                    return base + basis[:, :, nonzero[0]] * delta[nonzero[0]]
                return base + basis @ delta

            def anchors(shape):
                return data["shared_anchor_base"][i] + data["shared_anchor_basis"][
                    i
                ] @ (np.asarray(shape) - data["initial"])

            height, width = map(int, data["shared_center"][i][::-1] * 2)
            selection_vertices, selection_camera = None, None
            if selected is not None:
                coefficients = np.asarray(selected["shape_delta"])
                selection_vertices = generate(
                    data["initial"] + coefficients[: len(data["initial"])]
                ).copy()
                if template_local is not None:
                    local = (
                        np.einsum("ij,njd->nid", rotation, template_local)
                        * np.array([1, -1, -1])[None, :, None]
                    )
                    selection_vertices += local @ coefficients[len(data["initial"]) :]
                angle = float(
                    np.deg2rad(
                        selected.get("yaw_delta_deg", np.zeros(len(data["frames"])))[i]
                    )
                )
                if angle:
                    c, s = np.cos(angle), np.sin(angle)
                    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                    pivot = data["shared_anchor_base"][i].mean(0)
                    selection_vertices = (selection_vertices - pivot) @ yaw.T + pivot
                selection_camera = (
                    data["shared_camera"][i] + np.asarray(selected["camera_delta"])[i]
                )
            o = contour_observation(
                np.zeros((height, width), np.uint8),
                data["faces"],
                generate,
                data["initial"],
                data["shared_camera"][i],
                float(data["shared_focal"][i]),
                region_vertex_indices=profile["fit_vertex_indices"],
                region_weight=research.get("region_weight", 1),
                landmark_generate=anchors,
                landmark_target=data["shared_anchor_target"][i],
                observed_contour=data["shared_target"][i],
                observed_height=float(data["shared_mask_height"][i]),
                primitive_mode=research["fit"].get("contour_primitive", "vertex"),
                samples=research["fit"].get("contour_samples", 256),
                contour_order="arc"
                if research["fit"].get("distance_mode") == "segments"
                else "vertex_id",
                selection_vertices=selection_vertices,
                selection_camera=selection_camera,
            )
            o["source_order"] = source_order
            if "shared_anchor_weights" in data:
                o["anchor_weights"] = data["shared_anchor_weights"][i]
            ids = o["vertex_indices"]
            for mode in canonical:
                if "edge_vertex_indices" in o:
                    edge = o["edge_vertex_indices"]
                    fields = transport_basis(
                        data["rest_basis"][edge].reshape(-1, 3, 45),
                        o["endpoint_basis"].reshape(-1, 3, 45),
                        canonical[mode][edge].reshape(-1, 3, canonical[mode].shape[-1]),
                    ).reshape(len(edge), 2, 3, -1)
                    o["canonical_local_" + mode] = np.einsum(
                        "ne,necd->ncd", o["edge_weights"], fields
                    )
                else:
                    o["canonical_local_" + mode] = transport_basis(
                        data["rest_basis"][ids], o["basis"], canonical[mode][ids]
                    )
            o.pop("endpoint_basis", None)
            obs.append(o)
        rebuilt = {k: v for k, v in data.items() if not k.startswith("shared_")}
        for key in obs[0]:
            rebuilt["shared_" + key] = np.asarray([o[key] for o in obs])
        if reference_fit is not None:
            for key in (
                "shared_target",
                "shared_mask_height",
                "shared_source_order",
                "shared_camera",
                "shared_anchor_target",
                "shared_anchor_base",
                "shared_anchor_basis",
                "shared_center",
                "shared_focal",
            ):
                if not np.array_equal(rebuilt[key], data[key]):
                    raise InputError(
                        "Reference contour selection changed frozen image or camera observations"
                    )
        dest = root / row["cache"]
        np.savez_compressed(dest, **rebuilt)
        dest.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(dest),
                "source_cache_sha256": row["cache_sha256"],
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print("contour_cached", row["name"], flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--options", required=True)
    a = p.parse_args()
    cache(a.source, a.out, read_json(a.options))
