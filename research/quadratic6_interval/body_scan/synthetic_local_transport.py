"""Validate canonical residual transport using frozen estimated MHR observations."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(config_path, source_cache, output):
    import fcntl

    import numpy as np
    import psutil
    import torch
    from scipy.spatial import cKDTree

    from .local_shape import torso_basis, transport_basis
    from .pose_constraints import constrain_root_tilt
    from .synthetic_lab import snapshot_sources

    config = read_json(config_path)
    source_cache = Path(source_cache).resolve()
    manifest = read_json(source_cache / "cache-manifest.json")
    if (
        manifest.get("source_kind") != "synthetic"
        or manifest.get("diagnostic_only") is not True
    ):
        raise InputError("Transport cache requires synthetic observations")
    observed = (manifest.get("observation_options") or {}).get("fit", {})
    if not observed.get("shared_skeleton") or observed.get("pose_mode") != "profile":
        raise InputError("Transport cache requires frozen profile pose and skeleton")
    source = Path(manifest["source"])
    if (
        sha256(source / "report.json") != manifest["source_report_sha256"]
        or sha256(source / "store/profile.json") != manifest["profile_sha256"]
    ):
        raise InputError("Frozen source changed")
    profile = read_json(source / "store/profile.json")
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > config["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < config["min_available_mb"] * 1024**2
            ):
                os._exit(75)
            time.sleep(0.25)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    if sha256(Path(config["mhr_model"])) != config["mhr_model_sha256"]:
        raise InputError("MHR model changed")
    model = torch.jit.load(config["mhr_model"], map_location="cpu")
    root = new_run(output)
    snapshot_sources(root, config_path)
    rows = {r["name"]: r for r in read_json(source / "report.json")["results"]}
    result = {
        **manifest,
        "cases": [],
        "canonical_transport_validated": True,
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
    }
    for row in manifest["cases"]:
        original = source_cache / row["cache"]
        if original.parent != source_cache or sha256(original) != row["cache_sha256"]:
            raise InputError("Frozen observations changed")
        with np.load(original, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files}
        session = source / "store" / rows[row["name"]]["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Source input changed")
        attempts = list(session.glob("attempt_*/frames.json"))
        if len(attempts) != 1:
            raise InputError("Expected one frozen source reconstruction")
        frames = {r["frame"]: r for r in read_json(attempts[0])["frames"]}
        canonical = {
            mode: torso_basis(data["rest_base"], data["rest_anchors"], mode)
            for mode in ("radial", "independent")
        }
        parameters = []
        for frame in data["frames"]:
            path = attempts[0].parent / frames[int(frame)]["output"]
            if sha256(path) != row["frame_estimate_sha256"][str(int(frame))]:
                raise InputError("Estimated pose changed")
            with np.load(path, allow_pickle=False) as n:
                params = n["mhr_model_params"].copy()
            params[136:] = profile["skeleton_scales"]
            params[6:136] = data["pose_reference"]
            if "root_tilt_reference" in data:
                params = constrain_root_tilt(params, data["root_tilt_reference"])
            parameters.append(params)
        if manifest.get("native_mhr_rotation_convention") == "xyz":
            if (
                observed.get("root_mode") != "profile_tilt_xyz"
                or max(
                    row["rigid_transport_max_vertex_model_mm"],
                    row["posed_shape_probe_max_model_mm"],
                )
                > 0.005
            ):
                raise InputError("Native transport lacks verified model agreement")
            full = (
                np.einsum(
                    "fij,nj->fni",
                    data["frame_rotations"],
                    data["template_base"] - data["template_root"],
                )
                + data["frame_root_positions"][:, None]
            ) * [1, -1, -1]
        else:
            full = []
            for start in range(0, len(parameters), 8):
                batch = np.asarray(parameters[start : start + 8], np.float32)
                with torch.no_grad():
                    vertices, _ = model(
                        torch.tensor(data["initial"])[None].repeat(len(batch), 1),
                        torch.tensor(batch),
                        torch.zeros(len(batch), 72),
                    )
                full.extend(vertices.numpy() / 100 * [1, -1, -1])
            full = np.asarray(full)
        transformed = {mode: [] for mode in canonical}
        differences, maximum_distance = [], 0.0
        ids = []
        for order, base in enumerate(data["shared_base"]):
            source_order = int(data["shared_source_order"][order])
            distance, index = cKDTree(full[source_order]).query(base)
            maximum_distance = max(maximum_distance, float(distance.max()))
            if distance.max() > 5e-6:
                raise InputError(
                    "Frozen contour vertices do not match reconstructed estimated pose"
                )
            ids.append(index)
            for mode in canonical:
                transported = transport_basis(
                    data["rest_basis"][index],
                    data["shared_basis"][order],
                    canonical[mode][index],
                )
                transformed[mode].append(transported)
                if mode == "independent":
                    old = torso_basis(base, data["shared_anchor_base"][order], mode)
                    differences.append(
                        float(np.sqrt(np.mean((transported - old) ** 2)) * 0.01 * 1000)
                    )
        data["shared_vertex_indices"] = np.asarray(ids)
        for mode, values in transformed.items():
            data["shared_canonical_local_" + mode] = np.asarray(values)
        dest = root / row["cache"]
        np.savez_compressed(dest, **data)
        dest.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(dest),
                "source_cache_sha256": row["cache_sha256"],
                "maximum_vertex_match_model_mm": maximum_distance * 1000,
                "independent_transport_vs_posed_rms_model_mm_for_one_percent": float(
                    np.mean(differences)
                ),
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print(
            "transported",
            row["name"],
            "match_mm",
            maximum_distance * 1000,
            "rms_1pct_mm",
            np.mean(differences),
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache(args.runtime_config, args.source, args.out)
