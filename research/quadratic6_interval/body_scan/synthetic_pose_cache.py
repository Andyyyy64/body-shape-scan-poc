"""Rebuild fixed synthetic observations with validated native-MHR pose transport."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(config_path, source_cache, output, research, names=None):
    import fcntl

    import numpy as np
    import psutil
    import roma
    import torch

    from .capture_source import resolve_source, supported_observations
    from .fit_strategy import build_observations
    from .pose_constraints import constrain_root_tilt, root_tilt_reference
    from .research import options
    from .synthetic_lab import snapshot_sources

    config = read_json(config_path)
    research = options(research)
    strategy = research.get("fit", {})
    if (
        strategy.get("pose_mode") != "profile"
        or strategy.get("root_mode") != "profile_tilt_xyz"
        or strategy.get("initialization") != "profile"
        or not strategy.get("shared_skeleton")
        or strategy.get("anchor_source", "sam_projection") != "sam_projection"
    ):
        raise InputError(
            "Pose cache requires the native xyz profile strategy and SAM anchors"
        )
    source_cache = Path(source_cache).resolve()
    manifest = read_json(source_cache / "cache-manifest.json")
    if not supported_observations(manifest):
        raise InputError("Pose cache requires synthetic observations")
    if (
        manifest.get("observation_variant") != "observed_rgb_grabcut.v1"
        and (manifest.get("observation_options") or {})
        .get("fit", {})
        .get("mask_refinement")
        != "grabcut"
    ):
        raise InputError("Expected frozen observed RGB-refined contours")
    context = resolve_source(manifest, config_path)
    rows = context.rows
    chosen = [r for r in manifest["cases"] if names is None or r["name"] in names]
    if (
        not chosen
        or not context.is_reference(rows[chosen[0]["name"]])
        or (names and set(names) != {r["name"] for r in chosen})
    ):
        raise InputError("A complete observed baseline must be first")
    profile = context.profile
    scales = np.asarray(profile["skeleton_scales"], np.float32)
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
    for key in ("checkpoint", "mhr_model"):
        if sha256(Path(config[key])) != config[key + "_sha256"]:
            raise InputError("MHR/checkpoint changed")
    state = torch.load(
        config["checkpoint"], map_location="cpu", mmap=True, weights_only=False
    )
    state = state.get("state_dict", state)
    mapping = state["head_pose.keypoint_mapping"].numpy()[:11]
    faces = state["head_pose.faces"].numpy()
    model = torch.jit.load(config["mhr_model"], map_location="cpu")

    def native(shapes, parameters):
        with torch.no_grad():
            v, s = model(
                torch.tensor(np.asarray(shapes, np.float32)),
                torch.tensor(np.asarray(parameters, np.float32)),
                torch.zeros(len(shapes), 72),
            )
        return v.numpy() / 100, s.numpy()

    root = new_run(output)
    snapshot_sources(root, config_path)
    result = {
        **manifest,
        "cases": [],
        "observation_options": research,
        "native_mhr_rotation_convention": "xyz",
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
    }
    initial, pose_reference, root_reference, template = None, None, None, None
    for row in chosen:
        file = source_cache / row["cache"]
        if file.parent != source_cache or sha256(file) != row["cache_sha256"]:
            raise InputError("Frozen contour cache changed")
        with np.load(file, allow_pickle=False) as n:
            old = {k: n[k] for k in n.files}
        if not np.array_equal(
            old["shared_source_order"], np.arange(len(old["frames"]))
        ):
            raise InputError("Pose cache expects complete ordered frozen frames")
        session = context.store / rows[row["name"]]["session_id"]
        if sha256(session / "capture.bin") != row["video_sha256"]:
            raise InputError("Input capture changed")
        attempts = list(session.glob("attempt_*/frames.json"))
        if len(attempts) != 1:
            raise InputError("Expected one source reconstruction")
        frame_rows = {r["frame"]: r for r in read_json(attempts[0])["frames"]}
        values = []
        for index in old["frames"]:
            p = attempts[0].parent / frame_rows[int(index)]["output"]
            if sha256(p) != row["frame_estimate_sha256"][str(int(index))]:
                raise InputError("SAM estimate changed")
            with np.load(p, allow_pickle=False) as n:
                values.append({k: n[k] for k in n.files})
            if (
                strategy.get("heading_smoothness", 0)
                or strategy.get("heading_model", "acceleration") != "acceleration"
            ):
                values[-1]["frame_index"] = np.asarray(index, np.int64)
        parameters = np.asarray([v["mhr_model_params"] for v in values], np.float32)
        if initial is None:
            initial = old["initial"].copy()
            pose_reference = np.median(parameters[:, 6:136], axis=0)
            root_reference = root_tilt_reference(parameters)
            base_params = np.zeros(204, np.float32)
            base_params[6:136] = pose_reference
            base_params[136:] = scales
            basis_shapes = np.vstack([initial, initial + np.eye(45, dtype=np.float32)])
            tv, ts = native(basis_shapes, np.repeat(base_params[None], 46, 0))
            tj = mapping @ np.concatenate([tv, ts[:, :, :3] / 100], 1)
            template = (
                tv[0],
                np.moveaxis(tv[1:] - tv[0], 0, -1),
                tj[0],
                np.moveaxis(tj[1:] - tj[0], 0, -1),
                ts[0, 1, :3] / 100,
                roma.unitquat_to_rotmat(torch.tensor(ts[0, 1, 3:7])).numpy(),
            )
            rest_params = base_params.copy()
            rest_params[:136] = 0
            rv, rs = native(basis_shapes, np.repeat(rest_params[None], 46, 0))
            rj = mapping @ np.concatenate([rv, rs[:, :, :3] / 100], 1)
            rest = (
                rv[0],
                np.moveaxis(rv[1:] - rv[0], 0, -1),
                rj[0, [5, 6, 9, 10]],
                np.moveaxis(rj[1:, [5, 6, 9, 10]] - rj[0, [5, 6, 9, 10]], 0, -1),
            )
        parameters[:, 6:136] = pose_reference
        parameters[:, 136:] = scales
        if (
            strategy.get("heading_smoothness", 0)
            or strategy.get("heading_model", "acceleration") != "acceleration"
        ):
            from .pose_constraints import constrain_root_sequence

            parameters = constrain_root_sequence(
                parameters,
                root_reference,
                strategy.get("heading_smoothness", 0),
                old["frames"],
                model=strategy.get("heading_model", "acceleration"),
            )
        else:
            parameters = np.asarray(
                [constrain_root_tilt(p, root_reference) for p in parameters]
            )
        actual_v, actual_s = native(
            np.repeat(initial[None], len(parameters), 0), parameters
        )
        actual_j = mapping @ np.concatenate([actual_v, actual_s[:, :, :3] / 100], 1)
        rotations = (
            roma.unitquat_to_rotmat(torch.tensor(actual_s[:, 1, 3:7])).numpy()
            @ template[5].T
        )
        positions = actual_s[:, 1, :3] / 100
        generated = (
            np.einsum("fij,nj->fni", rotations, template[0] - template[4])
            + positions[:, None]
        )
        generated_j = (
            np.einsum("fij,nj->fni", rotations, template[2] - template[4])
            + positions[:, None]
        )
        vertex_error = float(np.max(abs(generated - actual_v)))
        joint_error = float(
            np.max(abs(generated_j[:, [5, 6, 9, 10]] - actual_j[:, [5, 6, 9, 10]]))
        )
        if max(vertex_error, joint_error) > 5e-6:
            raise InputError("MHR root joint 1 rigid transport failed validation")
        lookup = {p.tobytes(): i for i, p in enumerate(parameters)}

        def many(shapes, params):
            if params.tobytes() not in lookup:
                v, s = native(shapes, np.repeat(params[None], len(shapes), 0))
                return v, mapping @ np.concatenate([v, s[:, :, :3] / 100], 1)
            i = lookup[params.tobytes()]
            delta = np.asarray(shapes) - initial

            def linear(base, basis):
                output = []
                for difference in delta:
                    active = np.flatnonzero(difference)
                    if len(active) == 0:
                        output.append(base)
                    elif len(active) == 1:
                        output.append(
                            base + basis[:, :, active[0]] * difference[active[0]]
                        )
                    else:
                        output.append(base + basis @ difference)
                return np.asarray(output)

            v = linear(template[0], template[1])
            j = linear(template[2], template[3])
            return (v - template[4]) @ rotations[i].T + positions[i], (
                j - template[4]
            ) @ rotations[i].T + positions[i]

        def generate(shape, params):
            v, j = many(np.asarray(shape)[None], params)
            return v[0], j[0]

        generate.batch = many
        probe = initial + np.linspace(-0.1, 0.1, 45).astype(np.float32)
        pv, ps = native(probe[None], parameters[:1])
        proxy = many(probe[None], parameters[0])[0]
        probe_error = float(np.max(abs(pv - proxy)))
        if probe_error > 5e-6:
            raise InputError("MHR posed shape transport failed validation")
        estimates = []
        overrides = []
        height, width = map(int, old["shared_center"][0][::-1] * 2)
        for i, value in enumerate(values):
            estimates.append(
                (float(old["yaw"][i]), value, np.zeros((height, width), np.uint8))
            )
            overrides.append(
                {
                    "observed_contour": old["shared_target"][i],
                    "observed_height": float(old["shared_mask_height"][i]),
                }
            )
        observations = build_observations(
            estimates,
            initial,
            scales,
            faces,
            generate,
            profile["fit_vertex_indices"],
            research,
            pose_reference=pose_reference,
            root_reference=root_reference,
            observed_overrides=overrides,
        )
        data = {
            k: v for k, v in old.items() if not k.startswith(("native_", "shared_"))
        }
        data.update(
            initial=initial,
            rest_base=rest[0],
            rest_basis=rest[1],
            rest_anchors=rest[2],
            rest_anchor_basis=rest[3],
            pose_reference=pose_reference,
            root_tilt_reference=root_reference,
            template_base=template[0],
            template_basis=template[1],
            template_root=template[4],
            frame_rotations=rotations,
            frame_root_positions=positions,
        )
        for key in observations[0]:
            data["shared_" + key] = np.asarray([o[key] for o in observations])
        out = root / row["cache"]
        np.savez_compressed(out, **data)
        out.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(out),
                "source_cache_sha256": row["cache_sha256"],
                "rigid_transport_max_vertex_model_mm": vertex_error * 1000,
                "rigid_transport_max_anchor_model_mm": joint_error * 1000,
                "posed_shape_probe_max_model_mm": probe_error * 1000,
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print(
            "native_pose_cached",
            row["name"],
            "max_mm",
            max(vertex_error, joint_error, probe_error) * 1000,
            flush=True,
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--options", required=True)
    parser.add_argument("--cases", nargs="+")
    a = parser.parse_args()
    cache(a.runtime_config, a.source, a.out, read_json(a.options), a.cases)
