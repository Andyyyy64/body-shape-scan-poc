"""Compose smoothed estimated headings into validated synthetic rigid pose caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .io import InputError, new_run, read_json, replace_json, sha256


def cache(source_cache, output, strength, model="acceleration"):
    from .pose_constraints import harmonic_bias_headings, root_rotation, smooth_headings

    source_cache = Path(source_cache).resolve()
    m = read_json(source_cache / "cache-manifest.json")
    if (
        m.get("source_kind") != "synthetic"
        or m.get("diagnostic_only") is not True
        or m.get("native_mhr_rotation_convention") != "xyz"
        or m["observation_options"]["fit"].get("root_mode") != "profile_tilt_xyz"
    ):
        raise InputError(
            "Heading composition requires a verified native standing-axis cache"
        )
    root = new_run(output)
    source = Path(m["source"])
    if (
        sha256(source / "report.json") != m["source_report_sha256"]
        or sha256(source / "store/profile.json") != m["profile_sha256"]
    ):
        raise InputError("Source estimates changed")
    source_rows = {r["name"]: r for r in read_json(source / "report.json")["results"]}
    prior_strength = m["observation_options"]["fit"].get("heading_smoothness", 0)
    prior_model = m["observation_options"]["fit"].get("heading_model", "acceleration")
    if model not in ("acceleration", "harmonic_bias"):
        raise InputError("Invalid heading model")
    result = {
        **m,
        "cases": [],
        "source_cache_manifest_sha256": sha256(source_cache / "cache-manifest.json"),
        "derived_heading_transport": True,
        "heading_validation_scope": "Parent MHR rigid map validated; new headings composed. Requires final native pipeline verification.",
    }
    result["observation_options"]["fit"]["heading_smoothness"] = strength
    if model != "acceleration":
        result["observation_options"]["fit"]["heading_model"] = model
        result["motion_assumption"] = (
            "Constant angular velocity; rate and phase estimated from SAM; periodic angle bias removed"
        )
    else:
        result["observation_options"]["fit"].pop("heading_model", None)
    for row in m["cases"]:
        src = source_cache / row["cache"]
        if src.parent != source_cache or sha256(src) != row["cache_sha256"]:
            raise InputError("Frozen observations changed")
        with np.load(src, allow_pickle=False) as n:
            data = {k: n[k] for k in n.files}
        reference = data["root_tilt_reference"]
        old = data["frame_rotations"].copy()
        local = reference.T @ old
        heading = np.unwrap(
            np.arctan2(local[:, 0, 2] - local[:, 2, 0], local[:, 0, 0] + local[:, 2, 2])
        )
        raw_heading = heading
        if prior_strength or prior_model != "acceleration":
            session = source / "store" / source_rows[row["name"]]["session_id"]
            attempts = list(session.glob("attempt_*/frames.json"))
            if len(attempts) != 1:
                raise InputError("Expected one frozen SAM inference")
            frames = {r["frame"]: r for r in read_json(attempts[0])["frames"]}
            parameters = []
            for index in data["frames"]:
                file = attempts[0].parent / frames[int(index)]["output"]
                if sha256(file) != row["frame_estimate_sha256"][str(int(index))]:
                    raise InputError("Frozen SAM pose changed")
                with np.load(file, allow_pickle=False) as n:
                    parameters.append(n["mhr_model_params"])
            raw = reference.T @ root_rotation(parameters)
            raw_heading = np.unwrap(
                np.arctan2(raw[:, 0, 2] - raw[:, 2, 0], raw[:, 0, 0] + raw[:, 2, 2])
            )

        def rotations(h):
            c, s = np.cos(h), np.sin(h)
            r = np.zeros((len(h), 3, 3), np.float32)
            r[:, 0, 0] = r[:, 2, 2] = c
            r[:, 0, 2] = s
            r[:, 2, 0] = -s
            r[:, 1, 1] = 1
            return reference @ r

        if np.max(abs(rotations(heading) - old)) > 1e-5:
            raise InputError(
                "Native cache is not a pure standing-axis heading sequence"
            )
        smoothed = (
            harmonic_bias_headings(raw_heading, data["frames"])
            if model == "harmonic_bias"
            else smooth_headings(raw_heading, strength, data["frames"])
        )
        new = rotations(smoothed)
        update = new @ old.transpose(0, 2, 1)
        sign = np.array([1, -1, -1])
        data["frame_rotations"] = new
        for i in range(len(data["shared_base"])):
            order = int(data["shared_source_order"][i])
            pivot = data["frame_root_positions"][order]
            r = update[order]
            for key in ["shared_base", "shared_anchor_base"]:
                data[key][i] = ((data[key][i] * sign - pivot) @ r.T + pivot) * sign
            for key in ["shared_basis", "shared_anchor_basis"]:
                data[key][i] = (
                    np.einsum("ij,njd->nid", r, data[key][i] * sign[None, :, None])
                    * sign[None, :, None]
                )
        for key in list(data):
            if key.startswith("shared_canonical_local_"):
                del data[key]
        data["estimated_heading_raw"] = raw_heading
        data["estimated_heading_smoothed"] = smoothed
        dest = root / row["cache"]
        np.savez_compressed(dest, **data)
        dest.chmod(0o600)
        result["cases"].append(
            {
                **row,
                "cache_sha256": sha256(dest),
                "source_cache_sha256": row["cache_sha256"],
                "estimated_heading_adjustment_rms_deg": float(
                    np.sqrt(np.mean((smoothed - heading) ** 2)) * 180 / np.pi
                ),
            }
        )
        replace_json(root / "cache-manifest.json", result)
        print("heading_cached", row["name"], flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--strength", type=float, required=True)
    a = p.parse_args()
    cache(a.source, a.out, a.strength)
