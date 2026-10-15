"""Bind explicit capture IDs to immutable native pipeline observations.

This input manifest carries no renderer labels or shape-change truth. It is an
input boundary for the experimental measurement path, not a precision verdict.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .io import InputError, read_json, relative_asset, sha256, write_json

SCHEMA = "capture_measurement_inputs.v1"
DIGEST = re.compile(r"[0-9a-f]{64}")


def _capture(root, identifier, profile, kind, engine):
    from .web import session_path

    session = session_path(root, identifier)
    meta = read_json(session / "session.json")
    status = read_json(session / "reconstruction.json")
    mesh = read_json(session / "mesh.json")
    video_hash = sha256(session / "capture.bin")
    method = profile["method"]
    if (
        meta.get("source_kind") != kind
        or status.get("state") != "complete"
        or status.get("engine_id") != engine
        or status.get("method") != method
        or mesh.get("source_kind") != kind
        or mesh.get("method") != method
        or mesh.get("capture_sha256") != video_hash
        or meta.get("video_sha256") != video_hash
        or mesh.get("reference_frame_id") != "mhr_rest_" + profile["id"]
        or mesh.get("scale_reference_id") != "model_skeleton_" + profile["id"]
        or mesh.get("sections") != profile["sections"]
    ):
        raise InputError("Capture, profile, engine or native method does not match")
    attempts = list(session.glob("attempt_*/frames.json"))
    if len(attempts) != 1:
        raise InputError("An unambiguous native inference attempt is required")
    attempt = attempts[0].parent
    fit = read_json(attempt / "fit.json")
    qc = read_json(attempt / "qc.json")
    implementation = fit.get("implementation_sha256")
    if (
        fit.get("source_kind") != kind
        or fit.get("engine_id") != engine
        or fit.get("method") != method
        or qc.get("source_kind") != kind
        or qc.get("method") != method
        or qc.get("research_options") != fit.get("research_options")
        or not isinstance(implementation, dict)
        or not {
            "pipeline.py",
            "fit_strategy.py",
            "shared_shape.py",
            "contours.py",
            "silhouette_edges.py",
        }
        <= implementation.keys()
        or any(
            not isinstance(v, str) or not DIGEST.fullmatch(v)
            for v in implementation.values()
        )
    ):
        raise InputError("Native fit/QC provenance is missing or inconsistent")
    outcomes = read_json(attempts[0])["frames"]
    indices = [r["frame"] for r in outcomes]
    if (
        not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or indices != sorted(set(indices))
    ):
        raise InputError("Native frame identities are invalid or duplicated")
    accepted = [r for r in outcomes if r["state"] == "estimated"]
    accepted_indices = [r["frame"] for r in accepted]
    fit_indices = fit.get("fit_source_frame_indices")
    if (
        not accepted
        or status.get("total_frames") != len(outcomes)
        or status.get("usable_frames") != len(accepted)
        or qc.get("usable_frames") != len(accepted)
        or type(qc.get("required_frames")) is not int
        or len(accepted) < qc["required_frames"]
        or not isinstance(fit_indices, list)
        or not fit_indices
        or any(type(i) is not int for i in fit_indices)
        or len(fit_indices) != len(set(fit_indices))
        or not set(fit_indices) <= set(accepted_indices)
        or mesh.get("quality", {}).get("usable_frames") != len(accepted)
    ):
        raise InputError("Native usable frames or fit correspondence is inconsistent")
    frames = []
    for row in accepted:
        file = relative_asset(attempt, row["output"])
        frames.append(
            {
                "frame": row["frame"],
                "output": str(file.relative_to(root)),
                "estimate_sha256": sha256(file),
                "yaw_rad": row["yaw_rad"],
            }
        )
    return {
        "session_id": identifier,
        "video_sha256": video_hash,
        "session_sha256": sha256(session / "session.json"),
        "status_sha256": sha256(session / "reconstruction.json"),
        "mesh_sha256": sha256(session / "mesh.json"),
        "attempt": str(attempt.relative_to(root)),
        "frames_sha256": sha256(attempts[0]),
        "fit_sha256": sha256(attempt / "fit.json"),
        "qc_sha256": sha256(attempt / "qc.json"),
        "native_implementation_sha256": implementation,
        "native_research_options": fit.get("research_options"),
        "fit_source_frame_indices": fit_indices,
        "frames": frames,
    }


def build(store, reference, current, runtime_config):
    """Read native artifacts directly; all capture identities are caller supplied."""
    root = Path(store).resolve()
    marker = read_json(root / "store.json")
    profile = read_json(root / "profile.json")
    cfg = read_json(runtime_config)
    kind = profile.get("source_kind")
    if (
        marker.get("schema_version") != "private_scan_store.v1"
        or kind not in ("real", "synthetic")
        or (marker.get("synthetic_experiment") is True) != (kind == "synthetic")
        or profile.get("engine_id") != cfg.get("engine_id")
        or not isinstance(profile.get("method"), str)
    ):
        raise InputError("Store, profile source or runtime engine is inconsistent")
    current = list(current)
    if not current or len(current) != len(set(current)) or reference in current:
        raise InputError(
            "Distinct explicit reference and current capture IDs are required"
        )
    captures = [
        _capture(root, i, profile, kind, cfg["engine_id"])
        for i in [reference, *current]
    ]
    first = captures[0]
    if first["video_sha256"] != profile.get("source_capture_sha256"):
        raise InputError("Reference must be the capture that established this profile")
    for row in captures[1:]:
        if (
            row["native_implementation_sha256"] != first["native_implementation_sha256"]
            or row["native_research_options"] != first["native_research_options"]
        ):
            raise InputError(
                "Native processing implementation or options changed between captures"
            )
    return {
        "schema_version": SCHEMA,
        "source_kind": kind,
        "diagnostic_only": True,
        "physical_accuracy_validated": False,
        "store": str(root),
        "store_sha256": sha256(root / "store.json"),
        "profile_id": profile["id"],
        "profile_sha256": sha256(root / "profile.json"),
        "engine_id": cfg["engine_id"],
        "native_method": profile["method"],
        "runtime_config_sha256": sha256(Path(runtime_config)),
        "reference_capture_id": reference,
        "current_capture_ids": current,
        "captures": captures,
        "measurement_ready": False,
    }


def verify(manifest, runtime_config):
    """Re-read the actual video and estimate files, not only manifest hashes."""
    if manifest.get("schema_version") != SCHEMA:
        raise InputError("Unknown capture measurement input schema")
    actual = build(
        manifest["store"],
        manifest["reference_capture_id"],
        manifest["current_capture_ids"],
        runtime_config,
    )
    if actual != manifest:
        raise InputError("Frozen native capture inputs changed")
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", nargs="+", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(
        Path(args.out),
        build(args.store, args.reference, args.current, args.runtime_config),
    )
