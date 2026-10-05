"""Resolve explicit capture inputs or the existing synthetic lab source.

Only the legacy lab branch reads a report. Direct capture observations preserve
their real/synthetic provenance and obtain their reference role from capture ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .capture_manifest import verify
from .io import InputError, read_json, sha256


@dataclass
class CaptureSource:
    store: Path
    profile: dict
    rows: dict
    metadata: dict
    direct: bool

    def is_reference(self, row):
        if self.direct:
            return row["session_id"] == self.metadata["reference_capture_id"]
        return row["spec"]["group"] == "baseline"


def from_capture_inputs(path, runtime_config):
    path = Path(path).resolve()
    runtime_config = Path(runtime_config).resolve()
    inputs = verify(read_json(path), runtime_config)
    store = Path(inputs["store"])
    rows = {
        capture["session_id"]: {
            "name": capture["session_id"],
            "session_id": capture["session_id"],
            "state": "complete",
            "video_sha256": capture["video_sha256"],
            "frame_estimate_sha256": {
                str(f["frame"]): f["estimate_sha256"] for f in capture["frames"]
            },
        }
        for capture in inputs["captures"]
    }
    metadata = {
        "source_kind": inputs["source_kind"],
        "source": str(store),
        "profile_sha256": inputs["profile_sha256"],
        "capture_inputs_manifest": str(path),
        "capture_inputs_sha256": sha256(path),
        "capture_runtime_config": str(runtime_config),
        "reference_capture_id": inputs["reference_capture_id"],
    }
    return CaptureSource(store, read_json(store / "profile.json"), rows, metadata, True)


def supported_observations(manifest):
    if manifest.get("diagnostic_only") is not True:
        return False
    kind = manifest.get("source_kind")
    return kind == "synthetic" or (
        kind == "real" and "capture_inputs_manifest" in manifest
    )


def resolve_source(manifest, runtime_config=None):
    """Re-read direct pipeline artifacts before exposing their capture mappings."""
    if not supported_observations(manifest):
        raise InputError("Validated diagnostic capture observations are required")
    if "capture_inputs_manifest" in manifest:
        path = Path(manifest["capture_inputs_manifest"])
        if sha256(path) != manifest["capture_inputs_sha256"]:
            raise InputError("Frozen capture input manifest changed")
        context = from_capture_inputs(
            path, runtime_config or manifest["capture_runtime_config"]
        )
        if any(manifest.get(k) != v for k, v in context.metadata.items()):
            raise InputError(
                "Observation cache no longer matches its native capture source"
            )
        if any(row["name"] not in context.rows for row in manifest["cases"]):
            raise InputError("Observation cache contains an unknown capture ID")
        if any(
            row.get("video_sha256") != context.rows[row["name"]]["video_sha256"]
            or row.get("frame_estimate_sha256")
            != context.rows[row["name"]]["frame_estimate_sha256"]
            for row in manifest["cases"]
        ):
            raise InputError("Observation cache video or frame identities changed")
        return context
    source = Path(manifest["source"])
    store = source / "store"
    if (
        sha256(source / "report.json") != manifest["source_report_sha256"]
        or sha256(store / "profile.json") != manifest["profile_sha256"]
    ):
        raise InputError("Frozen source reconstruction changed")
    profile = read_json(store / "profile.json")
    report = read_json(source / "report.json")
    if (
        report.get("source_kind") != "synthetic"
        or profile.get("source_kind") != "synthetic"
    ):
        raise InputError("Legacy lab observations must remain synthetic")
    return CaptureSource(
        store, profile, {r["name"]: r for r in report["results"]}, {}, False
    )
