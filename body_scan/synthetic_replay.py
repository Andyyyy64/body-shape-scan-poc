"""Replay identical synthetic input bytes in a fresh store for one ablation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .io import InputError, new_run, read_json, sha256, write_json
from .research import options
from .synthetic_lab import snapshot_sources
from .web import session_path


def replay(
    source: str,
    output: str,
    config_path: str,
    research: dict | None,
    names: list[str] | None = None,
) -> Path:
    original = Path(source).resolve()
    manifest = read_json(original / "matrix.json")
    selected = [c for c in manifest["cases"] if not names or c["name"] in names]
    if names and set(names) != {c["name"] for c in selected}:
        raise InputError("Unknown replay case")
    enabled = research is not None
    research = options(research)
    root = new_run(output)
    snapshot_sources(root, config_path)
    for name in ("body.npz", "body.json", "protocol.json"):
        shutil.copyfile(original / name, root / name)
        (root / name).chmod(0o600)
    if enabled:
        write_json(root / "research-options.json", research)
    new_run(root / "store")
    write_json(
        root / "store" / "store.json",
        {
            "schema_version": "private_scan_store.v1",
            "synthetic_experiment": True,
        },
    )
    cases = []
    for previous in selected:
        if not previous.get("session_id"):
            raise InputError("Replay requires an existing synthetic capture")
        identifier = previous["session_id"]
        src = session_path(original / "store", identifier)
        meta = read_json(src / "session.json")
        if (
            meta.get("source_kind") != "synthetic"
            or sha256(src / "capture.bin") != meta["video_sha256"]
        ):
            raise InputError("Replay source provenance failed")
        dest = new_run(root / "store" / identifier)
        for file in [
            src / "capture.bin",
            src / "truth.json",
            *src.glob("preview_*.jpg"),
        ]:
            shutil.copyfile(file, dest / file.name)
            (dest / file.name).chmod(0o600)
        write_json(
            dest / "session.json",
            {
                **meta,
                "replayed_from_capture": identifier,
                "state": "recorded_not_reconstructed",
            },
        )
        cases.append(
            {
                "name": previous["name"],
                "spec": previous["spec"],
                "state": "rendered",
                "session_id": identifier,
            }
        )
    config = read_json(config_path)
    write_json(
        root / "matrix.json",
        {
            "source_kind": "synthetic",
            "engine_id": config["engine_id"],
            **({"research_options": research} if enabled else {}),
            "cases": cases,
            "source_experiment": str(original),
            "interpretation": "Same video bytes; ablation replay, not independent recapture",
        },
    )
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--options", help="Omit to replay the default method")
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    replay(
        args.source,
        args.out,
        args.runtime_config,
        read_json(args.options) if args.options else None,
        args.cases,
    )
    print("Prepared synthetic replay in a new store")
