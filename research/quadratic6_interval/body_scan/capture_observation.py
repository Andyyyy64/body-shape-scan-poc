"""Experimental native-capture -> image/pose/contour calibration observations.

Each model/image stage runs in its own isolated process. The reference/current
identities come from native pipeline sessions, without a renderer report.
"""

from __future__ import annotations

import argparse
import os
import shutil
import threading
import time
from pathlib import Path

from .capture_manifest import build
from .io import InputError, new_run, read_json, replace_json, sha256, write_json
from .research import options
from .synthetic_lab import launch, snapshot_sources

STEPS = (
    "image",
    "pose-inputs",
    "pose",
    "contour",
    "query-fit",
    "reference-input",
    "reference-fit",
    "reference-contour",
)


def stage(root, name):
    root = Path(root).resolve()
    plan = read_json(root / "plan.json")
    cfg = plan["runtime_config"]
    if sha256(Path(cfg)) != plan["runtime_config_sha256"]:
        raise InputError("Observation runtime configuration changed")
    import psutil

    limits = read_json(cfg)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > limits["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < limits["min_available_mb"] * 1024**2
            ):
                write_json(
                    root / (name + "-resource-failure.json"),
                    {"reason": "resource_guard", "not_qc_refusal": True},
                )
                os._exit(75)
            time.sleep(0.25)

    threading.Thread(target=guard, daemon=True).start()
    query = read_json(root / "query-options.json")
    reference = read_json(root / "reference-options.json")
    if (
        sha256(root / "query-options.json") != plan["query_options_sha256"]
        or sha256(root / "reference-options.json") != plan["reference_options_sha256"]
    ):
        raise InputError("Frozen observation options changed")
    inputs = read_json(root / "capture-inputs.json")
    if sha256(root / "capture-inputs.json") != plan["capture_inputs_sha256"]:
        raise InputError("Capture input plan changed")
    names = [inputs["reference_capture_id"], *inputs["current_capture_ids"]]
    if name == "image":
        from .synthetic_image_cache import cache

        cache(
            cfg,
            inputs["store"],
            root / "images",
            names,
            contour_samples=1024,
            capture_inputs=root / "capture-inputs.json",
        )
    elif name == "pose-inputs":
        import numpy as np

        parent = root / "images"
        manifest = read_json(parent / "cache-manifest.json")
        destination = new_run(root / "pose-inputs")
        for row in manifest["cases"]:
            file = parent / row["cache"]
            if sha256(file) != row["cache_sha256"]:
                raise InputError("Frozen image observations changed")
            with np.load(file, allow_pickle=False) as n:
                data = {k: n[k] for k in n.files}
            data["shared_target"] = data["legacy_target256"]
            np.savez_compressed(destination / row["cache"], **data)
            (destination / row["cache"]).chmod(0o600)
            row["source_image_cache_sha256"] = row["cache_sha256"]
            row["cache_sha256"] = sha256(destination / row["cache"])
        manifest["paired_dense_image_manifest_sha256"] = sha256(
            parent / "cache-manifest.json"
        )
        write_json(destination / "cache-manifest.json", manifest)
    elif name == "pose":
        from .synthetic_pose_cache import cache

        cache(cfg, root / "pose-inputs", root / "poses", query)
    elif name == "contour":
        from .synthetic_contour_refit import cache

        cache(root / "poses", root / "contours", query)
    elif name == "query-fit":
        from .synthetic_refit import run

        run(root / "contours", root / "query-fits", query)
    elif name == "reference-input":
        parent = root / "contours"
        manifest = read_json(parent / "cache-manifest.json")
        manifest["cases"] = [
            r for r in manifest["cases"] if r["name"] == inputs["reference_capture_id"]
        ]
        if len(manifest["cases"]) != 1:
            raise InputError("Exactly one explicit reference observation is required")
        destination = new_run(root / "reference-inputs")
        for row in manifest["cases"]:
            if sha256(parent / row["cache"]) != row["cache_sha256"]:
                raise InputError("Frozen reference observations changed")
            shutil.copyfile(parent / row["cache"], destination / row["cache"])
            (destination / row["cache"]).chmod(0o600)
        manifest["parent_subset_manifest_sha256"] = sha256(
            parent / "cache-manifest.json"
        )
        write_json(destination / "cache-manifest.json", manifest)
    elif name == "reference-fit":
        from .synthetic_refit import run

        run(root / "reference-inputs", root / "reference-fits", reference)
    elif name == "reference-contour":
        from .synthetic_contour_refit import cache

        cache(
            root / "reference-inputs",
            root / "reference-contours",
            reference,
            reference_fit=root / "reference-fits",
        )
    else:
        raise InputError("Unknown capture observation stage")


def run(
    config_path,
    store,
    reference_id,
    current_ids,
    query_options,
    reference_options,
    output,
):
    query, reference = (
        options(read_json(query_options)),
        options(read_json(reference_options)),
    )
    if (
        query.get("fit", {}).get("global_shape") is not False
        or reference.get("fit", {}).get("global_shape") is not True
    ):
        raise InputError(
            "Calibration requires explicit reference-global and query-local fit roles"
        )
    captures = build(store, reference_id, current_ids, config_path)
    root = new_run(output)
    snapshot_sources(root, config_path)
    write_json(root / "capture-inputs.json", captures)
    write_json(root / "query-options.json", query)
    write_json(root / "reference-options.json", reference)
    write_json(
        root / "plan.json",
        {
            "source_kind": captures["source_kind"],
            "diagnostic_only": True,
            "physical_accuracy_validated": False,
            "runtime_config": str(Path(config_path).resolve()),
            "runtime_config_sha256": sha256(Path(config_path)),
            "capture_inputs_sha256": sha256(root / "capture-inputs.json"),
            "query_options_sha256": sha256(root / "query-options.json"),
            "reference_options_sha256": sha256(root / "reference-options.json"),
            "observer_sha256": sha256(Path(__file__)),
            "steps": list(STEPS),
            "renderer_report_access": False,
        },
    )
    state = {"stage": "starting", "completed": [], "failures": []}
    replace_json(root / "status.json", state)
    for name in STEPS:
        if sha256(Path(__file__)) != read_json(root / "plan.json")["observer_sha256"]:
            raise InputError("Capture observer implementation changed")
        state["stage"] = name
        replace_json(root / "status.json", state)
        print("stage_start", name, flush=True)
        rc = launch(
            config_path,
            [
                "-m",
                "body_scan.capture_observation",
                "stage",
                "--root",
                str(root),
                "--name",
                name,
            ],
            root / (name + ".log"),
        )
        if rc:
            state["stage"] = "failed"
            state["failures"].append(
                {"stage": name, "exit_code": rc, "not_qc_refusal": True}
            )
            replace_json(root / "status.json", state)
            return rc
        state["completed"].append(name)
        replace_json(root / "status.json", state)
        print("stage_complete", name, flush=True)
    state["stage"] = "complete_observations"
    replace_json(root / "status.json", state)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--current", nargs="+", required=True)
    p.add_argument("--query-options", required=True)
    p.add_argument("--reference-options", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--root", required=True)
    p.add_argument("--name", choices=STEPS, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        stage(args.root, args.name)
    else:
        raise SystemExit(
            run(
                args.runtime_config,
                args.store,
                args.reference,
                args.current,
                args.query_options,
                args.reference_options,
                args.out,
            )
        )
