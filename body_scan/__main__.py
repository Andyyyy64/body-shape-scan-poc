"""Run `body-scan --help` for local-only commands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import __version__
from .geometry import measure_widths
from .io import InputError, new_run, read_json, sha256, write_json
from .statistics import analyze_pairs, detection_power


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local body-shape research workbench; no physical accuracy established"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "doctor",
        help="Inspect the base environment and an optional external Mac runtime",
    )
    p.add_argument("--runtime-config")
    for name in ("demo", "render-demo"):
        p = sub.add_parser(name, help="Run explicitly synthetic experiments")
        p.add_argument("--out", required=True)
        if name == "render-demo":
            p.add_argument("--delta-m", type=float, default=0.0)
    p = sub.add_parser(
        "power",
        help="Two-sided detection power with explicit independent-error assumptions",
    )
    p.add_argument("--sigma-m", type=float, required=True)
    p.add_argument("--delta-m", type=float, required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p = sub.add_parser(
        "measure-widths",
        help="Integrate metric orthographic widths, never raw pixel widths",
    )
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--max-gap-deg", type=float, default=30.0)
    p = sub.add_parser("analyze", help="Analyze declared independent session pairs")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold-m", type=float, required=True)
    p.add_argument("--target-m", type=float, required=True)
    p = sub.add_parser(
        "capture", help="Explicitly record a local camera; Q stops the optional preview"
    )
    p.add_argument("--source", required=True, help="Camera index or local video path")
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--delay-s", type=float, default=5.0)
    p.add_argument("--source-kind", choices=["real", "synthetic"], default="real")
    p = sub.add_parser("frames", help="Extract temporal samples; angles remain unknown")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=120)
    p = sub.add_parser(
        "calibrate",
        help="Camera intrinsics from 8+ chessboard images; does not establish body depth",
    )
    p.add_argument("images")
    p.add_argument("--out", required=True)
    p.add_argument("--columns", type=int, required=True, help="Inner corner columns")
    p.add_argument("--rows", type=int, required=True, help="Inner corner rows")
    p.add_argument("--square-m", type=float, required=True)
    p = sub.add_parser(
        "undistort",
        help="Use one full-image coordinate system before segmentation or SAM",
    )
    p.add_argument("images")
    p.add_argument("--calibration", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "board",
        help="Write a printable checkerboard SVG; verify printed square dimensions",
    )
    p.add_argument("--columns", type=int, required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--square-m", type=float, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "segment",
        help="Local single-person YOLO masks; explicit local checkpoint required",
    )
    p.add_argument("images")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "reconstruct",
        help="Cross-sections from masks + supplied calibrated metric body poses",
    )
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "template",
        help="Create incomplete private input contracts; no invented calibration",
    )
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "setup-mac",
        help="Install the pinned official SAM Mac runtime after model access approval",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=16)
    p = sub.add_parser(
        "serve",
        help="Open a private loopback capture and canonical 3D comparison screen",
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--runtime-config")
    p = sub.add_parser(
        "compare-meshes",
        help="Compare matching canonical meshes without date-based suppression",
    )
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--tolerance-mm", type=float)
    p.add_argument("--out", required=True)
    p = sub.add_parser(
        "audit", help="Audit public tracked files and optionally every commit"
    )
    p.add_argument("--staged", action="store_true")
    p.add_argument("--history", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "setup-mac":
            from .runtime_setup import setup

            result = setup(args.out, args.frames)
        elif args.command == "serve":
            from .web import serve

            if not 0 <= args.port <= 65535:
                raise InputError("Invalid local port")
            serve(args.data_root, args.port, args.runtime_config)
            return 0
        elif args.command == "compare-meshes":
            from .mesh import compare_meshes

            result = compare_meshes(
                read_json(args.before), read_json(args.after), args.tolerance_mm
            )
            root = new_run(args.out)
            write_json(root / "comparison.json", result)
            from .demo import html_report

            html_report(root, result)
            result = {
                "status": "comparison_written",
                "physical_accuracy_validated": False,
            }
        elif args.command == "doctor":
            modules = {
                name: importlib.util.find_spec(name) is not None
                for name in ["numpy", "cv2", "torch", "ultralytics", "sam_3d_body"]
            }
            cuda = False
            if modules["torch"]:
                import torch

                cuda = torch.cuda.is_available()
            result = {
                "python": sys.version.split()[0],
                "modules": modules,
                "cuda_available": cuda,
                "external_runtime": "not_checked",
                "physical_accuracy_validated": False,
                "network_upload": False,
            }
            if args.runtime_config:
                import subprocess

                config = read_json(args.runtime_config)
                present = all(
                    Path(config[k]).exists()
                    for k in [
                        "python",
                        "sam_source",
                        "dino_source",
                        "checkpoint",
                        "mhr_model",
                        "segmentation_model",
                    ]
                )
                probe = subprocess.run(
                    [
                        config["python"],
                        "-c",
                        "import json,torch; print(json.dumps({'mps_available':torch.backends.mps.is_available()}))",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                result["external_runtime"] = {
                    "model_files_present": present,
                    "encoder_device": config["encoder_device"],
                    **json.loads(probe.stdout),
                    "network_sandbox": config.get("network_sandbox"),
                }
        elif args.command in ("demo", "render-demo"):
            from .demo import render_demo, run_demo

            report = (
                run_demo(args.out)
                if args.command == "demo"
                else render_demo(args.out, args.delta_m)
            )
            result = {
                "status": "synthetic_experiment_complete",
                "physical_accuracy_validated": False,
            }
            if args.command == "render-demo":
                result["synthetic_error_m"] = report["error_m"]
        elif args.command == "power":
            result = detection_power(args.sigma_m, args.delta_m, args.alpha)
        elif args.command in ("measure-widths", "analyze"):
            from .demo import html_report

            data = read_json(args.input)
            result = (
                measure_widths(data, args.max_gap_deg)
                if args.command == "measure-widths"
                else analyze_pairs(data, args.threshold_m, args.target_m)
            )
            result.update(
                engine_version=__version__, input_sha256=sha256(Path(args.input))
            )
            root = new_run(args.out)
            write_json(root / "report.json", result)
            html_report(root, result)
            result = {"status": "report_written", "physical_accuracy_validated": False}
        elif args.command == "board":
            from .geometry import number

            if not 3 <= args.columns <= 30 or not 3 <= args.rows <= 30:
                raise InputError("Inner corner counts must be within 3..30")
            square_mm = 1000 * number(args.square_m, "square_m", positive=True)
            width = (args.columns + 1) * square_mm
            height = (args.rows + 1) * square_mm
            root = new_run(args.out)
            rectangles = "".join(
                f'<rect x="{x * square_mm}" y="{y * square_mm}" width="{square_mm}" height="{square_mm}"/>'
                for y in range(args.rows + 1)
                for x in range(args.columns + 1)
                if (x + y) % 2 == 0
            )
            board = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 20}mm" height="{height + 20}mm" viewBox="-10 -10 {width + 20} {height + 20}"><rect x="-10" y="-10" width="{width + 20}" height="{height + 20}" fill="white"/><g fill="black">{rectangles}</g></svg>'
            (root / "board.svg").write_text(board, encoding="utf-8")
            (root / "board.svg").chmod(0o600)
            result = {
                "status": "board_created",
                "square_mm": square_mm,
                "note": "Print at 100 percent and physically verify square size; no fit-to-page.",
            }
        elif args.command == "template":
            root = new_run(args.out)
            write_json(
                root / "scan.json",
                {
                    "schema_version": "silhouette_scan.v1",
                    "source_kind": "real",
                    "measurement_definition_id": None,
                    "pose_evidence_id": None,
                    "scale_evidence_id": None,
                    "camera": {"K": None, "distortion": None, "image_size": None},
                    "bounds_xz_m": None,
                    "grid_step_m": None,
                    "max_yaw_gap_deg": None,
                    "sections": [],
                    "views": [],
                },
            )
            write_json(
                root / "pairs.json",
                {
                    "schema_version": "paired_validation.v1",
                    "source_kind": "real",
                    "pairs": [],
                },
            )
            result = {"status": "templates_created_measurement_not_ready"}
        elif args.command == "audit":
            from .audit import audit

            result = audit(args.staged, args.history)
        else:
            from . import vision

            if args.command == "capture":
                result = vision.record(
                    args.source,
                    args.out,
                    args.seconds,
                    args.preview,
                    args.delay_s,
                    args.source_kind,
                )
            elif args.command == "frames":
                result = vision.extract(
                    args.video, args.out, args.stride, args.max_frames
                )
            elif args.command == "calibrate":
                result = vision.calibrate(
                    args.images, args.out, args.columns, args.rows, args.square_m
                )
            elif args.command == "undistort":
                result = vision.undistort(args.images, args.calibration, args.out)
            elif args.command == "segment":
                result = vision.segment(args.images, args.model, args.out)
            else:
                report = vision.reconstruct(args.manifest, args.out)
                result = {
                    "status": report["status"],
                    "sections": len(report["sections"]),
                    "physical_accuracy_validated": False,
                }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 1 if result.get("status") == "fail" else 0
    except Exception as exc:  # noqa: BLE001 - third-party errors may contain participant data
        # Runtime messages may include private paths or data. Never print tracebacks in normal operation.
        message = (
            str(exc)
            if isinstance(exc, InputError)
            else "Invalid input, missing dependency, or unavailable local file; see the runbook"
        )
        print(json.dumps({"status": "error", "message": message}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
