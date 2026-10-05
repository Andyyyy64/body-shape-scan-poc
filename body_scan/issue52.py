"""Frozen, one-decision real-video review for Soma #52; no candidate search."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .io import (
    InputError,
    new_run,
    private_path,
    read_json,
    replace_json,
    sha256,
    write_json,
)
from .mesh import compare_meshes, section_perimeter, validate_mesh
from .network_sandbox import sandbox_command, verify_network_sandbox
from .web import record_resolution_fallback, session_path, store_root

ISSUE = "https://github.com/Andyyyy64/soma/issues/52"
ACCEPTANCE = """- 感度: 変化ペアの差が既知変化と同じ向きで、無変化ペアの最大差より明確に大きい（無変化最大差の3倍以上を目安）。固定形状対照は不合格。
- 距離: 距離間の差が1%以内、または1%を超える距離を拒否/補正する規則が根拠付きで決まる。
- 失敗・除外・撮り直しを隠さず記録する。
- 判定: 両方合格 → #RM-1401 と #RM-1402 を進める。感度不合格 → 採用見送りを記録して停止。"""
SECTION = "model_abdomen_torso32.v1"
SLOTS = (
    *(f"normal_{i}" for i in range(1, 4)),
    *(f"changed_{i}" for i in range(1, 4)),
    *(f"distance_{d}_{i}" for d in ("1.5", "2.0", "2.5") for i in (1, 2)),
)
# Byte-equivalent options from boundary-selected-options.json, except the camera:
# renderer-only 640x640 intrinsics have no meaning for a real recording.
CANDIDATE = {
    "region": "abdomen",
    "region_weight": 8.0,
    "shape_weight": 0.01,
    "qc": {
        "mask_support": True,
        "require_rotation": True,
        "max_relative_fit_error": 0.02,
        "min_boundary_detail": 2.287923829396852,
    },
    "fit": {
        "normalize_observation": True,
        "anchor_weight": 1,
        "shared_skeleton": True,
        "max_frames": 48,
    },
}


def source_hashes(source: Path) -> dict:
    return {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*.py"))}


def prepare(out: str, runtime_config: str, budget_seconds: float = 14400) -> Path:
    if not 0 < budget_seconds <= 14400:
        raise InputError("Remaining budget must be positive and at most four hours")
    config = read_json(runtime_config)
    if config["sample_frames"] != 16 or config["max_rss_mb"] != 8192:
        raise InputError(
            "Preparation requires the unchanged 16-frame v1 / 8192 MiB runtime"
        )
    root = new_run(out)
    captures = store_root(str(root / "captures"))
    replace_json(
        captures / "store.json",
        {
            "schema_version": "private_scan_store.v1",
            "issue52_capture_only": True,
        },
    )
    source = root / "source"
    shutil.copytree(
        Path(__file__).parent,
        source / "body_scan",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for path in source.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    write_json(root / "v1.json", config)
    write_json(root / "oct03-48.json", {**config, "sample_frames": 48})
    write_json(root / "candidate.json", CANDIDATE)
    write_json(
        root / "plan.json",
        {
            "schema_version": "issue52_real_review.v1",
            "issue": ISSUE,
            "acceptance_verbatim": ACCEPTANCE,
            "slots": list(SLOTS),
            "methods": ["v1", "oct03-48"],
            "section": SECTION,
            "source_sha256": source_hashes(source),
            "files_sha256": {
                f: sha256(root / f)
                for f in ("v1.json", "oct03-48.json", "candidate.json")
            },
            "policy": {
                "null_pairs": "all 6 within-condition pairs in normal and changed groups",
                "change_pairs": "all 9 normal-to-changed pairs, all directed changes > 0 and >= 3 * null maximum",
                "distance_pairs": "all 12 cross-distance pairs; symmetric percent = 200*abs(a-b)/(a+b)",
                "other_pairs": "all remaining pairs recorded as references",
                "distance_rule": "if >1%, propose measured-distance rejection; do not apply and rescore",
                "fixed_shape_control": "copy one real mesh for all 12 slots and score identically",
                "missing_or_failed": "incomplete; defer, never silently exclude",
                "post_result_changes": False,
                "rescaling": False,
                "old_synthetic_gates": "reference only: +/-10, gain 0.8..1.2, 375 pairs",
                "source": "Oct 3 boundary-selected options, renderer camera removed only",
                "max_processing_seconds": budget_seconds,
            },
        },
    )
    return root


def inventory(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "captures").iterdir()):
        if not (path / "session.json").is_file():
            continue
        meta = read_json(path / "session.json")
        rows.append(
            {
                "id": path.name,
                "metadata": meta,
                "video_sha256": sha256(path / "capture.bin"),
            }
        )
    return rows


def bind(root: Path, change_mm: float, notes: str, selections: list[str]) -> None:
    if not math.isfinite(change_mm) or change_mm == 0 or not notes.strip():
        raise InputError(
            "Declare a nonzero known change and capture/retake notes before processing"
        )
    explicit = {}
    for selection in selections:
        slot, identifier = selection.split("=", 1)
        if slot not in SLOTS or slot in explicit:
            raise InputError("Invalid or duplicate slot selection")
        explicit[slot] = identifier
    captures = inventory(root)
    selected = {}
    for slot in SLOTS:
        choices = [r for r in captures if r["metadata"].get("issue52_slot") == slot]
        if slot in explicit:
            choices = [r for r in choices if r["id"] == explicit[slot]]
        if len(choices) != 1:
            raise InputError(
                f"{slot}: need exactly one recording; use --select slot=id to document a retake"
            )
        selected[slot] = choices[0]["id"]
    if len(set(selected.values())) != 12:
        raise InputError("Each slot must have an independent original recording")
    chosen = [r for r in captures if r["id"] in selected.values()]
    if len({r["video_sha256"] for r in chosen}) != 12:
        raise InputError(
            "The twelve originals must be independently recorded, not repeated uploads"
        )
    if any(
        r["metadata"].get("source_kind") != "real"
        or r["video_sha256"] != r["metadata"].get("video_sha256")
        for r in chosen
    ):
        raise InputError("Real source and immutable video provenance are required")
    sensitivity_ids = {selected[s] for s in SLOTS[:6]}
    times = [r["metadata"]["created_at"] for r in chosen if r["id"] in sensitivity_ids]
    if max(times) - min(times) > 3600:
        raise InputError("The six sensitivity recordings must be within one hour")
    write_json(
        root / "selection.json",
        {
            "known_change_mm": change_mm,
            "notes": notes,
            "selected": selected,
            "all_captures_and_retakes": captures,
            "unselected_ids": [
                r["id"] for r in captures if r["id"] not in selected.values()
            ],
            "bound_at": time.time(),
        },
    )


def sensitivity(values: dict[str, float], direction: float) -> dict:
    null = [
        abs(values[a] - values[b])
        for group in (SLOTS[:3], SLOTS[3:6])
        for a, b in itertools.combinations(group, 2)
    ]
    changes = [
        direction * (values[b] - values[a]) for a in SLOTS[:3] for b in SLOTS[3:6]
    ]
    maximum, minimum = max(null), min(changes)
    return {
        "null_max_model_m": maximum,
        "change_min_directed_model_m": minimum,
        "null_pair_count": len(null),
        "change_pair_count": len(changes),
        "margin_ratio": minimum / maximum if maximum > 0 else None,
        "pass": minimum > 0 and minimum >= 3 * maximum,
    }


def score(values: dict[str, float], change_mm: float) -> dict:
    if not math.isfinite(change_mm) or change_mm == 0:
        raise InputError("A declared nonzero known change is required")
    if set(values) != set(SLOTS) or any(
        not math.isfinite(v) or v <= 0 for v in values.values()
    ):
        raise InputError(
            "A decision requires all 12 finite positive abdomen perimeters"
        )
    pairs = [
        {
            "a": a,
            "b": b,
            "delta_model_m": values[b] - values[a],
            "symmetric_percent": 200
            * abs(values[b] - values[a])
            / (values[a] + values[b]),
        }
        for a, b in itertools.combinations(SLOTS, 2)
    ]
    distances = [
        p
        for p in pairs
        if p["a"].startswith("distance_")
        and p["b"].startswith("distance_")
        and p["a"].split("_")[1] != p["b"].split("_")[1]
    ]
    maximum = max(p["symmetric_percent"] for p in distances)
    return {
        "perimeters_model_m": values,
        "all_66_pairs": pairs,
        "sensitivity": sensitivity(values, math.copysign(1, change_mm)),
        "distance": {
            "cross_distance_pair_count": len(distances),
            "maximum_percent": maximum,
            "within_1_percent": maximum <= 1,
            "violating_pairs": [p for p in distances if p["symmetric_percent"] > 1],
        },
    }


def score_meshes(meshes: dict, change_mm: float) -> dict:
    for mesh in meshes.values():
        validate_mesh(mesh)
        if mesh["source_kind"] != "real" or mesh["units"] != "model_m":
            raise InputError("Acceptance uses real model-coordinate meshes")
    reference = meshes[SLOTS[0]]
    # Existing contract verifies topology, section, shared reference and method.
    for mesh in meshes.values():
        compare_meshes(reference, mesh)
    values = {}
    for slot, mesh in meshes.items():
        section = next((s for s in mesh["sections"] if s["id"] == SECTION), None)
        perimeter = section_perimeter(mesh, section) if section else None
        if perimeter is None:
            raise InputError(
                "Required abdomen section has no unambiguous closed perimeter"
            )
        values[slot] = perimeter
    result = score(values, change_mm)
    control = score({s: values[SLOTS[0]] for s in SLOTS}, change_mm)
    result["fixed_shape_control"] = control["sensitivity"]
    result["fixed_shape_control"]["source_mesh_slot"] = SLOTS[0]
    return result


def _run_worker(
    root: Path, method: str, store: Path, identifier: str, deadline: float
) -> dict:
    config_path = root / f"{method}.json"
    config = read_json(config_path)
    command = [
        config["python"],
        "-P",
        "-u",
        "-m",
        "body_scan.pipeline",
        "--config",
        str(config_path),
        "--store",
        str(store),
        "--session",
        identifier,
    ]
    if method != "v1":
        command += [
            "--research-options",
            str(root / "candidate.json"),
            "--diagnostic-real",
        ]
    path = store / identifier
    started = time.time()
    with (path / "runtime.log").open("xb") as log:
        (path / "runtime.log").chmod(0o600)
        process = subprocess.Popen(
            sandbox_command(config["network_sandbox"], command),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "source"),
                "OMP_NUM_THREADS": str(config["cpu_threads"]),
                "MKL_NUM_THREADS": str(config["cpu_threads"]),
                "OPENBLAS_NUM_THREADS": str(config["cpu_threads"]),
            },
        )
        write_json(
            path / "worker.json",
            {"pid": process.pid, "command": command, "started_at": started},
        )
        timed_out = False
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except BaseException as exc:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
            if not timed_out:
                raise
    state_path = path / "reconstruction.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else {"state": "failed", "stage": "no_status"}
    )
    fallback = record_resolution_fallback(root / "captures", identifier, state)
    attempts = [
        {
            "directory": a.name,
            **{
                f[:-5]: read_json(a / f)
                for f in ("frames.json", "qc.json")
                if (a / f).exists()
            },
        }
        for a in sorted(path.glob("attempt_*"))
    ]
    result = {
        "actual_exit_code": code,
        "timed_out": timed_out,
        "elapsed_s": time.time() - started,
        "state": state,
        "resolution_fallback_recorded": fallback,
        "attempts": attempts,
    }
    write_json(path / "terminal.json", result)
    return result


def process(root: Path, budget_seconds: float) -> dict:
    if not 0 < budget_seconds <= 14400:
        raise InputError(
            "Processing budget must be positive and no more than four hours"
        )
    plan, selection = read_json(root / "plan.json"), read_json(root / "selection.json")
    budget_seconds = min(budget_seconds, plan["policy"]["max_processing_seconds"])
    if plan["source_sha256"] != source_hashes(root / "source") or any(
        sha256(root / name) != digest for name, digest in plan["files_sha256"].items()
    ):
        raise InputError("Frozen source or method configuration changed")
    if inventory(root) != selection["all_captures_and_retakes"]:
        raise InputError(
            "Capture inventory changed after binding; do not silently revise the selection"
        )
    config = read_json(root / "v1.json")
    verify_network_sandbox(config["network_sandbox"], config["python"])
    # Exclusive creation is the durable one-shot barrier, including crashes.
    write_json(
        root / "execution.json",
        {
            "started_at": time.time(),
            "budget_seconds": budget_seconds,
            "plan_sha256": sha256(root / "plan.json"),
            "selection_sha256": sha256(root / "selection.json"),
        },
    )
    deadline = time.monotonic() + budget_seconds
    outcome = {
        "issue": ISSUE,
        "acceptance_verbatim": ACCEPTANCE,
        "diagnostic_label": "診断用",
        "known_change_mm": selection["known_change_mm"],
        "all_captures_and_retakes": selection["all_captures_and_retakes"],
        "unselected_ids": selection["unselected_ids"],
        "notes": selection["notes"],
        "methods": {},
        "stop_reason": None,
    }
    try:
        for method in plan["methods"]:
            store = store_root(str(root / method))
            jobs, meshes = {}, {}
            outcome["methods"][method] = {"jobs": jobs, "decision": "incomplete_defer"}
            for slot in SLOTS:
                if time.monotonic() >= deadline:
                    raise InputError("Four-hour processing budget exhausted")
                identifier = selection["selected"][slot]
                src = session_path(root / "captures", identifier)
                dst = new_run(store / identifier)
                for name in ("session.json", "capture.bin"):
                    shutil.copyfile(src / name, dst / name)
                    (dst / name).chmod(0o600)
                jobs[slot] = _run_worker(root, method, store, identifier, deadline)
                if (
                    jobs[slot]["actual_exit_code"] != 0
                    or jobs[slot]["state"].get("state") != "complete"
                ):
                    raise InputError(
                        f"{method}/{slot} failed; no exclusions, retries or new candidates"
                    )
                meshes[slot] = read_json(dst / "mesh.json")
            result = score_meshes(meshes, selection["known_change_mm"])
            sensitivity_pass = (
                result["sensitivity"]["pass"]
                and not result["fixed_shape_control"]["pass"]
            )
            distance_pass = result["distance"]["within_1_percent"]
            outcome["methods"][method].update(
                result=result,
                decision=(
                    "both_pass"
                    if sensitivity_pass and distance_pass
                    else "sensitivity_fail_defer"
                    if not sensitivity_pass
                    else "distance_rule_required_defer"
                ),
            )
            # Both declared methods are assessed once; no optimisation follows failure.
    except (Exception, KeyboardInterrupt) as exc:
        outcome["stop_reason"] = str(exc) or "Interrupted by operator"
    write_json(root / "decision.json", outcome)
    report(root, outcome)
    return outcome


def report(root: Path, outcome: dict) -> None:
    lines = [
        "# Soma #52 — one-time real-video review",
        "",
        f"Source: {ISSUE}",
        "",
        "Oct 3 candidate: 診断用. Units: model_m, not calibrated human metres.",
        "",
        "| Method | Null max | Minimum directed change | Margin | Distance max | Decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    detail_lines = []
    for method in ("v1", "oct03-48"):
        row = outcome["methods"].get(method, {})
        result = row.get("result")
        if result:
            s, d = result["sensitivity"], result["distance"]
            ratio = (
                f"{s['margin_ratio']:.3f}"
                if s["margin_ratio"] is not None
                else "null max = 0"
            )
            lines.append(
                f"| {method} | {s['null_max_model_m']:.6f} | {s['change_min_directed_model_m']:.6f} | {ratio} | {d['maximum_percent']:.3f}% | {row['decision']} |"
            )
            detail_lines += [
                "",
                f"{method}: fixed-shape control sensitivity pass = {result['fixed_shape_control']['pass']}; "
                f"directed change = {result['fixed_shape_control']['change_min_directed_model_m']:.6f} model_m.",
            ]
            if not d["within_1_percent"]:
                repeats = next(
                    p["symmetric_percent"]
                    for p in result["all_66_pairs"]
                    if p["a"] == "distance_2.0_1" and p["b"] == "distance_2.0_2"
                )
                rule = (
                    "restrict capture to the physically marked 2.0 m position and reject 1.5/2.5 m recordings. "
                    if repeats <= 1
                    else "reject all tested distances; even the 2.0 m repeats exceed 1%. "
                )
                detail_lines += [
                    "",
                    f"{method}: distance rejection proposal (not applied or rescored): "
                    f"{rule}"
                    f"Evidence: {len(d['violating_pairs'])} cross-distance pairs exceed 1%; "
                    f"2.0 m repeat difference = {repeats:.3f}%. "
                    "this requires an operational distance check before adoption. No gain correction is proposed.",
                ]
        else:
            lines.append(
                f"| {method} | — | — | — | — | {row.get('decision', 'not_run')} |"
            )
    lines += detail_lines
    passed = [m for m, r in outcome["methods"].items() if r["decision"] == "both_pass"]
    recommendation = (
        "v1 passes: proceed to the follow-up integration decisions in #52."
        if "v1" in passed
        else "Only the diagnostic candidate passes: retain the v1 default and propose a separate review."
        if passed
        else "Defer adoption. Stop; discuss the documented failures or distance rejection proposal separately."
    )
    failures = sum(
        j["actual_exit_code"] != 0
        for r in outcome["methods"].values()
        for j in r["jobs"].values()
    )
    exclusions = sum(
        frame.get("state") == "excluded"
        for r in outcome["methods"].values()
        for j in r["jobs"].values()
        for a in j["attempts"]
        for frame in a.get("frames", {}).get("frames", [])
    )
    lines += [
        "",
        recommendation,
        "",
        f"Failures: {failures}; excluded frames: {exclusions}; retakes/unselected captures: "
        f"{len(outcome['unselected_ids'])}. "
        "Complete capture, failure and frame ledger: private decision.json and method stores.",
        f"Stop reason: {outcome['stop_reason'] or 'one-time decision completed'}.",
        "",
        "No synthetic acceptance gates, result-driven exclusions, numerical rescaling, retries or new candidates.",
        "",
        "このまま続けるべきか：判定後の改善実行は止め、採用または別提案の判断を待つ。",
    ]
    text = "\n".join(lines) + "\n"
    for name in ("review.md", "issue52-summary-draft.md"):
        path = root / name
        with path.open("x", encoding="utf-8") as handle:
            path.chmod(0o600)
            handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--runtime-config", required=True)
    p.add_argument("--budget-seconds", type=float, default=14400)
    for name in ("inventory", "bind", "process"):
        p = sub.add_parser(name)
        p.add_argument("--review", required=True)
        if name == "bind":
            p.add_argument("--known-change-mm", type=float, required=True)
            p.add_argument("--notes", required=True)
            p.add_argument("--select", action="append", default=[])
        if name == "process":
            p.add_argument("--budget-seconds", type=float, default=14400)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare(args.out, args.runtime_config, args.budget_seconds))
        else:
            root = private_path(args.review)
            if args.command == "inventory":
                for r in inventory(root):
                    print(r["metadata"].get("issue52_slot", "unlabelled"), r["id"])
            elif args.command == "bind":
                bind(root, args.known_change_mm, args.notes, args.select)
                print("Selection frozen. Both methods will consume the same originals.")
            else:
                result = process(root, args.budget_seconds)
                print(root / "review.md")
                return 1 if result["stop_reason"] else 0
        return 0
    except (InputError, FileExistsError, KeyError, ValueError) as exc:
        print(f"Stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
