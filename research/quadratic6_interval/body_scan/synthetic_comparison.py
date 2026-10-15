"""Compare frozen synthetic ablations with input hashes and full denominators."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from .io import InputError, read_json, replace_json, sha256
from .synthetic_evidence import evidence


def sampling_metadata(root: Path, report: dict) -> dict:
    snapshot = root / "implementation" / "runtime-config.json"
    if snapshot.exists():
        return {
            "sample_frames": read_json(snapshot)["sample_frames"],
            "sampling_metadata_source": "runtime_config_snapshot",
        }
    # Legacy experiments lack this snapshot; preserve that limitation and use
    # their actual saved frame outcomes rather than inventing a configuration.
    counts = sorted(
        {
            len(read_json(path)["frames"])
            for path in (root / "store").glob("*/attempt_*/frames.json")
        }
    )
    return {
        "sample_frames": counts[0] if len(counts) == 1 else counts,
        "sampling_metadata_source": "observed_frame_outcomes; legacy_config_snapshot_absent",
    }


def compare(roots: dict[str, Path], output: Path) -> dict:
    original = read_json(next(iter(roots.values())) / "report.json")
    source_rows = {row["name"]: row for row in original["results"]}
    reports, summaries = {}, {}
    reference = next(
        (r["name"] for r in original["results"] if r["spec"]["group"] == "baseline"),
        None,
    )
    for label, root in roots.items():
        report = read_json(root / "report.json")
        if report["pending"]:
            raise InputError("Comparison requires terminal experiment results")
        for row in report["results"]:
            if (
                row["name"] not in source_rows
                or row["video_sha256"] != source_rows[row["name"]]["video_sha256"]
            ):
                raise InputError("Comparison input bytes do not match")
            if (
                sha256(root / "store" / row["session_id"] / "capture.bin")
                != row["video_sha256"]
            ):
                raise InputError("Comparison capture changed after reconstruction")
        reports[label] = report
        # Keep the historical analysis unchanged; new variants get the extended
        # analysis including hard-QC diagnostics and planned denominators.
        saved = root / "evidence.json"
        summary = read_json(saved) if saved.exists() else evidence(root)
        summaries[label] = summary
    result = {
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "input_hashes_match": True,
        "matched_stress_reference_case": reference,
        "variants": {
            label: {
                "options": read_json(root / "matrix.json").get("research_options", {}),
                "engine_id": reports[label]["engine_id"],
                **sampling_metadata(root, reports[label]),
                "summary": summaries[label],
            }
            for label, root in roots.items()
        },
        "rows": [
            {
                "name": name,
                "states": {
                    label: next(
                        (r["state"] for r in report["results"] if r["name"] == name),
                        "not_planned",
                    )
                    for label, report in reports.items()
                },
            }
            for name in source_rows
        ],
        "limitations": [
            "Same input replays, not independent captures",
            "Single procedural body per comparison",
            "Known intrinsics are privileged camera metadata; geometry truth is not used",
            "Thresholds are exploratory and not human-validated",
            "Changed method/profile section means absolute circumference is not a calibration test",
        ],
    }
    replace_json(output / "comparison.json", result)
    lines = [
        "# 修正候補の比較",
        "",
        "同一の合成動画を再処理。人体への採用判断は未検証。失敗・拒否を分母に含める。",
        "",
        "|実装|計画/完了/拒否・失敗|基準最大差 model mm|方向一致/計画pair|方向と3倍余裕/計画pair|距離最大差 %|",
        "|---|---|---:|---|---|---:|",
    ]
    for label, report in reports.items():
        s = summaries[label]
        planned = sum(
            r["spec"]["group"] == "baseline" for r in report["results"]
        ) * sum(r["spec"]["group"] == "change" for r in report["results"])
        null, distance = (
            s["baseline_max_null_model_mm"],
            s["distance_max_symmetric_percent"],
        )
        lines.append(
            f"|{label}|{s['case_count']}/{s['completed']}/{s['failed']}|{null if null is not None else '未測定'}|{s['correct_direction_pairs']}/{planned}|{s['direction_and_margin_pairs']}/{planned}|{distance if distance is not None else '未測定'}|"
        )
    lines += [
        "",
        "## 既知変形への反応",
        "",
        "baseline各本との差の平均と範囲。独立標本の信頼区間ではない。",
        "",
        "|実装|入力変化 model mm|推定変化 平均 [最小,最大] model mm|",
        "|---|---:|---|",
    ]
    for label, s in summaries.items():
        for name in sorted({p["after"] for p in s["change_pairs"]}):
            pairs = [p for p in s["change_pairs"] if p["after"] == name]
            values = [p["prediction_model_mm"] for p in pairs]
            lines.append(
                f"|{label}|{pairs[0]['truth_model_mm']:.2f}|{mean(values):.2f} [{min(values):.2f}, {max(values):.2f}]|"
            )
    lines += [
        "",
        "## 撮影条件",
        "",
        f"受理ケースの値は各実装の{reference}との対称差。同じ基準入力を使用する。拒否は寸法を返さない。衣服・呼吸等は外形自体も変える。",
        "",
        "|条件|" + "|".join(reports) + "|",
        "|---|" + "|".join("---" for _ in reports) + "|",
    ]
    for name in source_rows:
        if source_rows[name]["spec"]["group"] != "stress":
            continue
        values = []
        for report in reports.values():
            row = next((r for r in report["results"] if r["name"] == name), None)
            if row is None:
                values.append("対象外")
            elif row["state"] != "complete":
                values.append(row["state"])
            else:
                baseline = next(
                    (
                        r
                        for r in report["results"]
                        if r["name"] == reference and r["state"] == "complete"
                    ),
                    None,
                )
                estimate = row.get("estimated_perimeter_model_m")
                centre = (
                    baseline.get("estimated_perimeter_model_m") if baseline else None
                )
                value = (
                    200 * abs(estimate - centre) / (estimate + centre)
                    if estimate is not None and centre is not None
                    else None
                )
                values.append(f"{value:.3f}%" if value is not None else "基準なし")
        lines.append("|" + name + "|" + "|".join(values) + "|")
    lines += ["", "## 限界", ""] + ["- " + item for item in result["limitations"]]
    path = output / "comparison.md"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return result


def plot(roots: dict[str, Path], output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, root in roots.items():
        s = read_json(root / "evidence.json")
        if not s["change_pairs"]:
            continue
        groups = {}
        for pair in s["change_pairs"]:
            groups.setdefault(round(pair["truth_model_mm"]), []).append(
                pair["prediction_model_mm"]
            )
        x = np.array(sorted(groups))
        y = np.array([mean(groups[t]) for t in x])
        error = np.array(
            [
                [y[i] - min(groups[t]) for i, t in enumerate(x)],
                [max(groups[t]) - y[i] for i, t in enumerate(x)],
            ]
        )
        ax.errorbar(x, y, yerr=error, fmt="o-", capsize=3, label=label)
    ax.plot([-20, 40], [-20, 40], "--", color="gray", label="Renderer truth")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set(
        xlabel="Input circumference change (model mm)",
        ylabel="Estimated change (model mm)",
        title="Same-video ablations: mean and baseline range",
    )
    ax.legend()
    fig.savefig(output / "comparison.png", dpi=160)
    (output / "comparison.png").chmod(0o600)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="label=private_experiment_path",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    roots = dict(
        (label, Path(path).resolve())
        for label, path in (arg.split("=", 1) for arg in args.variant)
    )
    output = Path(args.out).resolve()
    compare(roots, output)
    plot(roots, output)
