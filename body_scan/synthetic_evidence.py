"""Summarize synthetic experiments, including every refusal and failure."""

from __future__ import annotations

import argparse
import html
import itertools
import math
from pathlib import Path
from statistics import mean

from .io import private_path, read_json, replace_json


def delta_result(prediction: float, truth: float, null: float) -> dict:
    return {
        "prediction_model_mm": prediction * 1000,
        "truth_model_mm": truth * 1000,
        "correct_direction": prediction * truth > 0,
        "clears_null_margin": abs(prediction) > 3 * null,
        "absolute_error_model_mm": abs(prediction - truth) * 1000,
    }


def refusal_kind(root: Path, row: dict) -> str | None:
    """Classify all refusals by saved diagnostics, including unplanned QC cases."""
    if row.get("state") != "failed" or not row.get("session_id"):
        return None
    status = row.get("reconstruction") or {}
    if status.get("stage") != "failed":
        return None
    error = status.get("error", "")
    session = root / "store" / row["session_id"]
    for path in session.glob("attempt_*/qc.json"):
        qc = read_json(path)
        if (
            error.startswith("使用できるフレームが")
            and qc.get("usable_frames", 1) < qc.get("required_frames", 0)
            and qc.get("exclusion_reasons")
        ):
            return "frame_quality"
        if error.startswith("回転範囲が不足") and qc.get("rotation_confirmed") is False:
            return "rotation"
    for path in session.glob("attempt_*/fit.json"):
        fit = read_json(path)
        limit = (
            fit.get("research_options", {}).get("qc", {}).get("max_relative_fit_error")
        )
        if (
            error.startswith("観測輪郭と推定形状の残差")
            and limit is not None
            and fit.get("relative_fit_error", 0) > limit
        ):
            return "fit_residual"
    return None


def evidence(root: Path) -> dict:
    data = read_json(root / "report.json")
    protocol = read_json(root / "protocol.json")
    rows = data["results"]
    baselines = [
        r
        for r in rows
        if r["spec"]["group"] == "baseline"
        and r["state"] == "complete"
        and r.get("estimated_perimeter_model_m") is not None
    ]
    centre = (
        mean(r["estimated_perimeter_model_m"] for r in baselines) if baselines else None
    )
    nulls = [
        abs(a["estimated_perimeter_model_m"] - b["estimated_perimeter_model_m"])
        for a, b in itertools.combinations(baselines, 2)
    ]
    null = max(nulls) if nulls else None
    changes, controls = [], []
    stress, refusals = [], []
    for row in rows:
        name, spec = row["name"], row["spec"]
        value = (
            row.get("estimated_perimeter_model_m")
            if row["state"] == "complete"
            else None
        )
        if spec["group"] == "change" and value is not None and null is not None:
            for base in baselines:
                truth = row["truth"]["actual_delta_model_m"]
                identity = {"before": base["name"], "after": name}
                changes.append(
                    {
                        **identity,
                        **delta_result(
                            value - base["estimated_perimeter_model_m"], truth, null
                        ),
                    }
                )
                # A constant mesh is evaluated by precisely the same criteria.
                controls.append({**identity, **delta_result(0, truth, 0)})
        if spec["group"] == "stress":
            change = (
                200 * abs(value - centre) / (value + centre)
                if value is not None and centre is not None
                else None
            )
            stress.append(
                {
                    "name": name,
                    "state": row["state"],
                    "change_symmetric_percent": change,
                    "input_geometry_changed": name in protocol["geometry_confounds"],
                    "rotation_confirmed": row.get("quality", {}).get(
                        "rotation_confirmed"
                    ),
                    "error": (row.get("reconstruction") or {}).get("error"),
                }
            )
            if (
                name in protocol["expected_rejections"]
                or name in protocol["expected_warning_or_rejection"]
            ):
                reasons = {}
                if row.get("session_id"):
                    path = root / "store" / row["session_id"]
                    for attempt in sorted(path.glob("attempt_*/frames.json")):
                        for frame in read_json(attempt)["frames"]:
                            if frame["state"] == "estimated":
                                continue
                            reason = frame.get("reason", "unknown")
                            reasons[reason] = reasons.get(reason, 0) + 1
                status = row.get("reconstruction") or {}
                qc_rejected = (
                    row["state"] == "failed"
                    and status.get("stage") == "failed"
                    and status.get("error", "").startswith("使用できるフレームが")
                    and bool(reasons)
                )
                # Hard rotation/residual rejections must carry matching
                # diagnostics; a crash or model/resource failure never counts.
                path = (
                    root / "store" / row["session_id"]
                    if row.get("session_id")
                    else None
                )
                if (
                    path
                    and row["state"] == "failed"
                    and status.get("stage") == "failed"
                ):
                    for diagnostic in path.glob("attempt_*/qc.json"):
                        qc = read_json(diagnostic)
                        if (
                            status.get("error", "").startswith("回転範囲が不足")
                            and qc.get("rotation_confirmed") is False
                        ):
                            qc_rejected = True
                    for diagnostic in path.glob("attempt_*/fit.json"):
                        fit = read_json(diagnostic)
                        limit = (
                            fit.get("research_options", {})
                            .get("qc", {})
                            .get("max_relative_fit_error")
                        )
                        if (
                            status.get("error", "").startswith(
                                "観測輪郭と推定形状の残差"
                            )
                            and limit is not None
                            and fit.get("relative_fit_error", 0) > limit
                        ):
                            qc_rejected = True
                warned = (
                    row["state"] == "complete"
                    and row.get("quality", {}).get("rotation_confirmed") is False
                )
                expected = (
                    "reject"
                    if name in protocol["expected_rejections"]
                    else "warn_or_reject"
                )
                refusals.append(
                    {
                        "name": name,
                        "expected": expected,
                        "qc_rejected": qc_rejected,
                        "warned": warned,
                        "met_expectation": qc_rejected
                        or (expected == "warn_or_reject" and warned),
                        "frame_reasons": reasons,
                    }
                )
    imaging = [
        s
        for s in stress
        if s["name"] in protocol["imaging_stress"]
        and s["change_symmetric_percent"] is not None
    ]
    limits = protocol["thresholds"]
    result = {
        "quality_refusals": [
            {"name": r["name"], "kind": refusal_kind(root, r)}
            for r in rows
            if refusal_kind(root, r)
        ],
        "unexpected_failures": [
            r["name"]
            for r in rows
            if r["state"] == "failed" and not refusal_kind(root, r)
        ],
        "source_kind": "synthetic",
        "physical_accuracy_validated": False,
        "case_count": data["case_count"],
        "unique_video_count": data.get("unique_video_count"),
        "duplicate_video_groups": data.get("duplicate_video_groups", []),
        "completed": data["completed"],
        "failed": data["failed"],
        "pending": data["pending"],
        "baseline_max_null_model_mm": null * 1000 if null is not None else None,
        "change_pairs": changes,
        "planned_change_pair_count": sum(r["spec"]["group"] == "baseline" for r in rows)
        * sum(r["spec"]["group"] == "change" for r in rows),
        "unavailable_change_pair_count": sum(
            r["spec"]["group"] == "baseline" for r in rows
        )
        * sum(r["spec"]["group"] == "change" for r in rows)
        - len(changes),
        "constant_mesh_negative_control": controls,
        "correct_direction_pairs": sum(c["correct_direction"] for c in changes),
        "direction_and_margin_pairs": sum(
            c["correct_direction"] and c["clears_null_margin"] for c in changes
        ),
        "negative_control_pass_pairs": sum(
            c["correct_direction"] and c["clears_null_margin"] for c in controls
        ),
        "distance_pair_count": len(data["distance_pairs"]),
        "planned_distance_pair_count": math.comb(
            sum(r["spec"]["group"] == "distance" for r in rows), 2
        ),
        "distance_max_symmetric_percent": max(
            (p["symmetric_percent"] for p in data["distance_pairs"]), default=None
        ),
        "accepted_imaging_stress_over_threshold": [
            s
            for s in imaging
            if s["change_symmetric_percent"]
            > limits["imaging_stress_false_change_percent"]
        ],
        "stress": stress,
        "refusal_expectations": refusals,
        "thresholds": limits,
        "limitations": data["limitations"],
    }
    replace_json(root / "evidence.json", result)
    return result


def artifacts(root: Path) -> dict:
    result = evidence(root)
    rows = read_json(root / "report.json")["results"]
    text = [
        "# 合成動画検証結果",
        "",
        "合成動画による探索的な結果。人体の実測精度・実撮影動作・採用ゲートは未検証。",
        "",
        f"計画 {result['case_count']} / 復元完了 {result['completed']} / 失敗 {result['failed']} / 未完了 {result['pending']}",
        f"動画hashで異なる入力 {result['unique_video_count']} / 同一入力の組 {result['duplicate_video_groups']}。同一入力は独立撮影として扱わない。",
        "",
        f"基準内の最大差: {result['baseline_max_null_model_mm']} model mm。",
        f"既知変化の方向一致: {result['correct_direction_pairs']}/{len(result['change_pairs'])} pair。",
        f"方向一致かつ基準内最大差の3倍超: {result['direction_and_margin_pairs']}/{len(result['change_pairs'])} pair。",
        f"一定mesh対照の通過: {result['negative_control_pass_pairs']}/{len(result['constant_mesh_negative_control'])} pair。",
        f"距離の最大対称差: {result['distance_max_symmetric_percent']}% ({result['distance_pair_count']} pair)。",
        "",
        "## ケース別結果",
        "",
        "|条件|状態|推定腹囲 model cm|回転網羅|エラー|",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        perimeter = row.get("estimated_perimeter_model_m")
        value = f"{perimeter * 100:.3f}" if perimeter is not None else "—"
        error = (row.get("reconstruction") or {}).get("error", "")
        text.append(
            f"|{row['name']}|{row['state']}|{value}|{row.get('quality', {}).get('rotation_confirmed', '—')}|{error.replace('|', '/')}|"
        )
    text += ["", "## 拒否・warning", ""]
    for row in result["refusal_expectations"]:
        text.append(
            f"- {row['name']}: 期待 {row['expected']} / QC拒否 {row['qc_rejected']} / warning {row['warned']} / 期待通り {row['met_expectation']} / {row['frame_reasons']}"
        )
    text += ["", "## 画像条件だけで変化が1%を超えた受理ケース", ""]
    for row in result["accepted_imaging_stress_over_threshold"]:
        text.append(f"- {row['name']}: {row['change_symmetric_percent']:.3f}%")
    text += ["", "## 解釈上の限界", ""] + ["- " + s for s in result["limitations"]]
    if (root / "attribution.json").exists():
        attribution = read_json(root / "attribution.json")["rows"]
        base = next((r for r in attribution if r["name"] == "baseline_17"), None)
        if base:
            text += [
                "",
                "## 初期推定とfitの切り分け",
                "",
                "baseline_17との比較。同一profileの断面・scaleで計算。",
                "",
                "|変形条件|初期推定の変化 model mm|fit後の変化 model mm|",
                "|---|---:|---:|",
            ]
            for row in attribution:
                if row["name"].startswith("change_"):
                    a = (
                        row["initial_perimeter_model_m"]
                        - base["initial_perimeter_model_m"]
                    ) * 1000
                    b = (
                        row["fitted_perimeter_model_m"]
                        - base["fitted_perimeter_model_m"]
                    ) * 1000
                    text.append(f"|{row['name']}|{a:.3f}|{b:.3f}|")
            text += [
                "",
                "初期推定の時点で過小反応があり、fitは反応を増やすが、合成変形の大きさには届かない。この比較だけでは人体・別のbody・別rendering条件での原因や性能は確定しない。",
            ]
    path = root / "results.md"
    path.write_text("\n".join(text) + "\n")
    path.chmod(0o600)
    # A self-contained interactive filter and previews. No remote resources.
    html_rows = []
    for row in rows:
        name, state = html.escape(row["name"]), html.escape(row["state"])
        perimeter = row.get("estimated_perimeter_model_m")
        value = f"{perimeter * 100:.3f}" if perimeter is not None else "—"
        preview = (
            f"<img loading='lazy' width='160' src='store/{row['session_id']}/preview_0.jpg'>"
            if row.get("session_id")
            else ""
        )
        error = html.escape((row.get("reconstruction") or {}).get("error", ""))
        html_rows.append(
            f"<tr data-state='{state}'><td>{name}</td><td>{state}</td><td>{value}</td><td>{html.escape(str(row.get('quality', {}).get('rotation_confirmed', '—')))}</td><td>{error}</td><td>{preview}</td></tr>"
        )
    page = (
        """<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>合成動画検証</title><style>body{font:16px system-ui;margin:32px;color:#222}table{border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}input,select{padding:8px}img{display:block}</style>
<h1>合成動画検証</h1><p>合成モデル上の結果です。人体の精度・実撮影・採用ゲートは未検証。</p>
<p>条件名 <input id='q'> 状態 <select id='s'><option value=''>すべて</option><option>complete</option><option>failed</option><option>pending</option></select></p>
<table><thead><tr><th>条件</th><th>状態</th><th>腹囲 model cm</th><th>回転網羅</th><th>エラー</th><th>入力例</th></tr></thead><tbody>"""
        + "".join(html_rows)
        + """</tbody></table>
<script>function filter(){for(const r of document.querySelectorAll('tbody tr'))r.hidden=!r.innerText.includes(q.value)||(s.value&&r.dataset.state!==s.value)}q.oninput=filter;s.onchange=filter;</script>"""
    )
    path = root / "results.html"
    path.write_text(page)
    path.chmod(0o600)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = artifacts(private_path(args.out))
    print({k: result[k] for k in ("completed", "failed", "pending")})
