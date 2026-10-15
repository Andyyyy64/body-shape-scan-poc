"""Declared error assumptions and grouped, paired validation; no adoption shortcut."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist, mean, stdev

from .geometry import number
from .io import InputError

NORMAL = NormalDist()


def detection_power(sigma_m: float, delta_m: float, alpha: float = 0.05) -> dict:
    sigma = number(sigma_m, "sigma_m", positive=True)
    delta = abs(number(delta_m, "delta_m"))
    alpha = number(alpha, "alpha", positive=True)
    if alpha >= 1:
        raise InputError("alpha must be below 1")
    z = NORMAL.inv_cdf(1 - alpha / 2)
    sd = math.sqrt(2) * sigma
    power = 1 - NORMAL.cdf(z - delta / sd) + NORMAL.cdf(-z - delta / sd)
    return {
        "sigma_m": sigma,
        "delta_m": delta,
        "alpha": alpha,
        "threshold_m": z * sd,
        "power": power,
        "assumptions": "independent, equal-variance normal errors; known sigma; two-sided test",
    }


def wilson(successes: int, total: int) -> dict:
    if total == 0:
        return {"n": 0, "rate": None, "interval_95": None}
    z = NORMAL.inv_cdf(0.975)
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return {
        "n": total,
        "rate": p,
        "interval_95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def analyze_pairs(data: dict, threshold_m: float, target_m: float) -> dict:
    if data.get("schema_version") != "paired_validation.v1":
        raise InputError("Expected paired_validation.v1")
    if data.get("source_kind") not in {"synthetic", "real"}:
        raise InputError("source_kind required")
    threshold = number(threshold_m, "threshold_m", positive=True)
    target = number(target_m, "target_m", positive=True)
    groups = defaultdict(list)
    seen = set()
    for row in data["pairs"]:
        required = (
            "pair_id",
            "subject_id",
            "baseline_session_id",
            "followup_session_id",
            "device_class",
            "part",
            "method",
            "measurement_definition_id",
            "status",
        )
        if any(not isinstance(row.get(k), str) or not row[k] for k in required):
            raise InputError(
                "Pair identifiers and grouping fields must be nonempty strings"
            )
        key = tuple(
            row[k]
            for k in ("device_class", "part", "method", "measurement_definition_id")
        )
        identity = (key, row["pair_id"])
        if identity in seen:
            raise InputError("Duplicate pair within a method group")
        if row["baseline_session_id"] == row["followup_session_id"]:
            raise InputError(
                "Two laps of one session are not independent paired sessions"
            )
        if row.get("split") != "evaluation":
            raise InputError("Only declared evaluation pairs are accepted")
        if row["status"] not in {"ok", "failed", "missing", "unscaled"}:
            raise InputError("Unknown pair status")
        number(row["reference_delta_m"], "reference_delta_m")
        number(row["reference_uncertainty_m"], "reference_uncertainty_m")
        if row["reference_uncertainty_m"] < 0:
            raise InputError("Reference uncertainty cannot be negative")
        if row["status"] == "ok":
            number(row["predicted_delta_m"], "predicted_delta_m")
        elif row.get("predicted_delta_m") is not None:
            raise InputError("Failed pairs must not contain a substituted prediction")
        seen.add(identity)
        groups[key].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        valid = [r for r in rows if r["status"] == "ok"]
        errors = [r["predicted_delta_m"] - r["reference_delta_m"] for r in valid]
        nulls = [
            r
            for r in rows
            if r["reference_delta_m"] == 0 and r["reference_uncertainty_m"] == 0
        ]
        positives = [
            r
            for r in rows
            if abs(r["reference_delta_m"]) - r["reference_uncertainty_m"] >= target
        ]

        def detected(r):
            return (
                r["status"] == "ok"
                and abs(r["predicted_delta_m"]) > threshold
                and r["predicted_delta_m"] * r["reference_delta_m"] > 0
            )

        false = sum(
            r["status"] == "ok" and abs(r["predicted_delta_m"]) > threshold
            for r in nulls
        )
        sensitivity = wilson(sum(detected(r) for r in positives), len(positives))
        fpr = wilson(false, len(nulls))
        bias = mean(errors) if errors else None
        sd = stdev(errors) if len(errors) >= 2 else None
        # These are descriptive pair intervals. Shared subjects/baselines are not independent trials.
        sessions = [
            (r["subject_id"], r[k])
            for r in rows
            for k in ("baseline_session_id", "followup_session_id")
        ]
        correlated = len(set(sessions)) < len(sessions) or len(
            {r["subject_id"] for r in rows}
        ) < len(rows)
        results.append(
            {
                "device_class": key[0],
                "part": key[1],
                "method": key[2],
                "measurement_definition_id": key[3],
                "attempts": len(rows),
                "usable": len(valid),
                "usable_rate": len(valid) / len(rows),
                "bias_m": bias,
                "mae_m": mean(map(abs, errors)) if errors else None,
                "loa95_m": [bias - 1.96 * sd, bias + 1.96 * sd]
                if sd is not None
                else None,
                "sensitivity_all_attempts": sensitivity,
                "false_positive_all_attempts": fpr,
                "false_positive_usable": wilson(
                    false, sum(r["status"] == "ok" for r in nulls)
                ),
                "null_pairs_with_unambiguous_reference": len(nulls),
                "correlated_pairs": correlated,
                "interval_caveat": "descriptive Wilson intervals; not cluster-adjusted",
                "decision": "insufficient_evidence",
            }
        )
    return {
        "schema_version": "validation_report.v1",
        "source_kind": data["source_kind"],
        "threshold_m": threshold,
        "target_m": target,
        "groups": results,
        "physical_accuracy_validated": False,
        "decision": "insufficient_evidence",
        "reason": "Exploratory summaries do not establish an independent confirmatory human validation.",
    }
