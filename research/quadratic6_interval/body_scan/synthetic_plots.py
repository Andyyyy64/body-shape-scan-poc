"""Standalone figures for synthetic evidence; no confidence-interval claim."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io import read_json


def plot(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_json(root / "report.json")
    evidence = read_json(root / "evidence.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    pairs = evidence["change_pairs"]
    axes[0].axhline(0, color="#999", linewidth=0.7)
    axes[0].plot([-25, 45], [-25, 45], "--", color="#777", label="Exact response")
    xs = sorted({p["truth_model_mm"] for p in pairs})
    for i, x in enumerate(xs):
        ys = [p["prediction_model_mm"] for p in pairs if p["truth_model_mm"] == x]
        axes[0].scatter(
            [x] * len(ys),
            ys,
            s=30,
            color="#2563a6",
            label="Compared with 3 baselines" if i == 0 else None,
        )
    null = evidence["baseline_max_null_model_mm"]
    if null is not None:
        axes[0].axhspan(
            -3 * null,
            3 * null,
            color="#bc4b51",
            alpha=0.1,
            label="3 x maximum baseline difference",
        )
    axes[0].set(
        xlabel="Injected perimeter change (model mm)",
        ylabel="Estimated change (model mm)",
        title="Known local changes",
    )
    axes[0].legend(fontsize=8)
    distances = [
        r
        for r in data["results"]
        if r["spec"]["group"] == "distance"
        and r.get("estimated_perimeter_model_m") is not None
    ]
    axes[1].scatter(
        [r["spec"]["distance"] for r in distances],
        [r["estimated_perimeter_model_m"] * 100 for r in distances],
        color="#2563a6",
    )
    axes[1].set(
        xlabel="Rendered camera distance (model m)",
        ylabel="Estimated perimeter (model cm)",
        title=f"Distance: max pair difference {evidence['distance_max_symmetric_percent']:.2f}%"
        if evidence["distance_max_symmetric_percent"] is not None
        else "Distance",
    )
    fig.suptitle("Synthetic only; not human accuracy evidence", fontsize=11)
    fig.savefig(root / "sensitivity-distance.png", dpi=160)
    plt.close(fig)
    stress = evidence["stress"]
    fig, ax = plt.subplots(
        figsize=(10, max(4, len(stress) * 0.3)), layout="constrained"
    )
    for i, row in enumerate(stress):
        value = row["change_symmetric_percent"]
        if value is None:
            ax.text(0, i, row["state"], fontsize=8, va="center", color="#777")
        else:
            ax.barh(
                i,
                value,
                color="#c98431" if row["input_geometry_changed"] else "#2563a6",
                alpha=0.8,
            )
            if row["rotation_confirmed"] is False:
                ax.text(value, i, " warning", fontsize=8, va="center")
    ax.set_yticks(
        range(len(stress)), [s["name"].removeprefix("stress_") for s in stress]
    )
    ax.invert_yaxis()
    ax.axvline(1, color="#bc4b51", linestyle="--", label="Exploratory 1% threshold")
    ax.set(
        xlabel="Symmetric difference from baseline mean (%)",
        title="Stress cases: blue image changes / orange geometry confounds",
    )
    ax.legend(fontsize=8)
    fig.savefig(root / "stress.png", dpi=160)
    plt.close(fig)
    for name in ("sensitivity-distance.png", "stress.png"):
        (root / name).chmod(0o600)
    # A compact, annotated contact sheet of actual inputs.
    import cv2
    import numpy as np

    selected = [
        r
        for r in data["results"]
        if r.get("session_id")
        and r["name"]
        in {
            "baseline_17",
            "stress_dim",
            "stress_backlight",
            "stress_blur",
            "stress_large_occlusion",
            "stress_two_people",
            "stress_cropped_head",
            "stress_combined",
        }
    ]
    if selected:
        sheet = np.full(
            (int(np.ceil(len(selected) / 4)) * 232, 4 * 210, 3), 245, np.uint8
        )
        for i, row in enumerate(selected):
            image = cv2.imread(
                str(root / "store" / row["session_id"] / "preview_0.jpg")
            )
            if image is None:
                continue
            x, y = (i % 4) * 210, (i // 4) * 232
            sheet[y : y + 210, x : x + 210] = cv2.resize(image, (210, 210))
            cv2.putText(
                sheet,
                row["name"].removeprefix("stress_"),
                (x + 4, y + 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (30, 30, 30),
                1,
            )
        cv2.imwrite(str(root / "inputs.jpg"), sheet)
        (root / "inputs.jpg").chmod(0o600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot(Path(args.out).resolve())
