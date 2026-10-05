"""Post-hoc attribution using saved estimates; never alters experimental inputs."""

from __future__ import annotations

import argparse
import fcntl
import os
import threading
import time
from pathlib import Path

from .io import InputError, read_json, replace_json, sha256
from .synthetic_lab import launch


def worker(config_path: str, root: Path, capacity: bool = False):
    import numpy as np
    import psutil
    import torch

    from .mesh import section_perimeter

    config = read_json(config_path)
    lock = (Path(config_path).resolve().parent / "inference.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def guard():
        while True:
            if (
                psutil.Process().memory_info().rss > config["max_rss_mb"] * 1024**2
                or psutil.virtual_memory().available
                < config["min_available_mb"] * 1024**2
            ):
                os._exit(75)
            time.sleep(0.1)

    threading.Thread(target=guard, daemon=True).start()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    if sha256(Path(config["mhr_model"])) != config["mhr_model_sha256"]:
        raise InputError("MHR checksum mismatch")
    model = torch.jit.load(config["mhr_model"], map_location="cpu")
    with np.load(root / "body.npz") as body:
        faces = body["faces"].tolist()
        original = body["vertices"]
    if capacity:
        from .synthetic_lab import deform

        body = read_json(root / "body.json")
        body["faces"] = np.asarray(faces)
        zero_params = torch.zeros(1, 204)

        def generate(shape):
            with torch.no_grad():
                vertices, _ = model(
                    torch.tensor(shape, dtype=torch.float32)[None],
                    zero_params,
                    torch.zeros(1, 72),
                )
            return vertices[0].numpy() / 100

        zero = np.zeros(45, np.float32)
        base = generate(zero)
        if np.max(abs(base - original)) > 1e-6:
            raise InputError("Capacity oracle does not match the generator base")
        basis = np.stack(
            [generate(np.eye(45, dtype=np.float32)[i]) - base for i in range(45)],
            axis=-1,
        )
        torso = (
            (base[:, 1] >= body["hip_y"])
            & (base[:, 1] <= body["shoulder_y"])
            & (abs(base[:, 0]) < 0.25)
        )
        abdomen = torso & (abs(base[:, 1] - body["section_y"]) < 0.12)
        section = {"y": body["section_y"], "center_xz": body["center_xz"]}
        rows = []
        for delta in (-0.02, -0.01, 0.01, 0.02, 0.04):
            target, before, after = deform(original, body, delta)
            for region, selected in (
                ("whole_mesh", np.ones(len(base), bool)),
                ("torso", torso),
                ("abdomen_band", abdomen),
            ):
                coefficients, _, rank, _ = np.linalg.lstsq(
                    basis[selected].reshape(-1, 45).astype(float),
                    (target - base)[selected].reshape(-1).astype(float),
                    rcond=1e-6,
                )
                fitted = generate(coefficients)
                actual = section_perimeter(
                    {"vertices": fitted.tolist(), "faces": faces}, section
                )
                if actual is None:
                    raise InputError("Oracle-fitted section is ambiguous")
                rows.append(
                    {
                        "region": region,
                        "truth_delta_model_mm": (after - before) * 1000,
                        "fitted_delta_model_mm": (actual - before) * 1000,
                        "surface_rms_model_mm": float(
                            np.sqrt(np.mean((fitted[selected] - target[selected]) ** 2))
                            * 1000
                        ),
                        "linearity_max_error_model_mm": float(
                            np.max(abs(fitted - (base + basis @ coefficients))) * 1000
                        ),
                        "basis_rank": int(rank),
                    }
                )
        replace_json(
            root / "capacity.json",
            {
                "source_kind": "synthetic",
                "diagnostic_only_oracle_geometry": True,
                "physical_accuracy_validated": False,
                "rows": rows,
                "interpretation": "Known 3D vertices directly fitted with fixed generator scales, no image inference or regularization. Region-dependent least-squares result, not a proof of a global capacity ceiling or attainable image-based accuracy.",
            },
        )
        return
    profile = read_json(root / "store" / "profile.json")
    if read_json(root / "store" / "store.json").get("synthetic_experiment") is not True:
        raise InputError("Diagnostics requires a synthetic experiment store")
    params = torch.zeros(1, 204)
    params[0, 136:] = torch.tensor(profile["skeleton_scales"])
    section = profile["sections"][0]
    manifest = read_json(root / "matrix.json")
    rows = []
    for case in manifest["cases"]:
        if case["state"] != "complete":
            continue
        path = root / "store" / case["session_id"]
        fits = sorted(path.glob("attempt_*/fit.json"))
        if len(fits) != 1:
            raise InputError("Expected exactly one successful fit per capture")
        fit = read_json(fits[0])
        row = {"name": case["name"]}
        for label, key in (
            ("initial", "initial_shape_params"),
            ("fitted", "shape_params"),
        ):
            with torch.no_grad():
                vertices, _ = model(
                    torch.tensor(fit[key])[None], params, torch.zeros(1, 72)
                )
            perimeter = section_perimeter(
                {"vertices": (vertices[0].numpy() / 100).tolist(), "faces": faces},
                section,
            )
            row[label + "_perimeter_model_m"] = perimeter
        row.update(
            initial_loss_px=fit["initial_loss_px"], final_loss_px=fit["final_loss_px"]
        )
        # Per-frame estimates are preserved even for excluded frames. Include
        # only estimated frames when comparing initialization vs fitted output.
        frames = read_json(fits[0].parent / "frames.json")["frames"]
        initials, scales = [], []
        for frame in frames:
            if frame["state"] == "estimated":
                with np.load(fits[0].parent / frame["output"]) as estimate:
                    initials.append(estimate["shape_params"])
                    scales.append(estimate["mhr_model_params"][136:])
        row["shape_initialization_matches_usable_frame_median"] = bool(
            np.allclose(np.median(initials, axis=0), fit["initial_shape_params"])
        )
        native_scales = np.median(scales, axis=0)
        difference = native_scales - np.asarray(profile["skeleton_scales"])
        row["native_scale_delta_rms"] = float(np.sqrt(np.mean(difference**2)))
        row["native_scale_delta_max_abs"] = float(np.max(abs(difference)))
        native_params = params.clone()
        native_params[0, 136:] = torch.tensor(native_scales)
        with torch.no_grad():
            vertices, _ = model(
                torch.tensor(fit["initial_shape_params"])[None],
                native_params,
                torch.zeros(1, 72),
            )
        row["initial_native_scale_perimeter_model_m"] = section_perimeter(
            {"vertices": (vertices[0].numpy() / 100).tolist(), "faces": faces}, section
        )
        row["fit_camera_delta_depth_mean_model_m"] = float(
            np.asarray(fit["camera_delta"])[:, 2].mean()
        )
        rows.append(row)
    replace_json(
        root / "attribution.json",
        {
            "source_kind": "synthetic",
            "physical_accuracy_validated": False,
            "rows": rows,
            "interpretation": "Initial and final mesh share profile scales and section; no refitting or ground-truth information was supplied to estimation.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if args.worker:
        worker(args.runtime_config, root, args.capacity)
    else:
        code = launch(
            args.runtime_config,
            [
                "-m",
                "body_scan.synthetic_diagnostics",
                "--worker",
                "--runtime-config",
                args.runtime_config,
                "--out",
                str(root),
                *(["--capacity"] if args.capacity else []),
            ],
            root / ("diagnostics_" + str(time.time_ns()) + ".log"),
        )
        if code:
            raise InputError(f"Diagnostics failed ({code}); see private log")
        print(
            "Saved oracle capacity diagnostics"
            if args.capacity
            else "Saved initial/fitted attribution"
        )
