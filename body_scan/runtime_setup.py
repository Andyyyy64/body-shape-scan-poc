"""Reproduce the tested Mac runtime from pinned official sources and local-only weights."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .io import InputError, new_run, sha256, write_json

# Immutable dependency revisions, not environment/user identifiers.
SAM_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
DINO_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
MODEL_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
SEGMENTATION_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"
)
SEGMENTATION_SHA256 = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"


def extract_source(payload: bytes, target: Path):
    target.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = Path(*Path(member.name).parts[1:])
            path = (target / relative).resolve()
            if not path.is_relative_to(target.resolve()):
                raise InputError("Invalid source archive path")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(archive.extractfile(member).read())


def patch_source(root: Path):
    edits = {
        "sam_3d_body/sam_3d_body_estimator.py": [
            (
                'batch = recursive_to(batch, "cuda")',
                "batch = recursive_to(batch, self.device)",
            )
        ],
        "sam_3d_body/models/meta_arch/sam3d_body.py": [
            (
                ".repeat(B, N, 1, 1, 1)\n            .cuda()",
                '.repeat(B, N, 1, 1, 1)\n            .to(batch["img"].device)',
            )
        ],
        "sam_3d_body/build_models.py": [
            (
                'torch.load(checkpoint_path, map_location="cpu", weights_only=False)',
                'torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)',
            )
        ],
        "sam_3d_body/models/backbones/dinov3.py": [
            ("import torch\n", "import os\nimport torch\n"),
            ('"facebookresearch/dinov3",', 'os.environ["BODY_SCAN_DINO_SOURCE"],'),
            ('source="github",', 'source="local",'),
            (
                "        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]\n\n        return y",
                "        output_device = x.device\n        parameter = next(self.encoder.parameters())\n        x = x.to(device=parameter.device, dtype=parameter.dtype)\n        y = self.encoder.get_intermediate_layers(x, n=1, reshape=True, norm=True)[-1]\n\n        return y.to(output_device)",
            ),
        ],
    }
    for name, replacements in edits.items():
        path = root / name
        source = path.read_text()
        for old, new in replacements:
            if source.count(old) != 1:
                raise InputError(
                    "Upstream source differs from the supported revision; no compatibility fallback was applied"
                )
            source = source.replace(old, new)
        path.write_text(source)
    return hashlib.sha256(b"".join((root / p).read_bytes() for p in edits)).hexdigest()


def setup(output: str, frames: int = 16) -> dict:
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").exists():
        raise InputError(
            "This model setup targets macOS with native process network isolation"
        )
    if not 3 <= frames <= 64:
        raise InputError("Select 3..64 frames per recording")
    # Verify entitlement before creating or installing a runtime; no token is put in command arguments.
    subprocess.run(
        [
            "uvx",
            "--from",
            "huggingface_hub",
            "hf",
            "download",
            "facebook/sam-3d-body-dinov3",
            "model.ckpt",
            "--revision",
            MODEL_REVISION,
            "--dry-run",
        ],
        check=True,
    )
    root = new_run(output)
    python = root / "venv/bin/python"
    subprocess.run(["uv", "venv", "--python", "3.11", str(root / "venv")], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(Path(__file__).with_name("runtime-requirements.txt")),
        ],
        check=True,
    )
    model = root / "models/sam-3d-body-dinov3"
    subprocess.run(
        [
            "uvx",
            "--from",
            "huggingface_hub",
            "hf",
            "download",
            "facebook/sam-3d-body-dinov3",
            "--revision",
            MODEL_REVISION,
            "--local-dir",
            str(model),
            "--max-workers",
            "1",
        ],
        check=True,
        env={**os.environ, "HF_HUB_DISABLE_XET": "1"},
    )
    for repo, revision, name in [
        ("facebookresearch/sam-3d-body", SAM_COMMIT, "sam-source"),
        ("facebookresearch/dinov3", DINO_COMMIT, "dino-source"),
    ]:
        with urllib.request.urlopen(
            f"https://api.github.com/repos/{repo}/tarball/{revision}", timeout=60
        ) as response:
            payload = response.read(64 * 1024 * 1024 + 1)
        if len(payload) > 64 * 1024 * 1024:
            raise InputError("Unexpectedly large source archive")
        extract_source(payload, root / name)
        (root / name / "UPSTREAM_COMMIT").write_text(revision + "\n")
    patched_hash = patch_source(root / "sam-source")
    segmentation = root / "models/person-seg.pt"
    with (
        urllib.request.urlopen(SEGMENTATION_URL, timeout=60) as response,
        segmentation.open("xb") as handle,
    ):
        while block := response.read(1024 * 1024):
            handle.write(block)
    if sha256(segmentation) != SEGMENTATION_SHA256:
        raise InputError("Segmentation checkpoint checksum mismatch")
    config = {
        "python": str(python),
        "sam_source": str(root / "sam-source"),
        "dino_source": str(root / "dino-source"),
        "checkpoint": str(model / "model.ckpt"),
        "mhr_model": str(model / "assets/mhr_model.pt"),
        "segmentation_model": str(segmentation),
        "encoder_device": "mps",
        "cpu_threads": 2,
        "sample_frames": frames,
        "max_rss_mb": 6144,
        "min_available_mb": 1200,
        "mps_memory_fraction": 0.5,
        "network_sandbox": "macos",
        "pipeline_version": "sam_parameter_ensemble.v1",
        "sam_commit": SAM_COMMIT,
        "dino_commit": DINO_COMMIT,
        "source_patch_sha256": patched_hash,
    }
    for key in ("checkpoint", "mhr_model", "segmentation_model"):
        config[key + "_sha256"] = sha256(Path(config[key]))
    config["runtime_packages"] = subprocess.check_output(
        ["uv", "pip", "freeze", "--python", str(python)], text=True
    ).splitlines()
    config["engine_id"] = hashlib.sha256(
        json.dumps(
            {
                k: v
                for k, v in config.items()
                if k.endswith("_sha256")
                or k.endswith("_commit")
                or k in ["encoder_device", "pipeline_version", "runtime_packages"]
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    write_json(root / "config.json", config)
    return {
        "status": "runtime_installed_inference_not_yet_verified",
        "config_file": str(root / "config.json"),
    }
