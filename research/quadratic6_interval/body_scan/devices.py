"""Explicit encoder placement; the fitting and MHR heads stay on CPU."""

from __future__ import annotations

from .io import InputError


def configure_device(torch, config: dict) -> str:
    device = config["encoder_device"]
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise InputError(
                "Configured MPS device is unavailable; no fallback was used"
            )
        torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise InputError(
                "Configured CUDA device is unavailable; no fallback was used"
            )
        fraction = config.get("cuda_memory_fraction", 0.5)
        if not 0 < fraction <= 1:
            raise InputError(
                "CUDA memory fraction must be greater than zero and at most one"
            )
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.cuda.reset_peak_memory_stats(0)
        # Execute a kernel before loading large models; detects unsupported GPU builds.
        try:
            torch.ones(1, device="cuda").add_(1)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            raise InputError(
                "Configured CUDA runtime cannot execute on this GPU; no fallback was used"
            ) from exc
    elif device != "cpu":
        raise InputError("Unknown encoder device; select mps, cuda or cpu")
    return device
