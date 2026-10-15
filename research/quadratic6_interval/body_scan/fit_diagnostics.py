"""Detached objective attribution; these values never modify a fit."""

from __future__ import annotations


def objective_snapshot(terms, parameters):
    """Return scalar terms and gradients without accumulating optimizer grads."""
    import torch

    active = [
        (name, value) for name, value in parameters.items() if value.requires_grad
    ]
    result = {}
    for name, term in terms.items():
        gradients = (
            torch.autograd.grad(
                term,
                [value for _, value in active],
                retain_graph=True,
                allow_unused=True,
            )
            if term.requires_grad and active
            else [None] * len(active)
        )
        result[name] = {
            "value": float(term.detach()),
            "gradient_norm": {
                key: 0.0 if grad is None else float(grad.detach().norm())
                for (key, _), grad in zip(active, gradients)
            },
            "gradient_components": {
                key: (None if grad is None else grad.detach().cpu().tolist())
                for (key, value), grad in zip(active, gradients)
                if value.numel() <= 256
            },
        }
    return result
