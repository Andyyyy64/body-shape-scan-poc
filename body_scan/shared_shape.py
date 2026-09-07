"""Experimental fixed-pose, shared-shape silhouette fit; one recording per call.

The contour topology and per-frame SAM poses/scales are held fixed. This is
not a calibrated measurement or a complete joint pose/camera reconstruction.
"""

from __future__ import annotations

from .io import InputError


def fit_observations(
    base,
    basis,
    target,
    camera,
    focal,
    principal,
    shape_scale,
    *,
    steps=300,
    learning_rate=0.01,
    shape_weight=0.1,
    camera_weight=10.0,
    optimize_camera=True,
):
    """Fit one shared shape delta to independent frame observations (pixels).

    Inputs: base [F,N,3], linear shape basis [F,N,3,D], boundary targets
    [F,M,2], camera translation [F,3], focal [F], principal point [F,2].
    No other recording, date, reference measurement or target delta is accepted.
    """
    import torch

    items = [
        torch.as_tensor(x, dtype=torch.float32).detach()
        for x in (base, basis, target, camera, focal, principal, shape_scale)
    ]
    base, basis, target, camera, focal, principal, scale = items
    if (
        base.ndim != 3
        or base.shape[-1] != 3
        or basis.ndim != 4
        or basis.shape[:3] != base.shape
        or target.ndim != 3
        or target.shape[0] != base.shape[0]
        or target.shape[-1] != 2
        or camera.shape != (base.shape[0], 3)
        or focal.shape != (base.shape[0],)
        or principal.shape != (base.shape[0], 2)
        or scale.shape != (basis.shape[-1],)
        or min(base.shape[:2]) < 1
        or target.shape[1] < 1
        or any(not torch.isfinite(x).all() for x in items)
        or (scale <= 0).any()
        or (focal <= 0).any()
        or not 1 <= steps <= 2000
        or not 0 < learning_rate <= 0.1
        or not 0 <= shape_weight <= 100
        or not 0 < camera_weight <= 1000
    ):
        raise InputError("Invalid shared-shape observations or optimizer settings")
    if base.shape[0] * base.shape[1] * target.shape[1] > 8_000_000:
        raise InputError("Contour fit exceeds the bounded CPU work budget")
    delta = torch.zeros(basis.shape[-1], requires_grad=True)
    camera_delta = torch.zeros_like(camera, requires_grad=optimize_camera)
    parameters = [delta, camera_delta] if optimize_camera else [delta]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def objective():
        vertices = base + torch.einsum("fncd,d->fnc", basis, delta)
        position = vertices + (camera + camera_delta)[:, None, :]
        if (position[:, :, 2] <= 0.1).any():
            raise InputError("Contour fit moved a point behind the camera")
        projection = (
            position[:, :, :2] / position[:, :, 2:] * focal[:, None, None]
            + principal[:, None, :]
        )
        # ponytail: fixed contour correspondences/topology; rerasterize and jointly
        # refine poses when held-out evidence shows this local model is inadequate.
        distances = torch.cdist(projection, target)
        observation = (
            (distances.min(2).values.square() + 4).sqrt().mean()
            + (distances.min(1).values.square() + 4).sqrt().mean()
        ) / 2
        prior = (
            shape_weight * (delta / scale).square().mean()
            + camera_weight * camera_delta.square().mean()
        )
        return observation, observation + prior

    initial_loss = float(objective()[0].detach())
    for _ in range(steps):
        optimizer.zero_grad()
        _, loss = objective()
        if not torch.isfinite(loss):
            raise InputError("Nonfinite contour fit")
        loss.backward()
        optimizer.step()
    observation, _ = objective()
    final_loss = float(observation.detach())
    if final_loss > initial_loss + 1e-4:
        raise InputError("Contour fit did not improve the observations")
    return {
        "shape_delta": delta.detach().numpy(),
        "camera_delta": camera_delta.detach().numpy(),
        "initial_loss_px": initial_loss,
        "final_loss_px": final_loss,
        "physical_accuracy_validated": False,
    }
