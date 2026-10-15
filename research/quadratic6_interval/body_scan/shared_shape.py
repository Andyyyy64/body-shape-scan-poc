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
    base_weights=None,
    target_weights=None,
    observation_scale=None,
    anchor_base=None,
    anchor_basis=None,
    anchor_target=None,
    anchor_weights=None,
    anchor_weight=0.0,
    pose_limit_deg=0.0,
    pose_weight=1.0,
    diagnostics=False,
    contour_softening=2.0,
    anchor_softening=2.0,
    camera_mode=None,
    distance_mode="legacy",
    local_count=0,
    local_limit=0.1,
    local_weight=0.01,
    global_shape=True,
    optimizer_mode="adam",
):
    """Fit one shared shape delta to independent frame observations (pixels).

    Inputs: base [F,N,3], linear shape basis [F,N,3,D], boundary targets
    [F,M,2], camera translation [F,3], focal [F], principal point [F,2].
    No other recording, date, reference measurement or target delta is accepted.
    """
    import torch

    if camera_mode not in (
        None,
        "free",
        "fixed",
        "fixed_depth",
        "shared_depth",
        "shared",
    ):
        raise InputError("Invalid camera fit mode")
    if camera_mode is not None:
        optimize_camera = camera_mode != "fixed"
    if not 0.05 <= contour_softening <= 8 or not 0.05 <= anchor_softening <= 8:
        raise InputError("Invalid robust-loss softening")
    if (
        local_count not in (0, 1, 2)
        or not 0.01 <= local_limit <= 0.25
        or not 0 <= local_weight <= 100
        or not isinstance(global_shape, bool)
    ):
        raise InputError("Invalid anatomical residual settings")

    if optimizer_mode not in ("adam", "adam_lbfgs64"):
        raise InputError("Invalid contour optimizer mode")
    dtype = torch.float64 if optimizer_mode == "adam_lbfgs64" else torch.float32
    items = [
        torch.as_tensor(x, dtype=dtype).detach()
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
    if local_count and basis.shape[-1] <= local_count:
        raise InputError("Local residual requires a global shape basis")
    if not local_count and not global_shape:
        raise InputError("Cannot disable global shape without a local residual")

    def weights(value, shape):
        if value is None:
            return None
        result = torch.as_tensor(value, dtype=dtype).detach()
        if (
            result.shape != shape
            or not torch.isfinite(result).all()
            or (result <= 0).any()
        ):
            raise InputError("Invalid contour observation weights")
        return result

    base_weights = weights(base_weights, base.shape[:2])
    target_weights = weights(target_weights, target.shape[:2])
    normalizer = weights(observation_scale, (base.shape[0],))
    if (
        not 0 <= anchor_weight <= 32
        or not 0 <= pose_limit_deg <= 10
        or not 0 <= pose_weight <= 100
    ):
        raise InputError("Invalid nuisance fit settings")
    anchors = (anchor_base, anchor_basis, anchor_target)
    if any(x is not None for x in anchors):
        if not all(x is not None for x in anchors):
            raise InputError("Incomplete landmark observations")
        anchor_base, anchor_basis, anchor_target = [
            torch.as_tensor(x, dtype=dtype).detach() for x in anchors
        ]
        if (
            anchor_base.ndim != 3
            or anchor_base.shape[0] != base.shape[0]
            or anchor_base.shape[-1] != 3
            or anchor_base.shape[1] < 4
            or anchor_basis.shape != (*anchor_base.shape, basis.shape[-1])
            or anchor_target.shape != (*anchor_base.shape[:2], 2)
            or any(
                not torch.isfinite(x).all()
                for x in (anchor_base, anchor_basis, anchor_target)
            )
        ):
            raise InputError("Invalid landmark observations")
    elif anchor_weight or pose_limit_deg:
        raise InputError("Nuisance refinement requires landmark observations")
    if anchor_weights is not None:
        anchor_weights = torch.as_tensor(anchor_weights, dtype=dtype).detach()
        if (
            anchor_base is None
            or anchor_weights.shape != anchor_base.shape[:2]
            or not torch.isfinite(anchor_weights).all()
            or (anchor_weights < 0).any()
            or (anchor_weights > 1).any()
            or anchor_weights.sum() <= 0
        ):
            raise InputError("Invalid landmark confidence weights")
    delta = torch.zeros(basis.shape[-1], dtype=dtype, requires_grad=True)
    camera_delta = torch.zeros_like(
        camera, requires_grad=optimize_camera and camera_mode != "shared"
    )
    parameters = (
        [delta, camera_delta]
        if optimize_camera and camera_mode != "shared"
        else [delta]
    )
    shared_depth = torch.zeros(
        (), dtype=dtype, requires_grad=camera_mode == "shared_depth"
    )
    camera_offset = torch.zeros_like(camera)
    if camera_mode == "shared_depth":
        camera_offset[:, 2] = camera[:, 2].median() - camera[:, 2]
        parameters.append(shared_depth)
    shared_translation = torch.zeros(
        3, dtype=dtype, requires_grad=camera_mode == "shared"
    )
    if camera_mode == "shared":
        camera_offset = camera.median(dim=0).values[None] - camera
        parameters.append(shared_translation)
    pose_raw = torch.zeros(
        base.shape[0], dtype=dtype, requires_grad=bool(pose_limit_deg)
    )
    if pose_limit_deg:
        parameters.append(pose_raw)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def effective_camera_delta():
        if camera_mode == "shared":
            return camera_offset + shared_translation[None]
        if camera_mode == "shared_depth":
            return (
                camera_delta * camera_delta.new_tensor([1, 1, 0])
                + camera_offset
                + torch.stack([shared_depth * 0, shared_depth * 0, shared_depth])
            )
        if camera_mode == "fixed_depth":
            return camera_delta * camera_delta.new_tensor([1, 1, 0])
        return camera_delta

    def effective_shape_delta():
        if not local_count:
            return delta
        global_delta = (
            delta[:-local_count]
            if global_shape
            else torch.zeros_like(delta[:-local_count])
        )
        return torch.cat([global_delta, delta[-local_count:].tanh() * local_limit])

    def project(vertices):
        if pose_limit_deg:
            angle = pose_raw.tanh() * (pose_limit_deg * torch.pi / 180)
            pivot = anchor_base.mean(1)[:, None, :]
            v = vertices - pivot
            c, s = angle.cos()[:, None], angle.sin()[:, None]
            vertices = (
                torch.stack(
                    (
                        c * v[:, :, 0] + s * v[:, :, 2],
                        v[:, :, 1],
                        -s * v[:, :, 0] + c * v[:, :, 2],
                    ),
                    -1,
                )
                + pivot
            )
        position = vertices + (camera + effective_camera_delta())[:, None, :]
        if (position[:, :, 2] <= 0.1).any():
            raise InputError("Contour fit moved a point behind the camera")
        return (
            position[:, :, :2] / position[:, :, 2:] * focal[:, None, None]
            + principal[:, None, :]
        )

    def objective():
        from .contour_distance import nearest_contour_distances

        effective_delta = effective_shape_delta()
        vertices = base + torch.einsum("fncd,d->fnc", basis, effective_delta)
        projection = project(vertices)
        # ponytail: fixed contour correspondences/topology; rerasterize and jointly
        # refine poses when held-out evidence shows this local model is inadequate.
        nearest_forward, nearest_backward = nearest_contour_distances(
            projection, target, distance_mode
        )
        forward = (nearest_forward.square() + 4).sqrt()
        backward = (nearest_backward.square() + 4).sqrt()
        normalized_forward = (
            nearest_forward
            if normalizer is None
            else nearest_forward / normalizer[:, None]
        )
        normalized_backward = (
            nearest_backward
            if normalizer is None
            else nearest_backward / normalizer[:, None]
        )
        fit_forward = (
            forward
            if normalizer is None
            else (normalized_forward.square() + contour_softening**2).sqrt()
        )
        fit_backward = (
            backward
            if normalizer is None
            else (normalized_backward.square() + contour_softening**2).sqrt()
        )
        if normalizer is None and contour_softening != 2:
            fit_forward = (nearest_forward.square() + contour_softening**2).sqrt()
            fit_backward = (nearest_backward.square() + contour_softening**2).sqrt()

        def average(values, weight):
            return (
                values.mean()
                if weight is None
                else ((values * weight).sum(1) / weight.sum(1)).mean()
            )

        observation = (
            average(fit_forward, base_weights) + average(fit_backward, target_weights)
        ) / 2
        unweighted = (forward.mean() + backward.mean()) / 2
        shape_prior = (
            shape_weight * (effective_delta / scale).square().mean()
            if not local_count
            else shape_weight
            * (effective_delta[:-local_count] / scale[:-local_count]).square().mean()
        )
        local_prior = delta.new_zeros(())
        if local_count:
            local_prior = (
                local_weight
                * (effective_delta[-local_count:] / scale[-local_count:])
                .square()
                .mean()
            )
        camera_adjustment = (
            effective_camera_delta() - camera_offset
            if camera_mode in ("shared_depth", "shared")
            else effective_camera_delta()
        )
        camera_prior = camera_weight * camera_adjustment.square().mean()
        prior = shape_prior + camera_prior
        if local_count:
            prior = prior + local_prior
        landmark_loss = delta.new_zeros(())
        yaw_prior = delta.new_zeros(())
        if anchor_weight:
            anchor_projection = project(
                anchor_base + torch.einsum("fncd,d->fnc", anchor_basis, effective_delta)
            )
            error = anchor_projection - anchor_target
            if normalizer is not None:
                error = error / normalizer[:, None, None]
            robust_anchor = (error.square().sum(-1) + anchor_softening**2).sqrt()
            landmark_loss = anchor_weight * (
                robust_anchor.mean()
                if anchor_weights is None
                else (robust_anchor * anchor_weights).sum() / anchor_weights.sum()
            )
            prior = prior + landmark_loss
        if pose_limit_deg:
            yaw_prior = pose_weight * pose_raw.tanh().square().mean()
            prior = prior + yaw_prior
        weighted_pixels = (
            average(forward, base_weights) + average(backward, target_weights)
        ) / 2
        total = observation + prior
        terms = {
            "contour": observation,
            "shape_prior": shape_prior,
            "camera_prior": camera_prior,
            "landmark": landmark_loss,
            "yaw_prior": yaw_prior,
            "local_shape_prior": local_prior,
            "total": total,
        }
        return observation, total, unweighted, weighted_pixels, terms

    initial = objective()
    initial_loss = float(initial[0].detach())
    initial_objective = float(initial[1].detach())
    initial_pixels = float(initial[3].detach())
    diagnostic_parameters = {"shape": delta, "camera": camera_delta, "yaw": pose_raw}
    if camera_mode == "shared_depth":
        diagnostic_parameters["shared_depth"] = shared_depth
    if camera_mode == "shared":
        diagnostic_parameters["shared_translation"] = shared_translation
    if diagnostics:
        from .fit_diagnostics import objective_snapshot

        initial_diagnostics = objective_snapshot(initial[4], diagnostic_parameters)
    for _ in range(steps):
        optimizer.zero_grad()
        _, loss, _, _, _ = objective()
        if not torch.isfinite(loss):
            raise InputError("Nonfinite contour fit")
        loss.backward()
        optimizer.step()
    refinement = None
    if optimizer_mode == "adam_lbfgs64":
        before = float(objective()[1].detach())
        before_shape = effective_shape_delta().detach().tolist()
        before_camera = effective_camera_delta().detach().tolist()
        refined = torch.optim.LBFGS(
            parameters,
            lr=1,
            max_iter=100,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-16,
        )
        calls = [0]

        def closure():
            refined.zero_grad()
            loss = objective()[1]
            if not torch.isfinite(loss):
                raise InputError("Nonfinite refined contour objective")
            loss.backward()
            calls[0] += 1
            return loss

        refined.step(closure)
        closure()
        gradients = torch.cat(
            [x.grad.detach().flatten() for x in parameters if x.grad is not None]
        )
        state = refined.state[parameters[0]]
        refinement = {
            "objective_before": before,
            "before_shape_delta": before_shape,
            "before_camera_delta": before_camera,
            "objective_after": float(objective()[1].detach()),
            "gradient_inf_norm": float(gradients.abs().max()),
            "iterations": int(state.get("n_iter", 0)),
            "closure_calls": calls[0],
            "convergence_criterion": "gradient_inf_norm <= 1e-7",
            "converged": bool(gradients.abs().max() <= 1e-7),
        }
    observation, total, unweighted, weighted_pixels, terms = objective()
    final_loss = float(observation.detach())
    worsening = (
        float(total.detach()) > initial_objective + 1e-4
        if anchor_weight
        else final_loss > initial_loss + 1e-4
    )
    if worsening:
        raise InputError("Contour fit did not improve the observations")
    result = {
        "shape_delta": effective_shape_delta().detach().numpy(),
        "camera_delta": effective_camera_delta().detach().numpy(),
        "initial_loss_px": initial_pixels,
        "final_loss_px": float(weighted_pixels.detach()),
        "final_unweighted_loss_px": float(unweighted.detach()),
        "physical_accuracy_validated": False,
    }
    if diagnostics:
        result["objective_diagnostics"] = {
            "initial": initial_diagnostics,
            "final": objective_snapshot(terms, diagnostic_parameters),
            "steps": steps,
            "learning_rate": learning_rate,
            "termination": "fixed_step_budget",
            "shape_scale": scale.tolist(),
            "diagnostic_only": True,
        }
    if refinement is not None:
        result["optimizer_refinement"] = refinement
    if normalizer is not None:
        result["observation_loss_units"] = "reference_pixels_256_mask_height"
        result["initial_normalized_loss"] = initial_loss
        result["final_normalized_loss"] = final_loss
    if pose_limit_deg:
        result["yaw_delta_deg"] = (pose_raw.detach().tanh() * pose_limit_deg).numpy()
    if anchor_weight:
        with torch.no_grad():
            error = (
                project(
                    anchor_base
                    + torch.einsum("fncd,d->fnc", anchor_basis, effective_shape_delta())
                )
                - anchor_target
            )
        square = error.square().sum(-1)
        result["anchor_rms_px"] = float(
            (
                square.mean()
                if anchor_weights is None
                else (square * anchor_weights).sum() / anchor_weights.sum()
            ).sqrt()
        )
        if anchor_weights is not None:
            result["confident_anchor_count"] = int((anchor_weights > 0).sum())
    return result
