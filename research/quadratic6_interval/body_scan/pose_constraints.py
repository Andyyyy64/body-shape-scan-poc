"""Estimated camera tilt shared across a standing capture; heading stays free."""

from __future__ import annotations

import numpy as np

from .io import InputError


def root_rotation(parameters):
    """MHR native model parameters follow extrinsic xyz (global joint 1)."""
    import roma
    import torch

    parameters = np.asarray(parameters, np.float32)
    if parameters.shape[-1:] != (204,) or not np.isfinite(parameters).all():
        raise InputError("Invalid native MHR pose")
    return roma.euler_to_rotmat(
        "xyz", torch.from_numpy(parameters[..., 3:6].copy())
    ).numpy()


def root_tilt_reference(parameters):
    angles = np.asarray(parameters, np.float32)
    if (
        angles.ndim != 2
        or len(angles) < 1
        or angles.shape[1] != 204
        or not np.isfinite(angles).all()
    ):
        raise InputError("Invalid estimated pose parameters")
    matrices = root_rotation(angles)
    up = np.median(matrices[:, :, 1], axis=0).astype(float)
    up /= max(np.linalg.norm(up), 1e-12)
    if up[1] < 0.5:
        raise InputError("Estimated standing axis is not supported")
    axis = np.cross([0, 1, 0], up)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.eye(3) + skew + skew @ skew / (1 + up[1])).astype("float32")


def constrain_root_tilt(parameters, reference):
    import roma
    import torch

    parameters = np.asarray(parameters, np.float32)
    reference = np.asarray(reference, np.float32)
    if (
        parameters.shape != (204,)
        or not np.isfinite(parameters).all()
        or reference.shape != (3, 3)
        or not np.isfinite(reference).all()
        or not np.allclose(reference.T @ reference, np.eye(3), atol=1e-5)
        or np.linalg.det(reference) < 0.99
    ):
        raise InputError("Invalid shared root tilt")
    matrix = root_rotation(parameters)
    local = reference.T @ matrix
    # Closest rotation around the standing axis in Frobenius distance.
    # Using only the projected forward vector biases heading under pitch.
    heading = np.arctan2(local[0, 2] - local[2, 0], local[0, 0] + local[2, 2])
    c, s = np.cos(heading), np.sin(heading)
    yaw = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    result = parameters.copy()
    result[3:6] = roma.rotmat_to_euler("xyz", torch.from_numpy(reference @ yaw)).numpy()
    return result


def smooth_headings(headings, strength, positions=None):
    """Penalize angular acceleration without imposing a known turn rate."""
    h = np.asarray(headings, float)
    t = (
        np.arange(len(h), dtype=float)
        if positions is None
        else np.asarray(positions, float)
    )
    if (
        h.ndim != 1
        or len(h) < 3
        or t.shape != h.shape
        or not np.isfinite(h).all()
        or not np.isfinite(t).all()
        or (np.diff(t) <= 0).any()
        or not np.isfinite(strength)
        or not 0 <= strength <= 100000
    ):
        raise InputError("Invalid heading sequence or smoothness")
    if strength == 0:
        return np.unwrap(h)
    spacing = np.diff(t) / np.median(np.diff(t))
    d = np.zeros((len(h) - 2, len(h)))
    for i, (left, right) in enumerate(zip(spacing[:-1], spacing[1:])):
        d[i, i : i + 3] = (
            np.array([1 / left, -1 / left - 1 / right, 1 / right]) * 2 / (left + right)
        )
    return np.linalg.solve(np.eye(len(h)) + strength * d.T @ d, np.unwrap(h))


def harmonic_bias_headings(headings, positions=None):
    """Estimate a constant turn rate separately from periodic pose-estimator bias."""
    from scipy.optimize import least_squares

    h = np.unwrap(np.asarray(headings, float))
    t = (
        np.arange(len(h), dtype=float)
        if positions is None
        else np.asarray(positions, float)
    )
    # Share the sequence/time checks with the general acceleration smoother.
    smooth_headings(h, 0, t)
    if len(h) < 12 or np.ptp(h) < np.pi:
        raise InputError("Constant-motion model requires a sufficiently observed turn")
    span = np.ptp(t)
    x = (t - t.mean()) / span
    rate, phase = np.polyfit(x, h, 1)
    if abs(rate) < np.pi / 2:
        raise InputError("Estimated turn direction is not identifiable")

    def residual(p):
        angle = p[1] * x
        predicted = (
            p[0]
            + angle
            + p[2] * np.sin(angle)
            + p[3] * np.cos(angle)
            + p[4] * np.sin(2 * angle)
            + p[5] * np.cos(2 * angle)
        )
        return np.r_[(predicted - h) / 0.03, np.asarray(p[2:]) / 0.3]

    low, high = sorted([rate * 0.7, rate * 1.3])
    fit = least_squares(
        residual,
        [phase, rate, 0, 0, 0, 0],
        bounds=(
            [-np.inf, low, -0.5, -0.5, -0.5, -0.5],
            [np.inf, high, 0.5, 0.5, 0.5, 0.5],
        ),
        loss="soft_l1",
        f_scale=1,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise InputError("Constant-motion heading fit did not converge")
    return fit.x[0] + fit.x[1] * x


def constrain_root_sequence(
    parameters, reference, strength, positions=None, model="acceleration"
):
    import roma
    import torch

    if model not in ("acceleration", "harmonic_bias"):
        raise InputError("Invalid heading motion model")

    constrained = np.asarray([constrain_root_tilt(p, reference) for p in parameters])
    local = reference.T @ root_rotation(constrained)
    h = np.arctan2(local[:, 0, 2] - local[:, 2, 0], local[:, 0, 0] + local[:, 2, 2])
    h = (
        harmonic_bias_headings(h, positions)
        if model == "harmonic_bias"
        else smooth_headings(h, strength, positions)
    )
    c, s = np.cos(h), np.sin(h)
    yaw = np.zeros((len(h), 3, 3), np.float32)
    yaw[:, 0, 0] = yaw[:, 2, 2] = c
    yaw[:, 0, 2] = s
    yaw[:, 2, 0] = -s
    yaw[:, 1, 1] = 1
    constrained[:, 3:6] = roma.rotmat_to_euler(
        "xyz", torch.from_numpy(reference @ yaw)
    ).numpy()
    return constrained
