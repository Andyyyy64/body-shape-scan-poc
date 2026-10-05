"""Research registration using image-estimated native standing rotations."""

import numpy as np

from .io import InputError


def native_headings(rotations, tilt):
    rotations, tilt = np.asarray(rotations, float), np.asarray(tilt, float)
    if (
        rotations.ndim != 3
        or rotations.shape[1:] != (3, 3)
        or tilt.shape != (3, 3)
        or not np.isfinite(rotations).all()
        or not np.isfinite(tilt).all()
        or not np.allclose(
            rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4
        )
        or not np.allclose(tilt.T @ tilt, np.eye(3), atol=1e-4)
        or (np.linalg.det(rotations) < 0.99).any()
        or np.linalg.det(tilt) < 0.99
    ):
        raise InputError("Invalid native standing rotations")
    local = tilt.T @ rotations
    angle = np.arctan2(local[:, 0, 2] - local[:, 2, 0], local[:, 0, 0] + local[:, 2, 2])
    c, s = np.cos(angle), np.sin(angle)
    expected = np.zeros_like(local)
    expected[:, 0, 0] = expected[:, 2, 2] = c
    expected[:, 0, 2] = s
    expected[:, 2, 0] = -s
    expected[:, 1, 1] = 1
    if abs(expected - local).max() > 1e-3:
        raise InputError("Standing-axis phase is not identified by these rotations")
    return angle


def periodic_reference(
    reference_angles, reference_values, query_angles, *, interpolation="linear"
):
    """Interpolate a complete observed turn by pose, never by frame identity."""
    phase = np.mod(np.asarray(reference_angles, float), 2 * np.pi)
    values = np.asarray(reference_values, float)
    query = np.asarray(query_angles, float)
    if (
        phase.ndim != 1
        or query.ndim != 1
        or values.shape[0] != len(phase)
        or values.ndim < 1
        or min(len(phase), len(query)) < 1
        or not all(np.isfinite(x).all() for x in (phase, values, query))
    ):
        raise InputError("Invalid reference phase observations")
    order = np.argsort(phase)
    phase = phase[order]
    values = values[order]
    gaps = np.diff(np.r_[phase, phase[0] + 2 * np.pi])
    if len(phase) < 24 or gaps.min() < 1e-5 or gaps.max() > np.pi / 4:
        raise InputError("Reference turn lacks independent complete angular coverage")
    flattened = values.reshape(len(values), -1)
    if interpolation == "linear":
        result = np.stack(
            [
                np.interp(query, phase, column, period=2 * np.pi)
                for column in flattened.T
            ],
            axis=-1,
        )
    elif interpolation == "cubic":
        from scipy.interpolate import CubicSpline

        curve = CubicSpline(
            np.r_[phase, phase[0] + 2 * np.pi],
            np.vstack([flattened, flattened[0]]),
            axis=0,
            bc_type="periodic",
        )
        result = curve(np.mod(query - phase[0], 2 * np.pi) + phase[0])
    else:
        raise InputError("Invalid periodic reference interpolation")
    return result.reshape(len(query), *values.shape[1:])
