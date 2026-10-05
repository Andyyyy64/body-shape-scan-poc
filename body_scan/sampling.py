"""Choose observed frames by angular coverage; never synthesize missing views."""

from __future__ import annotations

import numpy as np

from .io import InputError


def angular_indices(yaws, detail, count):
    yaws, detail = np.asarray(yaws, float), np.asarray(detail, float)
    if (
        yaws.ndim != 1
        or detail.shape != yaws.shape
        or len(yaws) < 1
        or not np.isfinite(yaws).all()
        or not np.isfinite(detail).all()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise InputError("Invalid angular sampling observations")
    if count >= len(yaws):
        return np.arange(len(yaws))
    selected = [int(detail.argmax())]
    # Fill the largest gaps. Detail only breaks near-equal angular choices, so
    # a sharp front view cannot displace the entire back/side coverage.
    while len(selected) < count:
        distance = np.abs(np.angle(np.exp(1j * (yaws[:, None] - yaws[selected])))).min(
            1
        )
        distance[selected] = -1
        candidates = np.where(distance >= distance.max() - 0.01)[0]
        selected.append(int(candidates[detail[candidates].argmax()]))
    return np.array(sorted(selected))
