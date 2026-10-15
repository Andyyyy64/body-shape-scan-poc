"""Independent image calf-region width: sum every connected foreground interval."""

import numpy as np

from body_scan.io import InputError


def scan_leg_anatomy(polygon, keypoints):
    polygon, keypoints = np.asarray(polygon, float), np.asarray(keypoints, float)
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or keypoints.shape != (17, 3)
        or not np.isfinite(polygon).all()
        or not np.isfinite(keypoints).all()
        or (keypoints[[13, 14, 15, 16], 2] < 0.5).any()
    ):
        raise InputError("Leg scale needs confident independent knee and ankle pixels")
    knee = keypoints[[13, 14], 1].mean()
    ankle = keypoints[[15, 16], 1].mean()
    if ankle - knee <= 10:
        raise InputError("Calf image span cannot support scale observations")
    rows = knee + np.linspace(0.15, 0.55, 32) * (ankle - knee)
    end = np.roll(polygon, -1, axis=0)
    intervals = []
    for y in rows:
        crossed = ((polygon[:, 1] <= y) & (y < end[:, 1])) | (
            (end[:, 1] <= y) & (y < polygon[:, 1])
        )
        dy = end[crossed, 1] - polygon[crossed, 1]
        x = np.sort(
            polygon[crossed, 0]
            + (y - polygon[crossed, 1]) * (end[crossed, 0] - polygon[crossed, 0]) / dy
        )
        if len(x) < 2 or len(x) % 2:
            raise InputError("Calf scanline lacks closed foreground intervals")
        span = np.stack([x[::2], x[1::2]], axis=1)
        if (np.diff(span, axis=1) <= 0).any():
            raise InputError("Degenerate calf foreground interval")
        intervals.append(span)
    return rows, intervals


def calf_widths(intervals):
    if len(intervals) != 32:
        raise InputError("All32 calf rows required")
    out = np.array([np.sum(span[:, 1] - span[:, 0]) for span in intervals])
    if not np.isfinite(out).all() or (out <= 1).any():
        raise InputError("Invalid calf foreground width")
    return out
