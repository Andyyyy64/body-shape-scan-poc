"""Head ROI from independent shoulder/hip pixels, independent of full-body height."""

import numpy as np

from body_scan.io import InputError


def scan_head_anatomy(polygon, keypoints):
    polygon, keypoints = map(lambda x: np.asarray(x, float), [polygon, keypoints])
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or keypoints.shape != (17, 3)
        or not np.isfinite(polygon).all()
        or not np.isfinite(keypoints).all()
        or (keypoints[[5, 6, 11, 12], 2] < 0.5).any()
    ):
        raise InputError(
            "Head ROI requires independent confident shoulder/hip image positions"
        )
    top = polygon[:, 1].min()
    shoulder = float(keypoints[[5, 6], 1].mean())
    hip = float(keypoints[[11, 12], 1].mean())
    bottom = shoulder - 0.15 * (hip - shoulder)
    if shoulder - top <= 10 or hip - shoulder <= 5 or bottom - top <= 10:
        raise InputError("Independent image anatomy cannot support head ROI")
    rows = np.linspace(top + 0.02 * (shoulder - top), bottom, 64)
    end = np.roll(polygon, -1, axis=0)
    position = []
    counts = []
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
            raise InputError("Anatomy head row has no closed foreground interval")
        intervals = np.stack([x[::2], x[1::2]], axis=1)
        index = int(np.argmax(intervals[:, 1] - intervals[:, 0]))
        if intervals[index, 1] - intervals[index, 0] <= 1:
            raise InputError("Anatomy head row is too narrow")
        position.append(intervals[index])
        counts.append(len(intervals))
    return rows, np.array(position), np.array(counts)
