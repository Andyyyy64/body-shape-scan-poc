"""Per-frame subject selection and capture-protocol checks (pure NumPy).

Protocol ``relaxed_arms_down_full_torso.v1``: the operator turns freely, but
the abdomen is relaxed, both arms hang below the elbows and the head, both
shoulders and both hips stay inside the image. Frames violating this are
excluded from reconstruction; a capture with too few remaining frames fails
instead of producing a mesh.
"""

from __future__ import annotations

import numpy as np

CAPTURE_PROTOCOL_ID = "relaxed_arms_down_full_torso.v1"

# Detections other than the largest person must be small relative to it
# (static objects such as bags get low-confidence "person" masks). Two
# comparably sized people make the frame ambiguous and it is rejected.
SECONDARY_AREA_RATIO = 0.25

# At least this fraction of the sampled frames must survive detection and
# protocol checks; otherwise the capture is refused rather than reconstructed
# from an angularly biased subset.
MIN_USABLE_FRACTION = 0.75

# Image-border margin (px) inside which a core landmark counts as truncated.
EDGE_MARGIN_PX = 2

# MHR-70 keypoint indices used by the checks.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_HIP, RIGHT_HIP = 9, 10
RIGHT_WRIST, LEFT_WRIST = 41, 62
KEYPOINT_COUNT = 70
CORE_LANDMARKS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

REASON_NO_PERSON = "no_person"
REASON_MULTIPLE_PEOPLE = "multiple_people"
REASON_CORE_OUT_OF_FRAME = "core_out_of_frame"
REASON_ARM_NOT_LOWERED = "arm_not_lowered"


class FrameRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def select_primary_person(mask_areas) -> int:
    """Return the index of the primary person mask or raise ``FrameRejected``."""
    areas = np.asarray(mask_areas, dtype=np.float64).reshape(-1)
    if areas.size == 0 or not np.isfinite(areas).all() or (areas < 0).any():
        raise FrameRejected(REASON_NO_PERSON)
    primary = int(np.argmax(areas))
    if areas[primary] <= 0:
        raise FrameRejected(REASON_NO_PERSON)
    others = np.delete(areas, primary)
    if others.size and (others > SECONDARY_AREA_RATIO * areas[primary]).any():
        raise FrameRejected(REASON_MULTIPLE_PEOPLE)
    return primary


def frame_protocol_violations(keypoints_2d, width: int, height: int) -> list[str]:
    """Return protocol violations for one frame from MHR-70 image keypoints."""
    points = np.asarray(keypoints_2d, dtype=np.float64)
    if (
        points.shape != (KEYPOINT_COUNT, 2)
        or not np.isfinite(points).all()
        or width < 2 * EDGE_MARGIN_PX + 1
        or height < 2 * EDGE_MARGIN_PX + 1
    ):
        raise ValueError("Expected finite MHR-70 image keypoints and an image size")
    reasons = []
    core = points[list(CORE_LANDMARKS)]
    if (
        (core[:, 0] < EDGE_MARGIN_PX).any()
        or (core[:, 0] > width - 1 - EDGE_MARGIN_PX).any()
        or (core[:, 1] < EDGE_MARGIN_PX).any()
        or (core[:, 1] > height - 1 - EDGE_MARGIN_PX).any()
    ):
        reasons.append(REASON_CORE_OUT_OF_FRAME)
    # Image y grows downward: a wrist above its elbow means a raised, crossed
    # or flexed arm, which changes the torso silhouette.
    if (
        points[LEFT_WRIST, 1] < points[LEFT_ELBOW, 1]
        or points[RIGHT_WRIST, 1] < points[RIGHT_ELBOW, 1]
    ):
        reasons.append(REASON_ARM_NOT_LOWERED)
    return reasons


def minimum_usable_frames(sampled: int) -> int:
    if sampled < 1:
        raise ValueError("At least one sampled frame is required")
    return int(np.ceil(MIN_USABLE_FRACTION * sampled))


def summarize_outcomes(outcomes: list[dict]) -> dict:
    """Count frame outcomes by reason for the reconstruction record."""
    reasons: dict[str, int] = {}
    for row in outcomes:
        if row["state"] == "estimated":
            continue
        for reason in row.get("reasons", [row["reason"]]):
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "capture_protocol_id": CAPTURE_PROTOCOL_ID,
        "excluded_frames": sum(row["state"] != "estimated" for row in outcomes),
        "exclusion_reasons": dict(sorted(reasons.items())),
    }
