"""Bounded RGB-transition localization at frozen central silhouette scanlines."""

import numpy as np

from .contour_width import WidthObservationError
from .io import InputError


def central_positions(polygon, rows, centers):
    polygon, rows, centers = (
        np.asarray(polygon, float),
        np.asarray(rows, float),
        np.asarray(centers, float),
    )
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or rows.ndim != 1
        or centers.shape != rows.shape
        or not all(np.isfinite(x).all() for x in (polygon, rows, centers))
    ):
        raise InputError("Invalid RGB contour scanlines")
    end = np.roll(polygon, -1, axis=0)
    y = rows[:, None]
    crossing = ((polygon[None, :, 1] <= y) & (y < end[None, :, 1])) | (
        (end[None, :, 1] <= y) & (y < polygon[None, :, 1])
    )
    dy = end[:, 1] - polygon[:, 1]
    fraction = (y - polygon[None, :, 1]) / np.where(abs(dy) > 1e-12, dy, 1)
    x = polygon[None, :, 0] + fraction * (end - polygon)[None, :, 0]
    left = np.where(crossing & (x <= centers[:, None]), x, -np.inf).max(axis=1)
    right = np.where(crossing & (x > centers[:, None]), x, np.inf).min(axis=1)
    inside = np.sum(crossing & (x <= centers[:, None]), axis=1) % 2 == 1
    if not inside.all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise WidthObservationError(
            "Central RGB scanline lacks a connected mask interval"
        )
    return np.stack([left, right], axis=1)


def pixel_row_positions(polygon, rows, centers):
    """Interpolate connected intervals measured at actual integer pixel rows."""
    rows = np.asarray(rows, float)
    lower, upper = np.floor(rows), np.ceil(rows)
    fraction = (rows - lower)[:, None]
    low = central_positions(polygon, lower, centers)
    high = central_positions(polygon, upper, centers)
    return low * (1 - fraction) + high * fraction


def refine_pixel_row_positions(bgr, polygon, rows, centers):
    """Refine both integer-row edges before interpolation, retaining evidence."""
    rows = np.asarray(rows, float)
    lower, upper = np.floor(rows), np.ceil(rows)
    low = central_positions(polygon, lower, centers)
    high = central_positions(polygon, upper, centers)
    count = len(rows)
    combined = refine_positions(bgr, np.r_[lower, upper], np.r_[low, high])
    fraction = rows - lower

    def paired(value):
        return np.stack([value[:count], value[count:]], axis=1)

    positions = paired(combined['positions'])
    weights = np.stack([1 - fraction, fraction], axis=1)
    original = np.stack([low, high], axis=1)
    return {
        'positions': np.sum(positions * weights[..., None], axis=1),
        'original_positions': np.sum(original * weights[..., None], axis=1),
        'refined': np.all(paired(combined['refined']), axis=1),
        'gradient_strength': np.min(paired(combined['gradient_strength']), axis=1),
        'fallback_reason': np.max(paired(combined['fallback_reason']), axis=1),
        'pixel_row_indices': np.stack([lower, upper], axis=1),
        'pixel_row_weights': weights,
        'pixel_row_original_positions': original,
        'pixel_row_rgb_positions': positions,
        'pixel_row_refined_edges': paired(combined['refined']),
        'pixel_row_fallback_reason': paired(combined['fallback_reason']),
        'pixel_row_gradient_strength': paired(combined['gradient_strength']),
    }


def peak_refine_positions(
    bgr, rows, original, *, radius=3.0, sigma=0.65, minimum_gradient=8.0
):
    """Refine strong unambiguous edges; retain mask coordinates on declared fallback.

    No row is discarded. A false refinement flag explicitly records that the
    RGB transition was insufficient and the original mask observation remains.
    """
    import cv2

    image = np.asarray(bgr)
    rows, original = np.asarray(rows, float), np.asarray(original, float)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or rows.ndim != 1
        or original.shape != (len(rows), 2)
        or not np.isfinite(image).all()
        or not np.isfinite(rows).all()
        or not np.isfinite(original).all()
        or min(image.shape[:2]) < 10
    ):
        raise InputError("Invalid RGB edge observations")
    if not (0 < radius <= 4 and 0 < sigma <= 2 and 0 < minimum_gradient < 255):
        raise InputError("Invalid RGB edge localization rule")
    if (
        (rows < 1).any()
        or (rows >= image.shape[0] - 2).any()
        or (original < 4).any()
        or (original >= image.shape[1] - 5).any()
    ):
        raise InputError("RGB edge scanline exceeds image coverage")
    smooth = cv2.GaussianBlur(
        image.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
    )
    derivative = cv2.Sobel(
        smooth,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        scale=1 / 8,
        borderType=cv2.BORDER_REFLECT_101,
    )
    y0 = rows.astype(int)
    fraction = rows - y0
    g = (
        derivative[y0] * (1 - fraction[:, None, None])
        + derivative[y0 + 1] * fraction[:, None, None]
    )
    magnitude = np.sqrt(np.sum(g * g, axis=2))
    refined = original.copy()
    valid = np.zeros(original.shape, bool)
    strength = np.zeros(original.shape, float)
    reason = np.zeros(original.shape, np.uint8)
    # reason: 0 refined, 1 weak, 2 peak beyond bounds, 3 competing transitions.
    for row_index in range(len(rows)):
        for side in range(2):
            seed = original[row_index, side]
            low = max(1, int(np.ceil(seed - radius)))
            high = min(image.shape[1] - 2, int(np.floor(seed + radius)))
            profile = magnitude[row_index]
            index = low + int(np.argmax(profile[low : high + 1]))
            strength[row_index, side] = profile[index]
            if profile[index] < minimum_gradient:
                reason[row_index, side] = 1
                continue
            if index == low or index == high:
                reason[row_index, side] = 2
                continue
            positions = np.arange(low, high + 1)
            competitors = positions[abs(positions - index) > 2]
            if len(competitors) and profile[competitors].max() >= 0.85 * profile[index]:
                reason[row_index, side] = 3
                continue
            values = np.log(np.maximum(profile[index - 1 : index + 2], 1e-5))
            curvature = values[0] - 2 * values[1] + values[2]
            offset = (
                0.5 * (values[0] - values[2]) / curvature if curvature < -1e-10 else 0.0
            )
            position = index + np.clip(offset, -0.5, 0.5)
            if abs(position - seed) > radius:
                reason[row_index, side] = 2
                continue
            refined[row_index, side] = position
            valid[row_index, side] = True
    if (refined[:, 1] <= refined[:, 0]).any():
        raise WidthObservationError("RGB transitions invert the central width")
    return {
        "positions": refined,
        "refined": valid,
        "gradient_strength": strength,
        "fallback_reason": reason,
        "original_positions": original.copy(),
    }


def refine_positions(
    bgr, rows, original, *, radius=3.0, sigma=0.65, minimum_gradient=8.0
):
    """Estimate fractional pixel coverage from local affine foreground/background colors.

    Strong bounded RGB peaks seed the edge. Separate color plateaux estimate
    slow shading without known colors. Pixel-coverage integration preserves
    symmetric-blur area; an invalid local color model explicitly falls back.
    """
    result = peak_refine_positions(
        bgr,
        rows,
        original,
        radius=radius,
        sigma=sigma,
        minimum_gradient=minimum_gradient,
    )
    image = np.asarray(bgr, float)
    rows = np.asarray(rows, float)
    y0 = rows.astype(int)
    fraction = rows - y0
    profiles = (
        image[y0] * (1 - fraction[:, None, None])
        + image[y0 + 1] * fraction[:, None, None]
    )
    for row in range(len(rows)):
        for side in range(2):
            if not result["refined"][row, side]:
                continue
            center = int(np.rint(result["positions"][row, side]))
            if center < 8 or center > image.shape[1] - 9:
                result["refined"][row, side] = False
                result["positions"][row, side] = original[row, side]
                result["fallback_reason"][row, side] = 4
                continue
            sample = profiles[row]
            outside = center + np.array([-7, -6, -5, -4] if side == 0 else [4, 5, 6, 7])
            inside = center + np.array([4, 5, 6, 7] if side == 0 else [-7, -6, -5, -4])
            colors = []
            bad = False
            integral_pixels = np.arange(center - 3, center + 4)
            for plateau in [outside, inside]:
                design = np.c_[np.ones(4), plateau - center]
                coefficient = np.linalg.lstsq(design, sample[plateau], rcond=None)[0]
                residual = sample[plateau] - design @ coefficient
                bad |= np.sqrt(np.mean(residual**2)) > 4.0
                colors.append(np.c_[np.ones(7), integral_pixels - center] @ coefficient)
            background, foreground = colors
            difference = foreground - background
            squared = np.sum(difference**2, axis=1)
            bad |= squared.min() < 20.0**2
            alpha = np.sum(
                (sample[integral_pixels] - background) * difference, axis=1
            ) / np.maximum(squared, 1e-12)
            bad |= (
                not np.isfinite(alpha).all() or alpha.min() < -0.2 or alpha.max() > 1.2
            )
            position = (
                integral_pixels[-1] + 0.5 - alpha.sum()
                if side == 0
                else integral_pixels[0] - 0.5 + alpha.sum()
            )
            bad |= abs(position - original[row, side]) > radius
            if bad:
                result["positions"][row, side] = original[row, side]
                result["refined"][row, side] = False
                result["fallback_reason"][row, side] = 4
            else:
                result["positions"][row, side] = position
    if (result["positions"][:, 1] <= result["positions"][:, 0]).any():
        raise WidthObservationError(
            "Pixel-coverage transitions invert the central width"
        )
    return result
