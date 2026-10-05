"""Local quadratic RGB plateaux, with the original pixel-row geometry operator.

Only the foreground/background color extrapolation changes. The peak seed,
sampling pixels, radius, contrast and residual thresholds are fixed at the
source values. No image colors or edge locations come from renderer metadata.
"""

import numpy as np

from .contour_width import WidthObservationError
from .io import InputError
from .subpixel_edges import (
    central_positions,
    peak_refine_positions,
    refine_pixel_row_positions,
)


def quadratic_refine_positions(bgr, rows, original):
    result = peak_refine_positions(bgr, rows, original)
    image, rows, original = np.asarray(bgr, float), np.asarray(rows, float), np.asarray(original, float)
    y0 = rows.astype(int)
    fraction = rows-y0
    profiles = image[y0]*(1-fraction[:, None, None])+image[y0+1]*fraction[:, None, None]
    shape = original.shape
    plateau_valid = np.zeros(shape, bool)
    quadratic_coefficients = np.zeros((*shape, 2, 3, 3))
    affine_rms = np.zeros((*shape, 2))
    quadratic_rms = np.zeros((*shape, 2))
    for row in range(len(rows)):
        for side in range(2):
            if not result['refined'][row, side]:
                continue
            center = int(np.rint(result['positions'][row, side]))
            if center < 8 or center > image.shape[1]-9:
                result['refined'][row, side] = False
                result['positions'][row, side] = original[row, side]
                result['fallback_reason'][row, side] = 4
                continue
            outside = center+np.array([-7, -6, -5, -4] if side == 0 else [4, 5, 6, 7])
            inside = center+np.array([4, 5, 6, 7] if side == 0 else [-7, -6, -5, -4])
            pixels = np.arange(center-3, center+4)
            colors, bad = [], False
            for i, plateau in enumerate((outside, inside)):
                offsets = plateau-center
                design = np.c_[np.ones(4), offsets, offsets**2]
                coefficient = np.linalg.lstsq(design, profiles[row, plateau], rcond=None)[0]
                affine = np.linalg.lstsq(design[:, :2], profiles[row, plateau], rcond=None)[0]
                rms = float(np.sqrt(np.mean((profiles[row, plateau]-design @ coefficient)**2)))
                quadratic_coefficients[row, side, i] = coefficient
                affine_rms[row, side, i] = np.sqrt(np.mean((profiles[row, plateau]-design[:, :2] @ affine)**2))
                quadratic_rms[row, side, i] = rms
                bad |= rms > 4.
                prediction = np.c_[np.ones(7), pixels-center, (pixels-center)**2] @ coefficient
                colors.append(prediction)
            plateau_valid[row, side] = True
            background, foreground = colors
            difference = foreground-background
            squared = np.sum(difference**2, axis=1)
            bad |= squared.min() < 20.**2
            alpha = np.sum((profiles[row, pixels]-background)*difference, axis=1)/np.maximum(squared, 1e-12)
            bad |= not np.isfinite(alpha).all() or alpha.min() < -.2 or alpha.max() > 1.2
            position = pixels[-1]+.5-alpha.sum() if side == 0 else pixels[0]-.5+alpha.sum()
            bad |= abs(position-original[row, side]) > 3.
            if bad:
                result['positions'][row, side] = original[row, side]
                result['refined'][row, side] = False
                result['fallback_reason'][row, side] = 4
            else:
                result['positions'][row, side] = position
    if (result['positions'][:, 1] <= result['positions'][:, 0]).any():
        raise WidthObservationError('Quadratic color coverage inverts the central width')
    return {**result, 'color_plateau_valid': plateau_valid,
            'color_quadratic_coefficients': quadratic_coefficients,
            'color_affine_plateau_rms': affine_rms,
            'color_quadratic_plateau_rms': quadratic_rms}


class PolynomialColorObserver:
    def __init__(self, mode):
        if mode not in ('original', 'quadratic_color'):
            raise InputError('Unknown fixed RGB plateau model')
        self.mode = mode

    def refine(self, bgr, polygon, rows, centers):
        if self.mode == 'original':
            return refine_pixel_row_positions(bgr, polygon, rows, centers)
        rows, centers = np.asarray(rows, float), np.asarray(centers, float)
        lower, upper = np.floor(rows), np.ceil(rows)
        low, high = central_positions(polygon, lower, centers), central_positions(polygon, upper, centers)
        count = len(rows)
        combined = quadratic_refine_positions(bgr, np.r_[lower, upper], np.r_[low, high])

        def paired(value):
            return np.stack([value[:count], value[count:]], axis=1)

        positions = paired(combined['positions'])
        weights = np.stack([1-(rows-lower), rows-lower], axis=1)
        original = np.stack([low, high], axis=1)
        return {'positions': np.sum(positions*weights[..., None], axis=1),
                'original_positions': np.sum(original*weights[..., None], axis=1),
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
                'color_plateau_valid': paired(combined['color_plateau_valid']),
                'color_quadratic_coefficients': paired(combined['color_quadratic_coefficients']),
                'color_affine_plateau_rms': paired(combined['color_affine_plateau_rms']),
                'color_quadratic_plateau_rms': paired(combined['color_quadratic_plateau_rms'])}
