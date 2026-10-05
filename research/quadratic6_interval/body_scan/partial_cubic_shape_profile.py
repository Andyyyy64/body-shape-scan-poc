"""Axis ablation: one cubic and one quadratic profile in original10 fields.

Both fixed maps contain the entire bounded quadratic6 domain. Bounds, physical
fields, observations and perimeter definition remain those of the source fit.
This is a shape-model restriction, not general human-accuracy evidence.
"""

import numpy as np

from .io import InputError
from .pixel_interval_fit import fit_curve as original_fit_curve


def coefficient_map(cubic_axis):
    if cubic_axis not in ('x', 'z'):
        raise InputError('cubic_axis must be x or z')
    t = np.linspace(0, 1, 5)
    quadratic = np.stack([(1-t)**2, 2*t*(1-t), t**2], axis=-1)
    cubic = np.stack([(1-t)**3, 3*t*(1-t)**2, 3*t**2*(1-t), t**3], axis=-1)
    matrix = np.zeros((10, 7))
    if cubic_axis == 'x':
        matrix[:5, :4], matrix[5:, 4:] = cubic, quadratic
    else:
        matrix[:5, :3], matrix[5:, 3:] = quadratic, cubic
    return matrix


def fit_curve(calibration, heading, observed, sigma, initial, *, cubic_axis,
              restricted=False, softening=1., max_evaluations=200,
              phase_covariance=None, interval_half_width=None):
    matrix = coefficient_map(cubic_axis)
    arguments = {'softening': softening, 'max_evaluations': max_evaluations,
                 'phase_covariance': phase_covariance, 'interval_half_width': interval_half_width}
    if not restricted:
        return original_fit_curve(calibration, heading, observed, sigma, initial, **arguments)
    jacobian = np.asarray(calibration['jacobian'], float)
    beta = np.asarray(initial['delta_coefficients'], float)
    if jacobian.ndim != 3 or jacobian.shape[-1] != 10 or beta.shape != (10,) or not np.isfinite(jacobian).all() or not np.isfinite(beta).all():
        raise InputError('Partial cubic subspace requires complete original10 shape fields')
    projected = np.linalg.lstsq(matrix, beta, rcond=None)[0]
    latent_initial = np.clip(projected, -.1, .1)
    reduced = {**calibration, 'jacobian': jacobian @ matrix}
    reduced_initial = {**initial, 'delta_coefficients': latent_initial}
    fit = original_fit_curve(reduced, heading, observed, sigma, reduced_initial, **arguments)
    latent = np.asarray(fit['delta_coefficients'], float)
    mapped = matrix @ latent
    if latent.shape != (7,) or not np.isfinite(latent).all() or abs(latent).max() > .1 or not np.isfinite(mapped).all() or abs(mapped).max() > .1:
        raise InputError('Restricted fit violates the original coefficient domain')
    return {**fit, 'delta_coefficients': mapped, 'latent_delta_coefficients': latent.copy(),
            'shape_subspace': 'partial_cubic_bernstein_coefficients_per_axis',
            'cubic_axis': cubic_axis,
            'original_shape_coefficient_count': 10, 'latent_shape_coefficient_count': 7,
            'projected_initial_latent_coefficients': projected,
            'initial_latent_coefficients_after_domain_clip': latent_initial,
            'initial_projection_residual_l2': float(np.linalg.norm(matrix @ latent_initial-beta)),
            'original_geometry_rows_perimeter_and_all_observations_retained': True,
            'shape_profile_restriction_not_general_accuracy': True}
