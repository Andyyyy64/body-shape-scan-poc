"""Fixed quadratic coefficient profiles within the original10 shape fields.

No new mesh, rows, measurement definition, image filtering or gain correction.
The six bounded Bernstein controls map to the original10 bounded coefficients
by positive convex combinations. This is an explicit shape-model restriction,
not evidence that all legitimate body changes are quadratic.
"""

import numpy as np

from .io import InputError
from .pixel_interval_fit import fit_curve as original_fit_curve


def coefficient_map():
    t = np.linspace(0, 1, 5)
    axial = np.stack([(1-t)**2, 2*t*(1-t), t**2], axis=-1)
    matrix = np.zeros((10, 6))
    matrix[:5, :3], matrix[5:, 3:] = axial, axial
    return matrix


def fit_curve(calibration, heading, observed, sigma, initial, *, restricted=False,
              softening=1., max_evaluations=200, phase_covariance=None, interval_half_width=None):
    arguments = {'softening':softening,'max_evaluations':max_evaluations,
                 'phase_covariance':phase_covariance,'interval_half_width':interval_half_width}
    if not restricted:
        return original_fit_curve(calibration, heading, observed, sigma, initial, **arguments)
    matrix = coefficient_map()
    jacobian = np.asarray(calibration['jacobian'], float)
    beta = np.asarray(initial['delta_coefficients'], float)
    if jacobian.ndim != 3 or jacobian.shape[-1] != 10 or beta.shape != (10,) or not np.isfinite(jacobian).all() or not np.isfinite(beta).all():
        raise InputError('Quadratic subspace requires complete original10 shape fields')
    projected = np.linalg.lstsq(matrix, beta, rcond=None)[0]
    latent_initial = np.clip(projected, -.1, .1)
    reduced = {**calibration,'jacobian':jacobian @ matrix}
    reduced_initial = {**initial,'delta_coefficients':latent_initial}
    fit = original_fit_curve(reduced,heading,observed,sigma,reduced_initial,**arguments)
    latent = np.asarray(fit['delta_coefficients'],float)
    mapped = matrix @ latent
    if latent.shape != (6,) or not np.isfinite(latent).all() or abs(latent).max() > .1 or not np.isfinite(mapped).all() or abs(mapped).max() > .1:
        raise InputError('Restricted fit violates the original coefficient domain')
    return {**fit,'delta_coefficients':mapped,'latent_delta_coefficients':latent.copy(),
            'shape_subspace':'quadratic_bernstein_coefficients_per_axis',
            'original_shape_coefficient_count':10,'latent_shape_coefficient_count':6,
            'projected_initial_latent_coefficients':projected,
            'initial_latent_coefficients_after_domain_clip':latent_initial,
            'initial_projection_residual_l2':float(np.linalg.norm(matrix @ latent_initial-beta)),
            'original_geometry_rows_perimeter_and_all_observations_retained':True,
            'shape_profile_restriction_not_general_accuracy':True}
