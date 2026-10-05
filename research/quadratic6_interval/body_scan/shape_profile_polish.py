"""Explicit research-only final shape polishing after original camera/RGB stages.

No camera proposal, pixel-row observer, variance or phase prior is changed.
The default is delegated to the original fitter without extra diagnostics.
"""

import hashlib
import json

import numpy as np

from . import calibrated_width
from .framewise_width import fit_curve as original_fit
from .io import InputError
from .phase_registration import periodic_reference

PROFILES = ('quadratic', 'cubic', 'partial-cubic-x', 'partial-cubic-z')
LIKELIHOODS = ('point', 'pixel-interval')


def validate_options(shape_profile, width_likelihood, *, scale_camera=True):
    if shape_profile is not None and shape_profile not in PROFILES:
        raise InputError('Unknown fixed research shape profile')
    if width_likelihood not in LIKELIHOODS:
        raise InputError('Unknown research width likelihood')
    enabled = shape_profile is not None or width_likelihood != 'point'
    if enabled and not scale_camera:
        raise InputError('Research shape polishing requires the scale-camera measurement path')
    return enabled


def configuration(shape_profile, width_likelihood):
    validate_options(shape_profile, width_likelihood)
    value = {'shape_profile': shape_profile, 'width_likelihood': width_likelihood,
             'original_camera_proposal_and_actual_cv_variance_retained': True,
             'original10_physical_shape_fields_perimeter_and_rows_retained': True}
    identity = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return value, identity


def pixel_interval(prepared, calibration, heading, normalizer, observed):
    normalizer = np.asarray(normalizer, float)
    if normalizer.shape != (48, 1) or np.asarray(observed).shape != (48, 41) or not np.isfinite(normalizer).all() or (normalizer <= 0).any():
        raise InputError('Research pixel interval requires all48 normalizers and41 widths')
    reference = prepared['reference']
    depth = calibrated_width.estimated_depth(reference, reference['normalization_camera'])
    reference_normalizer = reference['shared_focal']/depth*calibration['reference_height_model_m']/256
    if reference_normalizer.shape != (48,) or not np.isfinite(reference_normalizer).all() or (reference_normalizer <= 0).any():
        raise InputError('Research reference pixel normalizers are invalid')
    reference_interval = periodic_reference(calibration['heading'], 1/reference_normalizer, heading, interpolation='linear')
    return np.broadcast_to(1/normalizer+reference_interval[:, None], (48, 41)).copy()


def polish(prepared, calibration, heading, observed, sigma, initial, phase_covariance,
           normalizer, *, shape_profile=None, width_likelihood='point'):
    enabled = validate_options(shape_profile, width_likelihood)
    if not enabled:
        return original_fit(calibration, heading, observed, sigma, initial, phase_covariance=phase_covariance)
    interval = pixel_interval(prepared, calibration, heading, normalizer, observed) if width_likelihood == 'pixel-interval' else None
    arguments = {'phase_covariance': phase_covariance, 'interval_half_width': interval}
    if shape_profile is None:
        from .pixel_interval_fit import fit_curve
    elif shape_profile == 'quadratic':
        from .quadratic_shape_profile import fit_curve
        arguments['restricted'] = True
    elif shape_profile == 'cubic':
        from .cubic_shape_profile import fit_curve
        arguments['restricted'] = True
    else:
        from .partial_cubic_shape_profile import fit_curve
        arguments.update(restricted=True, cubic_axis=shape_profile[-1])
    result = fit_curve(calibration, heading, observed, sigma, initial, **arguments)
    options, identity = configuration(shape_profile, width_likelihood)
    return {**result, 'research_shape_polish': {**options, 'configuration_sha256': identity,
                                              'physical_accuracy_validated': False,
                                              'numeric_and_qc_adoption_gates_verified': False}}
