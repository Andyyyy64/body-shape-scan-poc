"""Marginalize fixed pixel-scale observation uncertainty under the old loss.

Uniform offsets are integrated with symmetric Gauss-Legendre quadrature; the
reference/query pixel scales set the interval, not known shape changes. The
old weighted pseudo-Huber density is retained inside the integration. This
uncertainty model is a research hypothesis, not calibrated image confidence.
"""

from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from .io import InputError


@lru_cache(maxsize=3)
def quadrature_rule(nodes):
    return np.polynomial.legendre.leggauss(nodes)


def interval_residual(residual, weight, half_width, *, softening=1., nodes=32):
    r, w, h = np.broadcast_arrays(*[np.asarray(x, float) for x in (residual, weight, half_width)])
    if not all(np.isfinite(x).all() for x in (r, w, h)) or (w <= 0).any() or (h < 0).any() or softening <= 0 or not np.isfinite(softening) or nodes not in (32, 64, 128):
        raise InputError('Interval likelihood needs finite positive weights and nonnegative pixel intervals')
    abscissa, quadrature = quadrature_rule(nodes)
    offsets = h[..., None] * abscissa
    log_quadrature = np.log(quadrature / 2)

    def kernel(value):
        scaled = value / softening
        root = np.sqrt(1 + scaled**2)
        return w[..., None] * softening**2 * scaled**2 / (root + 1)

    centered = -offsets
    shifted = r[..., None] - offsets
    base_log = log_quadrature - kernel(centered)
    log_probability = log_quadrature - kernel(shifted)
    base_partition = logsumexp(base_log, axis=-1)
    partition = logsumexp(log_probability, axis=-1)
    loss = np.array(2 * (base_partition - partition))
    # Pair the symmetric quadrature nodes before adding probabilities near zero.
    # Rationalized root differences avoid subtraction of two nearly equal logs.
    near = abs(r) < .01
    if near.any():
        u = offsets[near][..., nodes // 2:]
        rn, wn = r[near][..., None], w[near][..., None]
        root0 = np.sqrt(1 + (u / softening)**2)
        root_plus = np.sqrt(1 + ((rn - u) / softening)**2)
        root_minus = np.sqrt(1 + ((rn + u) / softening)**2)
        plus = wn * (2 * rn * u - rn**2) / (root0 + root_plus)
        minus = wn * (-2 * rn * u - rn**2) / (root0 + root_minus)
        average, difference = (plus + minus) / 2, (plus - minus) / 2
        probability_delta = np.expm1(average) + np.exp(average) * 2 * np.sinh(difference / 2)**2
        base_mass = 2 * np.exp(base_log[near][..., nodes // 2:] - base_partition[near][..., None])
        loss[near] = -2 * np.log1p(np.sum(base_mass * probability_delta, axis=-1))
    if (loss < -1e-10).any():
        raise InputError('Integrated symmetric density lost its central maximum')
    posterior = np.exp(log_probability - partition[..., None])
    score = shifted / np.sqrt(1 + (shifted / softening)**2)
    loss_derivative = 2 * w * np.sum(posterior * score, axis=-1)
    root_loss = np.sqrt(np.maximum(loss, 0))
    transformed = np.sign(r) * root_loss
    derivative = np.sign(r) * loss_derivative / (2 * np.maximum(root_loss, 1e-30))
    small = abs(r) < 1e-5
    base_posterior = np.exp(base_log - base_partition[..., None])
    base_score = centered / np.sqrt(1 + (centered / softening)**2)
    half_hessian = w * np.sum(base_posterior * (1 + (centered / softening)**2)**(-1.5), axis=-1) - w**2 * np.sum(base_posterior * base_score**2, axis=-1)
    if (half_hessian <= 0).any() or not np.isfinite(half_hessian).all():
        raise InputError('Pixel interval likelihood is numerically uninformative')
    derivative = np.where(small, np.sqrt(half_hessian), derivative)
    transformed = np.where(small, r * np.sqrt(half_hessian), transformed)
    if not np.isfinite(transformed).all() or not np.isfinite(derivative).all():
        raise InputError('Invalid integrated pixel likelihood residual or derivative')
    return transformed, derivative
