"""Private pixel interval likelihood fit, with exact original delegation."""

import numpy as np
from scipy.interpolate import CubicSpline

from .framewise_width import fit_curve as original_fit_curve
from .io import InputError
from .phase_registration import periodic_reference
from .pixel_interval_likelihood import interval_residual


def fit_curve(
    calibration,
    heading,
    observed,
    sigma,
    initial,
    *,
    softening=1.0,
    max_evaluations=200,
    phase_covariance=None,
    interval_half_width=None,
):
    """One shared phase plus regularized per-frame deviations; every width is retained."""
    if interval_half_width is None:
        return original_fit_curve(calibration, heading, observed, sigma, initial, softening=softening, max_evaluations=max_evaluations, phase_covariance=phase_covariance)
    half_width = np.asarray(interval_half_width, float)
    if half_width.shape != np.asarray(observed).shape or not np.isfinite(half_width).all() or (half_width <= 0).any():
        raise InputError('Pixel interval fit needs all finite positive observation intervals')
    heading, observed, sigma = (
        np.asarray(heading, float),
        np.asarray(observed, float),
        np.asarray(sigma, float),
    )
    ref_phase = np.mod(np.asarray(calibration["heading"], float), 2 * np.pi)
    ref_width = np.asarray(calibration["width"], float)
    ref_j = np.asarray(calibration["jacobian"], float)
    if (
        observed.ndim != 2
        or len(heading) != len(observed)
        or sigma.shape != heading.shape
        or not np.isfinite(sigma).all()
        or (sigma <= 0).any()
    ):
        raise InputError("Invalid framewise phase observations")
    if (
        ref_j.shape[:2] != ref_width.shape
        or ref_width.shape[1] != observed.shape[1]
        or not np.isfinite(observed).all()
    ):
        raise InputError("Invalid calibrated width curve")
    order = np.argsort(ref_phase)
    phase = ref_phase[order]
    gaps = np.diff(np.r_[phase, phase[0] + 2 * np.pi])
    if len(phase) < 24 or gaps.min() < 1e-5 or gaps.max() > np.pi / 4:
        raise InputError("Reference turn lacks independent complete angular coverage")
    knots = np.r_[phase, phase[0] + 2 * np.pi]
    curve = CubicSpline(
        knots,
        np.concatenate([ref_width[order], ref_width[order[:1]]]),
        axis=0,
        bc_type="periodic",
    )
    jcurve = CubicSpline(
        knots,
        np.concatenate([ref_j[order], ref_j[order[:1]]]),
        axis=0,
        bc_type="periodic",
    )
    n, rows = observed.shape
    k = ref_j.shape[-1]
    initial_phase = np.deg2rad(initial["phase_refinement_deg"])
    x = np.r_[
        np.asarray(initial["delta_coefficients"], float),
        initial_phase,
        np.full(n, initial_phase),
    ]
    if len(x) != k + 1 + n:
        raise InputError("Invalid initial width fit")
    query_order = np.argsort(np.mod(heading, 2 * np.pi))
    sorted_phase = np.mod(heading, 2 * np.pi)[query_order]
    query_gaps = np.diff(np.r_[sorted_phase, sorted_phase[0] + 2 * np.pi])
    angular = np.zeros(n)
    angular[query_order] = (query_gaps + np.roll(query_gaps, 1)) / 2 * n / (2 * np.pi)
    if (angular <= 0).any():
        raise InputError("Query phases are not independent")
    angular = np.repeat(angular, rows)
    if "noise_sigma" in calibration:
        noise_sigma = np.asarray(calibration["noise_sigma"], float)
        if (
            noise_sigma.shape != (rows,)
            or not np.isfinite(noise_sigma).all()
            or (noise_sigma <= 0).any()
        ):
            raise InputError("Invalid reference width noise")
        angular = angular / np.tile(noise_sigma**2, n)
        if not np.isfinite(angular).all() or (angular <= 0).any():
            raise InputError("Invalid reference width precision")
    if "noise_variance_curve" in calibration:
        reference_variance = np.asarray(calibration["noise_variance_curve"], float)
        if (
            reference_variance.shape != ref_width.shape
            or not np.isfinite(reference_variance).all()
            or (reference_variance <= 0).any()
        ):
            raise InputError("Invalid reference angular width variance")
        observed_variance = periodic_reference(
            calibration["heading"], reference_variance, heading, interpolation="linear"
        )
        if not np.isfinite(observed_variance).all() or (observed_variance <= 0).any():
            raise InputError("Invalid transported width variance")
        if "query_noise_variance" in calibration:
            query_variance = np.asarray(calibration["query_noise_variance"], float)
            if (
                query_variance.shape != observed.shape
                or not np.isfinite(query_variance).all()
                or (query_variance < 0).any()
            ):
                raise InputError("Invalid query width variance")
            observed_variance = observed_variance + query_variance
        angular = angular / observed_variance.ravel()
    penalty = np.zeros((n, len(x)))
    if phase_covariance is None:
        whitener = np.diag(1 / sigma)
    else:
        covariance = np.asarray(phase_covariance, float)
        if (
            covariance.shape != (n, n)
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-12)
        ):
            raise InputError("Invalid phase covariance")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues.min() <= 1e-12:
            raise InputError("Phase covariance is not positive definite")
        whitener = (eigenvectors / np.sqrt(eigenvalues)).T
        sigma = np.sqrt(np.diag(covariance))
    penalty[:, k] = -whitener @ np.ones(n)
    penalty[:, k + 1 :] = whitener

    def evaluate(value, jacobian=False):
        total_phase = value[k + 1 :]
        if abs(total_phase).max() > np.deg2rad(10) or abs(value[:k]).max() > 0.1:
            raise InputError("Framewise phase fit exceeds its local domain")
        q = np.mod(heading + total_phase - phase[0], 2 * np.pi) + phase[0]
        j = jcurve(q)
        residual = (curve(q) + j @ value[:k] - observed).reshape(-1)
        transformed, _ = interval_residual(residual, angular, half_width.ravel(), softening=softening)
        objective = float(np.sum(transformed**2) + np.sum((penalty @ value)**2))
        if not jacobian:
            return objective, residual
        derivative = curve(q, 1) + jcurve(q, 1) @ value[:k]
        matrix = np.zeros((n * rows, len(x)))
        matrix[:, :k] = j.reshape(-1, k)
        for i in range(n):
            matrix[i * rows : (i + 1) * rows, k + 1 + i] = derivative[i]
        return objective, residual, matrix

    from scipy.optimize import least_squares

    def residual_function(value):
        _, residual = evaluate(value)
        transformed, _ = interval_residual(residual, angular, half_width.ravel(), softening=softening)
        return np.r_[transformed, penalty @ value]

    def jacobian_function(value):
        _, _, matrix = evaluate(value, True)
        _, derivative = interval_residual(evaluate(value)[1], angular, half_width.ravel(), softening=softening)
        return np.vstack([matrix * derivative[:, None], penalty])

    limit = np.deg2rad(10)
    lower = np.r_[np.full(k, -0.1), np.full(n + 1, -limit)]
    upper = -lower
    solution = least_squares(
        residual_function,
        x,
        jac=jacobian_function,
        bounds=(lower, upper),
        loss='linear',
        method="trf",
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-8,
        max_nfev=max_evaluations,
    )
    x = solution.x
    status = (
        "converged_"
        + {1: "gradient", 2: "objective", 3: "step", 4: "objective_and_step"}.get(
            solution.status, "unknown"
        )
        if solution.success
        else "evaluation_limit"
    )
    objective, residual, matrix = evaluate(x, True)
    transformed, derivative = interval_residual(residual, angular, half_width.ravel(), softening=softening)
    independent_residual, independent_derivative = interval_residual(residual, angular, half_width.ravel(), softening=softening, nodes=128)
    quadrature_error = float(abs(transformed - independent_residual).max())
    derivative_error = float(abs(derivative - independent_derivative).max())
    if quadrature_error > 1e-7 or derivative_error > 1e-6:
        raise InputError('Pixel interval quadrature is insufficient at the actual final fit')
    weight = derivative**2
    # Profile nuisance phase using its prior, retaining a separate shape-information check.
    hessian = matrix.T @ (weight[:, None] * matrix) + penalty.T @ penalty
    information = (
        hessian[:k, :k]
        - hessian[:k, k:]
        @ np.linalg.pinv(hessian[k:, k:], rcond=1e-10)
        @ hessian[k:, :k]
    )
    eigen = np.linalg.eigvalsh((information + information.T) / 2)
    singular = np.sqrt(np.maximum(eigen, 0))[::-1]
    if singular[0] <= 1e-8 or singular[-1] < singular[0] * 1e-4:
        raise InputError("Shape response is unobservable after phase uncertainty")
    return {
        "delta_coefficients": x[:k],
        "singular_values": singular,
        "phase_refinement_deg": float(np.rad2deg(x[k])),
        "frame_phase_deviations_deg": np.rad2deg(x[k + 1 :] - x[k]).tolist(),
        "total_frame_phase_deg": np.rad2deg(x[k + 1 :]).tolist(),
        "phase_sigma_deg": np.rad2deg(sigma).tolist(),
        "phase_prior_mode": "correlated_smoother"
        if phase_covariance is not None
        else "independent_frames",
        "frame_phase_deviation_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean((x[k + 1 :] - x[k]) ** 2)))
        ),
        "width_residual_rms_reference_px": float(np.sqrt(np.mean(residual**2))),
        "objective": objective,
        "width_likelihood": "uniform_pixel_interval_marginalized_pseudo_huber",
        "interval_half_width_reference_px": half_width.tolist(),
        "quadrature_32_vs128_residual_max_error": quadrature_error,
        "quadrature_32_vs128_derivative_max_error": derivative_error,
        "function_evaluations": solution.nfev,
        "solver_status": status,
        "solver_optimality": float(solution.optimality),
        "solver_status_code": solution.status,
        "all_observation_rows_retained": True,
        "frame_count": n,
        "row_count": rows,
        "physical_accuracy_validated": False,
        "registration": "native image-estimated pose with heuristic uncertainty-constrained framewise phases",
        "registration_interpolation": "cubic",
        "projection_normalization": "depth",
        "row_correspondence": calibration["row_mode"],
        **(
            {
                "width_noise_mode": calibration.get(
                    "noise_mode", "reference_angular_variance"
                ),
                "width_noise_variance_reference_px2": observed_variance.tolist(),
            }
            if "noise_variance_curve" in calibration
            else {}
        ),
        **(
            {
                "width_noise_mode": calibration.get("noise_mode", "declared_row_std"),
                "width_noise_sigma_reference_px": noise_sigma.tolist(),
            }
            if "noise_sigma" in calibration
            else {}
        ),
    }
