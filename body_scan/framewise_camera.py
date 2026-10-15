"""Research-only pose-uncertainty-constrained longitudinal width fit."""

import numpy as np
from scipy.interpolate import CubicSpline

from body_scan.io import InputError
from body_scan.phase_registration import periodic_reference


def pose_sigma(query, reference):
    """A bounded heuristic, not a validated physical angle confidence interval."""
    q = np.asarray(query, float)
    r = np.asarray(reference, float)
    if (
        q.ndim != 1
        or r.ndim != 1
        or min(len(q), len(r)) < 24
        or not np.isfinite(q).all()
        or not np.isfinite(r).all()
        or min(q.min(), r.min()) < 0
    ):
        raise InputError("Invalid image-estimated phase uncertainty")
    lower, upper = np.deg2rad([0.25, 5])
    return np.sqrt(
        np.clip(q, lower, upper) ** 2 + np.median(np.clip(r, lower, upper)) ** 2
    )


def fit_curve_camera(
    calibration,
    heading,
    observed,
    sigma,
    initial,
    *,
    camera_derivative,
    camera_prior_mean,
    camera_prior_sigma,
    camera_bound,
    softening=1.0,
    max_evaluations=200,
    phase_covariance=None,
):
    """One shared phase plus regularized per-frame deviations; every width is retained."""
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
    camera_derivative = np.asarray(camera_derivative, float)
    if (
        camera_derivative.shape != observed.shape
        or not np.isfinite(camera_derivative).all()
        or not np.isfinite([camera_prior_mean, camera_prior_sigma, camera_bound]).all()
        or camera_prior_sigma <= 0
        or camera_bound <= 0
    ):
        raise InputError("Invalid joint camera width prior")
    initial_phase = np.deg2rad(initial["phase_refinement_deg"])
    x = np.r_[
        np.asarray(initial["delta_coefficients"], float),
        initial_phase,
        np.full(n, initial_phase),
        0.0,
    ]
    if len(x) != k + 2 + n:
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
    penalty = np.zeros((n + 1, len(x)))
    penalty_target = np.zeros(n + 1)
    penalty_target[-1] = camera_prior_mean / camera_prior_sigma
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
    penalty[:n, k] = -whitener @ np.ones(n)
    penalty[:n, k + 1 : -1] = whitener
    penalty[-1, -1] = 1 / camera_prior_sigma

    def evaluate(value, jacobian=False):
        total_phase = value[k + 1 : -1]
        if abs(total_phase).max() > np.deg2rad(10) or abs(value[:k]).max() > 0.1:
            raise InputError("Framewise phase fit exceeds its local domain")
        q = np.mod(heading + total_phase - phase[0], 2 * np.pi) + phase[0]
        j = jcurve(q)
        residual = (
            curve(q) + j @ value[:k] - observed - camera_derivative * value[-1]
        ).reshape(-1)
        objective = float(
            np.sum(
                angular
                * 2
                * softening**2
                * (np.sqrt(1 + (residual / softening) ** 2) - 1)
            )
            + np.sum((penalty @ value - penalty_target) ** 2)
        )
        if not jacobian:
            return objective, residual
        derivative = curve(q, 1) + jcurve(q, 1) @ value[:k]
        matrix = np.zeros((n * rows, len(x)))
        matrix[:, :k] = j.reshape(-1, k)
        for i in range(n):
            matrix[i * rows : (i + 1) * rows, k + 1 + i] = derivative[i]
        matrix[:, -1] = -camera_derivative.ravel()
        return objective, residual, matrix

    from scipy.optimize import least_squares

    def residual_function(value):
        _, residual = evaluate(value)
        return np.r_[residual * np.sqrt(angular), penalty @ value - penalty_target]

    def jacobian_function(value):
        _, _, matrix = evaluate(value, True)
        return np.vstack([matrix * np.sqrt(angular[:, None]), penalty])

    def loss_function(z):
        result = np.zeros((3, len(z)))
        result[0] = z
        result[1] = 1.0
        scale = angular * softening**2
        t = 1 + z[: n * rows] / scale
        result[0, : n * rows] = 2 * scale * (np.sqrt(t) - 1)
        result[1, : n * rows] = t ** (-0.5)
        result[2, : n * rows] = -0.5 / scale * t ** (-1.5)
        return result

    limit = np.deg2rad(10)
    lower = np.r_[np.full(k, -0.1), np.full(n + 1, -limit), -camera_bound]
    upper = -lower
    solution = least_squares(
        residual_function,
        x,
        jac=jacobian_function,
        bounds=(lower, upper),
        loss=loss_function,
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
    weight = angular / np.sqrt(1 + (residual / softening) ** 2)
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
        "camera_depth_offset_model_m": float(x[-1]),
        "camera_prior_mean_model_m": float(camera_prior_mean),
        "camera_prior_sigma_model_m": float(camera_prior_sigma),
        "camera_information_profiled_in_shape": True,
        "phase_refinement_deg": float(np.rad2deg(x[k])),
        "frame_phase_deviations_deg": np.rad2deg(x[k + 1 : -1] - x[k]).tolist(),
        "total_frame_phase_deg": np.rad2deg(x[k + 1 : -1]).tolist(),
        "phase_sigma_deg": np.rad2deg(sigma).tolist(),
        "phase_prior_mode": "correlated_smoother"
        if phase_covariance is not None
        else "independent_frames",
        "frame_phase_deviation_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean((x[k + 1 : -1] - x[k]) ** 2)))
        ),
        "width_residual_rms_reference_px": float(np.sqrt(np.mean(residual**2))),
        "objective": objective,
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
        "joint_camera_linearized": True,
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
