"""Explicit, scaled age-moment losses and a derivative-free calibration path.

No survey measurement map is assumed: targets must supply their provenance,
units, age origin, participation convention and residual scales. A synthetic
exercise checks the numerical pipeline, not empirical identification.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .bellman import BellmanConfig, solve_bellman
from .continuum import PopulationNodes, PopulationResult, simulate_population
from .model import ModelParams

MOMENT_UNITS = {
    "hours": "fraction of model time endowment",
    "participation": "population fraction",
    "training_time": "fraction of model time endowment",
    "consumption": "model goods per model year",
    "earnings": "model goods per model year",
    "assets": "model goods",
    "human_capital": "model human capital units",
    "log_human_capital": "log model human capital units",
    "asset_floor_mass": "population fraction at numerical asset floor",
}


@dataclass(frozen=True)
class AgeMomentTarget:
    """One unconditional age profile, including an explicit measurement scale.

    ``scale`` has the same units as ``values`` (e.g. a standard error or a
    substantively chosen tolerance), and ``weights`` are dimensionless relative
    weights per observation. Arrays have one entry per age; scale and weights
    can instead be positive/nonnegative scalars. Calendar ages are mapped to
    model time by subtracting the problem's ``age_origin``.
    """

    moment: str
    ages: np.ndarray
    values: np.ndarray
    scale: Any
    weights: Any
    units: str


@dataclass(frozen=True)
class CalibrationTargets:
    """Targets with provenance and exact model participation convention."""

    profiles: tuple[AgeMomentTarget, ...]
    age_origin: float
    participation_hours_threshold: float
    source: str


@dataclass(frozen=True)
class MomentLoss:
    """Weighted mean squared standardized residuals, a dimensionless loss."""

    loss: float
    predicted: dict[str, np.ndarray]
    standardized_residuals: dict[str, np.ndarray]
    weighted_squared_residuals: dict[str, np.ndarray]
    total_weight: float


@dataclass(frozen=True)
class CalibrationEvaluation:
    params: ModelParams
    population: PopulationResult
    objective: MomentLoss
    solve_seconds: float
    population_and_loss_seconds: float
    total_seconds: float


def weighted_age_moment_loss(
    population: PopulationResult, targets: CalibrationTargets
) -> MomentLoss:
    """Integrate model profiles into L = sum(w*((m-target)/scale)²)/sum(w).

    Linear interpolation in age uses only the supported period-start times
    for controls/earnings and all boundaries for states. Extrapolation is
    rejected, including terminal-age control targets. No cross-sectional age
    distribution is implicit: observation weights must be supplied explicitly.
    """

    if not targets.profiles or not targets.source.strip() or not np.isfinite(targets.age_origin):
        raise ValueError("targets require profiles, finite age_origin and explicit source")
    threshold = float(targets.participation_hours_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("target participation threshold must lie in [0, 1)")
    if threshold != population.moments.participation_hours_threshold:
        raise ValueError("model and target participation thresholds must match")
    predicted: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    contributions: dict[str, np.ndarray] = {}
    total_weight, total_loss = 0.0, 0.0
    for profile in targets.profiles:
        name = profile.moment
        if name not in MOMENT_UNITS or name in predicted:
            raise ValueError("moment names must be supported and unique")
        if profile.units != MOMENT_UNITS[name]:
            raise ValueError(f"{name} targets must be converted to units: {MOMENT_UNITS[name]}")
        ages = np.asarray(profile.ages, dtype=np.float64)
        values = np.asarray(profile.values, dtype=np.float64)
        if ages.ndim != 1 or ages.size == 0 or values.shape != ages.shape:
            raise ValueError("target ages and values must be nonempty matching one-dimensional arrays")
        try:
            scale = np.broadcast_to(np.asarray(profile.scale, dtype=np.float64), ages.shape)
            weights = np.broadcast_to(np.asarray(profile.weights, dtype=np.float64), ages.shape)
        except ValueError as error:
            raise ValueError("target scale and weights must broadcast to target ages") from error
        if not all(np.all(np.isfinite(x)) for x in (ages, values, scale, weights)):
            raise ValueError("targets, scales and weights must be finite")
        if np.any(scale <= 0.0) or np.any(weights < 0.0):
            raise ValueError("target scales must be positive and weights nonnegative")
        states = name in {"assets", "human_capital", "log_human_capital", "asset_floor_mass"}
        model = population.state_moments if states else population.moments
        model_time = np.asarray(model.time)
        requested_time = ages - targets.age_origin
        if np.any(requested_time < model_time[0]) or np.any(requested_time > model_time[-1]):
            raise ValueError(f"{name} target ages lie outside the model profile; no extrapolation")
        model_values = np.asarray(getattr(model, name))
        if not np.all(np.isfinite(model_values)):
            raise ValueError(f"{name} model profile contains nonfinite values")
        prediction = np.interp(requested_time, model_time, model_values)
        residual = (prediction - values) / scale
        contribution = weights * residual**2
        predicted[name], residuals[name], contributions[name] = prediction, residual, contribution
        total_weight += float(weights.sum())
        total_loss += float(contribution.sum())
    if total_weight <= 0.0 or not np.isfinite(total_weight) or not np.isfinite(total_loss):
        raise ValueError("loss must have finite positive total weight and finite residual loss")
    return MomentLoss(total_loss / total_weight, predicted, residuals, contributions, total_weight)


def evaluate_calibration(
    params: ModelParams,
    config: BellmanConfig,
    initial_nodes: PopulationNodes,
    targets: CalibrationTargets,
    *,
    backend: Literal["quadrature", "cohort", "transport"] = "quadrature",
    distribution_grid: Any = None,
    store_snapshots: bool = False,
) -> CalibrationEvaluation:
    """Complete policy solve, feasible population evolution and scaled loss.

    Fixed solver/grid/node shapes reuse the underlying JAX caches across
    parameter changes. Host validation and discrete participation do not form
    a differentiable objective; use derivative-free search unless derivatives
    have been separately validated at several finite-difference step sizes.
    """

    started = perf_counter()
    solution = solve_bellman(params, config)
    solved = perf_counter()
    population = simulate_population(
        solution, initial_nodes, backend=backend,
        participation_hours_threshold=targets.participation_hours_threshold,
        distribution_grid=distribution_grid, store_snapshots=store_snapshots,
    )
    objective = weighted_age_moment_loss(population, targets)
    finished = perf_counter()
    return CalibrationEvaluation(
        params, population, objective, solved - started, finished - solved, finished - started
    )


def fit_scalar_calibration(
    parameter: str,
    bounds: tuple[float, float],
    *,
    params: ModelParams,
    config: BellmanConfig,
    initial_nodes: PopulationNodes,
    targets: CalibrationTargets,
    backend: Literal["quadrature", "cohort", "transport"] = "quadrature",
    distribution_grid: Any = None,
    parameter_tolerance: float = 1e-3,
    max_evaluations: int = 30,
) -> Any:
    """Bounded derivative-free one-parameter fit with recorded evaluations.

    Returns SciPy's OptimizeResult with ``evaluations`` and ``best_evaluation``.
    Numerical success does not establish identification or global optimality.
    Fix shape/domain parameters: this routine estimates ModelParams fields only.
    """

    from scipy.optimize import minimize_scalar

    if parameter not in params._fields or parameter in {"horizon", "asset_floor"}:
        raise ValueError("parameter must be a ModelParams field other than horizon or asset_floor")
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be finite and strictly ordered")
    if not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0 or max_evaluations < 1:
        raise ValueError("parameter tolerance and maximum evaluations must be positive")
    evaluations: list[CalibrationEvaluation] = []

    def objective(value: float) -> float:
        evaluation = evaluate_calibration(
            params._replace(**{parameter: float(value)}), config, initial_nodes, targets,
            backend=backend, distribution_grid=distribution_grid,
        )
        evaluations.append(evaluation)
        return evaluation.objective.loss

    result = minimize_scalar(
        objective, bounds=bounds, method="bounded",
        options={"xatol": parameter_tolerance, "maxiter": max_evaluations},
    )
    result.evaluations = evaluations
    result.best_evaluation = min(evaluations, key=lambda evaluation: evaluation.objective.loss)
    return result
