"""Explicit, scaled age-moment losses and a derivative-free calibration path.

No survey measurement map is assumed: targets must supply their provenance,
units, age origin, participation convention and residual scales. A synthetic
exercise checks the numerical pipeline, not empirical identification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .bellman import BellmanConfig, BellmanSolution, solve_bellman
from .continuum import (
    PopulationNodes,
    PopulationResult,
    SyntheticInitialDistribution,
    initial_quadrature,
    simulate_population,
)
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
    "near_asset_floor_mass": "population fraction within a fixed numerical-floor asset band",
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
    near_asset_floor_width: float | None = None


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
    if any(profile.moment == "near_asset_floor_mass" for profile in targets.profiles):
        width = targets.near_asset_floor_width
        if width is None or not np.isfinite(width) or width <= 0:
            raise ValueError("near-floor targets require a finite positive near_asset_floor_width")
        if width != population.state_moments.near_asset_floor_width:
            raise ValueError("model and target near-floor widths must match")
        if population.state_moments.near_asset_floor_mass is None:
            raise ValueError("population does not contain the requested near-floor moment")
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
            raise ValueError(
                "target ages and values must be nonempty matching one-dimensional arrays"
            )
        try:
            scale = np.broadcast_to(np.asarray(profile.scale, dtype=np.float64), ages.shape)
            weights = np.broadcast_to(np.asarray(profile.weights, dtype=np.float64), ages.shape)
        except ValueError as error:
            raise ValueError("target scale and weights must broadcast to target ages") from error
        if not all(np.all(np.isfinite(x)) for x in (ages, values, scale, weights)):
            raise ValueError("targets, scales and weights must be finite")
        if np.any(scale <= 0.0) or np.any(weights < 0.0):
            raise ValueError("target scales must be positive and weights nonnegative")
        states = name in {
            "assets",
            "human_capital",
            "log_human_capital",
            "asset_floor_mass",
            "near_asset_floor_mass",
        }
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
        solution,
        initial_nodes,
        backend=backend,
        participation_hours_threshold=targets.participation_hours_threshold,
        distribution_grid=distribution_grid,
        store_snapshots=store_snapshots,
        near_asset_floor_width=targets.near_asset_floor_width,
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
    evaluation_callback: Callable[[CalibrationEvaluation], None] | None = None,
) -> Any:
    """Bounded derivative-free one-parameter fit with recorded evaluations.

    Evaluate the supplied parameter as an incumbent when it lies in bounds;
    the returned x/fun always select the best actually evaluated candidate.
    ``raw_optimizer`` preserves the bounded search result and termination,
    and ``selection_source`` identifies an incumbent retained over that search.
    The evaluation cap includes the incumbent. Success records termination,
    not identification, global optimality, or numerical-resolution acceptance.
    Fix shape/domain parameters: this routine estimates ModelParams fields only.
    """

    from scipy.optimize import OptimizeResult, minimize_scalar

    if parameter not in params._fields or parameter in {"horizon", "asset_floor"}:
        raise ValueError("parameter must be a ModelParams field other than horizon or asset_floor")
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be finite and strictly ordered")
    if not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0:
        raise ValueError("parameter_tolerance must be finite and positive")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    evaluations: list[CalibrationEvaluation] = []

    def objective(value: float) -> float:
        evaluation = evaluate_calibration(
            params._replace(**{parameter: float(value)}),
            config,
            initial_nodes,
            targets,
            backend=backend,
            distribution_grid=distribution_grid,
        )
        evaluations.append(evaluation)
        if evaluation_callback is not None:
            evaluation_callback(evaluation)
        return evaluation.objective.loss

    initial_value = float(getattr(params, parameter))
    incumbent = None
    if bounds[0] <= initial_value <= bounds[1]:
        objective(initial_value)
        incumbent = evaluations[-1]
    remaining = max_evaluations - len(evaluations)
    result: Any
    if remaining:
        result = minimize_scalar(
            objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": parameter_tolerance, "maxiter": remaining},
        )
        result.raw_optimizer = {
            key: result[key] for key in ("x", "fun", "success", "status", "message", "nfev")
        }
    else:
        result = OptimizeResult(
            success=False, status=1, message="Evaluation budget consumed by initial parameter."
        )
        result.raw_optimizer = None
    result.evaluations = evaluations
    result.best_evaluation = min(evaluations, key=lambda evaluation: evaluation.objective.loss)
    result.x = float(getattr(result.best_evaluation.params, parameter))
    result.fun = result.best_evaluation.objective.loss
    result.nfev = len(evaluations)
    result.selection_source = (
        "initial_parameter" if result.best_evaluation is incumbent else "bounded_search"
    )
    return result


@dataclass(frozen=True)
class InitialLawCalibrationEvaluation:
    """An initial-law evaluation reusing a fixed structural policy solution."""

    law: SyntheticInitialDistribution
    nodes_per_dimension: int
    population: PopulationResult
    objective: MomentLoss
    total_seconds: float


def evaluate_initial_law(
    solution: BellmanSolution,
    law: SyntheticInitialDistribution,
    targets: CalibrationTargets,
    *,
    nodes_per_dimension: int = 16,
) -> InitialLawCalibrationEvaluation:
    """Evaluate initial heterogeneity without an unnecessary Bellman re-solve."""
    started = perf_counter()
    nodes = initial_quadrature(
        law, nodes_per_dimension=nodes_per_dimension, asset_floor=solution.config.asset_minimum
    )
    population = simulate_population(
        solution,
        nodes,
        participation_hours_threshold=targets.participation_hours_threshold,
        near_asset_floor_width=targets.near_asset_floor_width,
    )
    loss = weighted_age_moment_loss(population, targets)
    return InitialLawCalibrationEvaluation(
        law, nodes_per_dimension, population, loss, perf_counter() - started
    )


def _replace_initial_parameter(law, parameter, value, atom_index):
    if parameter == "atom_mass":
        if (
            isinstance(atom_index, bool)
            or not isinstance(atom_index, (int, np.integer))
            or not 0 <= atom_index < len(law.atoms)
        ):
            raise ValueError("atom_mass requires an explicit valid atom_index and atom location")
        atoms = list(law.atoms)
        atoms[atom_index] = replace(atoms[atom_index], mass=float(value))
        return replace(law, atoms=tuple(atoms))
    if parameter not in {
        "asset_lower",
        "asset_upper",
        "log_human_capital_lower",
        "log_human_capital_upper",
        "correlation",
        "asset_floor_mass",
    }:
        raise ValueError("unsupported initial-law parameter")
    if atom_index is not None:
        raise ValueError("atom_index is used only when estimating atom_mass")
    return replace(law, **{parameter: float(value)})


def fit_initial_law_calibration(
    parameter: str,
    bounds: tuple[float, float],
    *,
    solution: BellmanSolution,
    law: SyntheticInitialDistribution,
    targets: CalibrationTargets,
    nodes_per_dimension: int = 16,
    atom_index: int | None = None,
    parameter_tolerance: float = 1e-3,
    max_evaluations: int = 30,
) -> Any:
    """Estimate one assumed-law input with a fixed structural policy solution.

    Supports floor share, continuous-component correlation and support bounds.
    ``atom_mass`` requires an explicit atom location/index, including a zero-mass
    placeholder if desired; it never inserts a population atom implicitly.
    These fits need observed initial-distribution/state information to identify
    initial heterogeneity separately from structural preferences/technology.
    """
    from scipy.optimize import minimize_scalar

    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bounds must be finite and strictly ordered")
    if not np.isfinite(parameter_tolerance) or parameter_tolerance <= 0:
        raise ValueError("parameter_tolerance must be finite and positive")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, (int, np.integer))
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer")
    # Both endpoints are checked before requesting any greedy rollout/compilation.
    for endpoint in bounds:
        candidate = _replace_initial_parameter(law, parameter, endpoint, atom_index)
        nodes = initial_quadrature(
            candidate,
            nodes_per_dimension=nodes_per_dimension,
            asset_floor=solution.config.asset_minimum,
        )
        if (
            np.min(nodes.assets) < solution.asset_grid[0]
            or np.max(nodes.assets) > solution.asset_grid[-1]
            or np.min(np.log(nodes.human_capital)) < solution.log_human_capital_grid[0]
            or np.max(np.log(nodes.human_capital)) > solution.log_human_capital_grid[-1]
        ):
            raise ValueError("initial-law fitting bounds must remain inside the Bellman domain")
    evaluations: list[InitialLawCalibrationEvaluation] = []

    def objective(value):
        evaluation = evaluate_initial_law(
            solution,
            _replace_initial_parameter(law, parameter, value, atom_index),
            targets,
            nodes_per_dimension=nodes_per_dimension,
        )
        evaluations.append(evaluation)
        return evaluation.objective.loss

    fit: Any = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": parameter_tolerance, "maxiter": max_evaluations},
    )
    fit.evaluations = evaluations
    fit.best_evaluation = min(evaluations, key=lambda evaluation: evaluation.objective.loss)
    fit.parameter_name, fit.atom_index = parameter, atom_index
    return fit


@dataclass(frozen=True)
class NumericalAcceptanceThresholds:
    """Explicit numerical tolerances; moment differences use target residual scales."""

    maximum_standardized_moment_difference: float
    maximum_loss_difference: float
    maximum_parameter_difference: float
    maximum_full_period_floor_violation: float = 1e-10


def compare_calibration_resolutions(
    coarse: CalibrationEvaluation,
    reference: CalibrationEvaluation,
    targets: CalibrationTargets,
    *,
    fitted_parameters: tuple[float, float],
    optimizer_success: tuple[bool, bool],
    thresholds: NumericalAcceptanceThresholds,
) -> dict[str, Any]:
    """Gate moments, loss and fitted parameters against a fixed numerical reference.

    Baseline evaluations must use the same structural parameters and explicit
    initial law. The numerical configuration or quadrature order must differ:
    matching a setup to itself cannot provide convergence evidence. Comparisons
    use common target ages, their units/scales and positive-weight observations.
    A single passing pair is local stability evidence, not a continuum or global
    optimization certificate. Check further orders, time/state grids and domains.
    """
    tolerance_values = np.asarray(list(asdict(thresholds).values()))
    if not np.all(np.isfinite(tolerance_values)) or np.any(tolerance_values < 0):
        raise ValueError("numerical acceptance tolerances must be finite and nonnegative")
    if len(optimizer_success) != 2:
        raise ValueError("two optimizer success flags are required")
    if len(fitted_parameters) != 2 or not np.all(np.isfinite(fitted_parameters)):
        raise ValueError("two finite fitted parameter values are required")
    coarse_loss = weighted_age_moment_loss(coarse.population, targets)
    reference_loss = weighted_age_moment_loss(reference.population, targets)
    differences, standardized = {}, []
    for profile in targets.profiles:
        weights = np.broadcast_to(np.asarray(profile.weights), np.shape(profile.ages))
        selected = weights > 0
        if not np.any(selected):
            continue
        scale = np.broadcast_to(np.asarray(profile.scale), np.shape(profile.ages))[selected]
        difference = np.abs(
            coarse_loss.predicted[profile.moment][selected]
            - reference_loss.predicted[profile.moment][selected]
        )
        differences[profile.moment] = float(np.max(difference))
        standardized.append(float(np.max(difference / scale)))
    max_standardized = max(standardized)
    loss_difference = abs(coarse_loss.loss - reference_loss.loss)
    parameter_difference = abs(fitted_parameters[0] - fitted_parameters[1])
    coarse_diagnostics, fine_diagnostics = (
        coarse.population.diagnostics,
        reference.population.diagnostics,
    )
    same_law = coarse_diagnostics.get("initial_law") is not None and coarse_diagnostics[
        "initial_law"
    ] == fine_diagnostics.get("initial_law")
    same_params = coarse.params == reference.params
    coarse_floor = coarse.population.simulation.solution.config.asset_minimum
    reference_floor = reference.population.simulation.solution.config.asset_minimum
    # A positive initial floor mixture moves physically when the numerical floor changes.
    if same_law and coarse_diagnostics["initial_law"]["asset_floor_mass"] > 0:
        same_law = coarse_floor == reference_floor

    def numerical_config(population):
        config = asdict(population.simulation.solution.config)
        for field in ("compute_platform", "device_index", "control_batch_size"):
            config.pop(field)
        if config["asset_feasibility"] == "continuous":
            config.pop("path_checkpoints")
        return config

    distinct_resolution = numerical_config(coarse.population) != numerical_config(
        reference.population
    ) or coarse_diagnostics.get("quadrature_order") != fine_diagnostics.get("quadrature_order")
    floor_violations = [
        diagnostics.get("maximum_full_period_floor_violation", np.inf)
        for diagnostics in (coarse_diagnostics, fine_diagnostics)
    ]
    feasible = bool(
        np.all(np.isfinite(floor_violations))
        and np.max(floor_violations) <= thresholds.maximum_full_period_floor_violation
    )
    criteria = {
        "same_structural_parameters": bool(same_params),
        "same_explicit_initial_law": same_law,
        "distinct_numerical_resolutions": distinct_resolution,
        "moments_within_tolerance": max_standardized
        <= thresholds.maximum_standardized_moment_difference,
        "loss_within_tolerance": loss_difference <= thresholds.maximum_loss_difference,
        "fitted_parameter_within_tolerance": parameter_difference
        <= thresholds.maximum_parameter_difference,
        "both_optimizers_succeeded": bool(all(optimizer_success)),
        "both_full_period_paths_feasible": feasible,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": asdict(thresholds),
        "maximum_absolute_moment_difference": differences,
        "maximum_standardized_moment_difference": max_standardized,
        "absolute_loss_difference": loss_difference,
        "absolute_fitted_parameter_difference": parameter_difference,
        "maximum_full_period_floor_violations": floor_violations,
        "scope": "local synthetic stability; no empirical or continuum certification",
    }
