"""Numerical and economic diagnostics for collocation solutions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from .model import (
    dynamics,
    earnings_tradeoff,
    earnings_tradeoff_prime,
    marginal_power_utility,
)
from .solver import CollocationResult
from .transcription import (
    asset_path_slack,
    collocation_defects,
    decision_bounds,
    equality_constraints,
    scaled_negative_objective,
    training_time_slack,
)


@dataclass(frozen=True)
class PathDiagnostics:
    """A compact set of acceptance checks for an optimized lifecycle."""

    optimizer_success: bool
    accepted_success: bool
    lifetime_utility: float
    max_equality_violation: float
    max_unscaled_dynamic_defect: float
    projected_kkt_residual: float
    max_consumption_foc_residual: float
    max_hours_foc_violation: float
    max_investment_foc_violation: float
    independent_asset_error: float
    independent_human_capital_relative_error: float
    minimum_assets: float
    minimum_collocation_path_assets: float
    minimum_independently_integrated_assets: float
    minimum_consumption: float
    minimum_leisure: float
    asset_floor_binding_nodes: int
    normal_regime_order: bool
    regime_sequence: str

    def as_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


def regime_labels(
    hours: np.ndarray, investment_share: np.ndarray, *, tolerance: float = 1e-4
) -> np.ndarray:
    """Classify nodes as schooling, training, work, or retirement."""

    labels = np.full(hours.shape, "on-the-job training", dtype=object)
    labels[investment_share <= tolerance] = "pure work"
    labels[investment_share >= 1.0 - tolerance] = "schooling"
    labels[hours <= tolerance] = "retirement"
    return labels.astype(str)


def _compressed_regime_sequence(labels: np.ndarray) -> list[str]:
    if labels.size == 0:
        return []
    sequence = [str(labels[0])]
    for label in labels[1:]:
        if label != sequence[-1]:
            sequence.append(str(label))
    return sequence


def _normal_regime_order(sequence: list[str]) -> bool:
    rank = {
        "schooling": 0,
        "on-the-job training": 1,
        "pure work": 2,
        "retirement": 3,
    }
    ranks = [rank[label] for label in sequence]
    return all(current <= following for current, following in pairwise(ranks))


def _current_value_costates(result: CollocationResult) -> tuple[np.ndarray, np.ndarray]:
    """Recover continuous-time costates using terminal transversality."""

    params = result.params
    time = result.time
    step = params.horizon / result.config.intervals
    terminal_mu = params.bequest_weight * result.assets[-1] ** (params.bequest_power - 1.0)
    asset_costate = terminal_mu * np.exp(
        (params.rho - params.interest_rate) * (time - params.horizon)
    )

    coefficient = (
        params.rho
        + params.human_capital_depreciation
        - params.human_capital_productivity * result.investment_share * result.hours
    )
    source = (
        asset_costate
        * result.hours
        * np.asarray(earnings_tradeoff(jnp.asarray(result.investment_share)))
    )
    human_capital_costate = np.zeros_like(time)
    for index in range(time.size - 2, -1, -1):
        numerator = human_capital_costate[index + 1] * (
            1.0 - 0.5 * step * coefficient[index + 1]
        ) + 0.5 * step * (source[index] + source[index + 1])
        denominator = 1.0 + 0.5 * step * coefficient[index]
        human_capital_costate[index] = numerator / denominator
    return asset_costate, human_capital_costate


def _max_bound_foc_violation(
    values: np.ndarray,
    derivatives: np.ndarray,
    lower: float,
    upper: float,
    *,
    tolerance: float = 1e-5,
) -> float:
    """FOC violation for maximizing a differentiable function over a box."""

    at_lower = values <= lower + tolerance
    at_upper = values >= upper - tolerance
    interior = ~(at_lower | at_upper)
    violations = np.zeros_like(derivatives)
    # At a lower bound, a positive derivative is an improving feasible move.
    violations[at_lower] = np.maximum(derivatives[at_lower], 0.0)
    # At an upper bound, a negative derivative is an improving feasible move.
    violations[at_upper] = np.maximum(-derivatives[at_upper], 0.0)
    violations[interior] = np.abs(derivatives[interior])
    return float(np.max(violations))


def _continuous_foc_diagnostics(
    result: CollocationResult,
) -> tuple[float, float, float]:
    params = result.params
    asset_costate, human_capital_costate = _current_value_costates(result)
    human_capital = result.human_capital
    tradeoff = np.asarray(earnings_tradeoff(jnp.asarray(result.investment_share)))
    tradeoff_prime = np.asarray(earnings_tradeoff_prime(jnp.asarray(result.investment_share)))

    consumption_marginal_utility = params.consumption_weight * np.asarray(
        marginal_power_utility(jnp.asarray(result.consumption), params.consumption_power)
    )
    consumption_residual = np.abs(consumption_marginal_utility - asset_costate) / (
        1.0 + np.abs(asset_costate)
    )

    leisure_marginal_utility = params.leisure_weight * np.asarray(
        marginal_power_utility(jnp.asarray(result.leisure), params.leisure_power)
    )
    hours_derivative = (
        -leisure_marginal_utility
        + asset_costate * human_capital * tradeoff
        + human_capital_costate
        * params.human_capital_productivity
        * result.investment_share
        * human_capital
    )
    investment_derivative = (
        result.hours
        * human_capital
        * (
            asset_costate * tradeoff_prime
            + params.human_capital_productivity * human_capital_costate
        )
    )
    asset_tolerance = 1e-6 * max(1.0, abs(params.initial_assets))
    binding_indices = np.flatnonzero(result.assets <= params.asset_floor + asset_tolerance)
    valid = np.ones(result.time.size, dtype=bool)
    if binding_indices.size:
        # The simple analytic asset costate omits the state-constraint
        # multiplier. It is valid only after the path leaves its last contact
        # with the borrowing limit.
        valid[: binding_indices[-1] + 1] = False
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")

    hours_violation = _max_bound_foc_violation(
        result.hours[valid],
        hours_derivative[valid],
        0.0,
        1.0 - result.config.leisure_floor,
    )
    investment_violation = _max_bound_foc_violation(
        result.investment_share[valid], investment_derivative[valid], 0.0, 1.0
    )
    return (
        float(np.max(consumption_residual[valid])),
        hours_violation,
        investment_violation,
    )


def _projected_kkt_residual(result: CollocationResult) -> float:
    """Estimate first-order NLP stationarity, including active box bounds."""

    params = result.params
    config = result.config
    decision = jnp.asarray(result.decision)

    def objective(vector: jax.Array) -> jax.Array:
        return scaled_negative_objective(
            vector,
            params,
            config.intervals,
            config.scheme,
        )

    def constraints(vector: jax.Array) -> jax.Array:
        return equality_constraints(vector, params, config.intervals, config.scheme)

    gradient = np.asarray(jax.grad(objective)(decision))
    equality_jacobian = np.asarray(jax.jacrev(constraints)(decision))
    jacobian_parts = [equality_jacobian]
    active_inequality_count = 0

    inequality_values = np.asarray(training_time_slack(decision, config.intervals))
    inequality_jacobian = np.asarray(
        jax.jacrev(lambda vector: training_time_slack(vector, config.intervals))(decision)
    )
    active_inequalities = inequality_values <= 1e-6
    if np.any(active_inequalities):
        active_jacobian = inequality_jacobian[active_inequalities]
        jacobian_parts.append(active_jacobian)
        active_inequality_count += active_jacobian.shape[0]

    path_values = np.asarray(asset_path_slack(decision, params, config.intervals, config.scheme))
    path_jacobian = np.asarray(
        jax.jacrev(
            lambda vector: asset_path_slack(
                vector,
                params,
                config.intervals,
                config.scheme,
            )
        )(decision)
    )
    active_path_constraints = path_values <= 1e-6
    if np.any(active_path_constraints):
        active_jacobian = path_jacobian[active_path_constraints]
        jacobian_parts.append(active_jacobian)
        active_inequality_count += active_jacobian.shape[0]
    jacobian = np.vstack(jacobian_parts)
    lower, upper = decision_bounds(
        params,
        config.intervals,
        consumption_floor=config.consumption_floor,
        leisure_floor=config.leisure_floor,
        terminal_asset_floor=config.terminal_asset_floor,
    )
    bound_tolerance = 1e-6
    at_lower = result.decision <= lower + bound_tolerance
    at_upper = result.decision >= upper - bound_tolerance
    free = ~(at_lower | at_upper)

    optimizer_multipliers = getattr(result.optimizer_result, "multipliers", None)
    expected_multiplier_count = (
        equality_jacobian.shape[0] + inequality_jacobian.shape[0] + path_jacobian.shape[0]
    )
    if (
        config.optimizer == "SLSQP"
        and optimizer_multipliers is not None
        and np.asarray(optimizer_multipliers).size == expected_multiplier_count
    ):
        # SciPy reports SLSQP's equality multipliers followed by the full
        # inequality vectors in the same order supplied to minimize. Its
        # convention is grad(f) - J'lambda for both groups.
        multipliers = np.asarray(optimizer_multipliers)
        equality_end = equality_jacobian.shape[0]
        training_end = equality_end + inequality_jacobian.shape[0]
        lagrangian_gradient = (
            gradient
            - equality_jacobian.T @ multipliers[:equality_end]
            - inequality_jacobian.T @ multipliers[equality_end:training_end]
            - path_jacobian.T @ multipliers[training_end:]
        )
    elif np.any(free):
        multiplier_matrix = jacobian[:, free].T
        if active_inequality_count:
            multiplier_lower = np.full(jacobian.shape[0], -np.inf)
            multiplier_upper = np.full(jacobian.shape[0], np.inf)
            # Inequalities are represented as g(z) >= 0. With the +J'lambda
            # convention used here, their KKT multipliers must be nonpositive.
            multiplier_upper[equality_jacobian.shape[0] :] = 0.0
            multiplier_result = lsq_linear(
                multiplier_matrix,
                -gradient[free],
                bounds=(multiplier_lower, multiplier_upper),
                lsq_solver="exact",
                tol=1e-12,
            )
            multipliers = multiplier_result.x
        else:
            multipliers, *_ = np.linalg.lstsq(
                multiplier_matrix,
                -gradient[free],
                rcond=None,
            )
        lagrangian_gradient = gradient + jacobian.T @ multipliers
    else:
        multipliers = np.zeros(jacobian.shape[0])
        lagrangian_gradient = gradient + jacobian.T @ multipliers
    projected_violation = np.array(lagrangian_gradient, copy=True)
    projected_violation[at_lower] = np.minimum(lagrangian_gradient[at_lower], 0.0)
    projected_violation[at_upper] = np.maximum(lagrangian_gradient[at_upper], 0.0)
    return float(np.max(np.abs(projected_violation)))


def independently_integrate_controls(
    result: CollocationResult,
    *,
    samples_per_interval: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate controls on a denser grid, independently of collocation."""

    if samples_per_interval < 1:
        raise ValueError("samples_per_interval must be positive")

    control_times = jnp.asarray(result.time)
    validation_times = jnp.linspace(
        0.0,
        result.params.horizon,
        result.config.intervals * samples_per_interval + 1,
    )
    controls = jnp.column_stack(
        (
            jnp.asarray(result.consumption),
            jnp.asarray(result.hours),
            jnp.asarray(result.training_time),
        )
    )

    def vector_field(
        time: int | float | jax.Array | np.ndarray,
        state: jax.Array,
        args: None,
    ) -> jax.Array:
        del args
        control = jnp.stack(
            tuple(jnp.interp(time, control_times, controls[:, column]) for column in range(3))
        )
        return dynamics(state, control, result.params)

    solution = dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        dfx.Tsit5(),
        t0=0.0,
        t1=result.params.horizon,
        dt0=min(0.1, result.params.horizon / result.config.intervals / 4.0),
        y0=jnp.array(
            [
                result.params.initial_assets,
                jnp.log(result.params.initial_human_capital),
            ]
        ),
        saveat=dfx.SaveAt(ts=validation_times),
        stepsize_controller=dfx.PIDController(rtol=1e-9, atol=1e-11),
        max_steps=100_000,
    )
    states = np.asarray(solution.ys)
    return np.asarray(validation_times), states[:, 0], np.exp(states[:, 1])


def diagnose_path(
    result: CollocationResult, *, independent_integration: bool = True
) -> PathDiagnostics:
    """Run feasibility, KKT, Pontryagin, regime, and integration checks."""

    defects = np.asarray(
        collocation_defects(
            jnp.asarray(result.decision),
            result.params,
            result.config.intervals,
            result.config.scheme,
        )
    )
    path_slacks = np.asarray(
        asset_path_slack(
            jnp.asarray(result.decision),
            result.params,
            result.config.intervals,
            result.config.scheme,
        )
    )
    max_consumption_foc, max_hours_foc, max_investment_foc = _continuous_foc_diagnostics(result)
    if independent_integration:
        validation_substeps = 10
        _, integrated_assets, integrated_human_capital = independently_integrate_controls(
            result,
            samples_per_interval=validation_substeps,
        )
        integrated_assets_at_nodes = integrated_assets[::validation_substeps]
        integrated_human_capital_at_nodes = integrated_human_capital[::validation_substeps]
        asset_error = float(np.max(np.abs(integrated_assets_at_nodes - result.assets)))
        human_capital_relative_error = float(
            np.max(
                np.abs(integrated_human_capital_at_nodes - result.human_capital)
                / np.maximum(1.0, result.human_capital)
            )
        )
        minimum_integrated_assets = float(np.min(integrated_assets))
    else:
        asset_error = float("nan")
        human_capital_relative_error = float("nan")
        minimum_integrated_assets = float("nan")

    labels = regime_labels(result.hours, result.investment_share)
    regime_sequence = _compressed_regime_sequence(labels)
    asset_tolerance = 1e-6 * max(1.0, abs(result.params.initial_assets))
    projected_kkt = _projected_kkt_residual(result)
    asset_scale = max(1.0, float(np.max(np.abs(result.assets))))
    integration_accepted = not independent_integration or (
        asset_error <= 1e-3 * asset_scale
        and human_capital_relative_error <= 1e-4
        and minimum_integrated_assets >= result.params.asset_floor - 1e-4 * asset_scale
    )
    accepted = result.success and projected_kkt <= 1e-4 and integration_accepted
    return PathDiagnostics(
        optimizer_success=result.optimizer_success,
        accepted_success=accepted,
        lifetime_utility=result.lifetime_utility,
        max_equality_violation=result.max_constraint_violation,
        max_unscaled_dynamic_defect=float(np.max(np.abs(defects))),
        projected_kkt_residual=projected_kkt,
        max_consumption_foc_residual=max_consumption_foc,
        max_hours_foc_violation=max_hours_foc,
        max_investment_foc_violation=max_investment_foc,
        independent_asset_error=asset_error,
        independent_human_capital_relative_error=human_capital_relative_error,
        minimum_assets=float(np.min(result.assets)),
        minimum_collocation_path_assets=float(result.params.asset_floor + np.min(path_slacks)),
        minimum_independently_integrated_assets=minimum_integrated_assets,
        minimum_consumption=float(np.min(result.consumption)),
        minimum_leisure=float(np.min(result.leisure)),
        asset_floor_binding_nodes=int(
            np.count_nonzero(result.assets <= result.params.asset_floor + asset_tolerance)
        ),
        normal_regime_order=_normal_regime_order(regime_sequence),
        regime_sequence=" → ".join(regime_sequence),
    )
