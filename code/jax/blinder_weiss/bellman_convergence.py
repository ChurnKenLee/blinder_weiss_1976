"""Convergence-driven refinement for Bellman value and policy functions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from .bellman import (
    BellmanConfig,
    BellmanDiagnostics,
    BellmanSolution,
    _exprel,
    _interpolate_value_jax,
    bellman_state_grids,
    constant_control_transition,
    diagnose_bellman,
    greedy_policy_value_at,
    maximum_feasible_consumption,
    policy_at,
    solve_bellman,
    value_at,
)
from .model import (
    ModelParams,
    benchmark_params,
    bequest_utility,
    flow_utility,
)


@dataclass(frozen=True)
class BellmanConvergenceConfig:
    """Stopping rules and refinement schedule for high-accuracy solutions."""

    max_levels: int = 4
    required_consecutive_passes: int = 2
    value_tolerance: float = 1e-3
    policy_regret_tolerance: float = 1e-4
    monotonicity_tolerance: float = 1e-6
    asset_padding_fraction: float = 0.20
    log_human_capital_padding_fraction: float = 0.10
    control_node_increment: int = 2
    consumption_node_increment: int = 4
    maximum_refinement_starts: int = 4
    control_batch_size: int = 256


@dataclass(frozen=True)
class BellmanConvergenceLevel:
    """One solved refinement level and its common-grid accuracy diagnostics."""

    level: int
    solution: BellmanSolution
    diagnostics: BellmanDiagnostics
    normalized_value_change: float | None
    maximum_policy_regret: float | None
    maximum_optimizer_shortfall: float
    maximum_interpolated_policy_regret: float
    normalized_monotonicity_violation: float
    target_next_states_inside_domain: bool
    passed: bool

    def as_dict(self) -> dict[str, bool | float | int | None]:
        return {
            "level": self.level,
            "periods": self.solution.config.periods,
            "asset_nodes": self.solution.config.asset_nodes,
            "human_capital_nodes": self.solution.config.human_capital_nodes,
            "solve_seconds": self.solution.solve_seconds,
            "lifetime_utility": (
                self.solution.initial_value
                - self.diagnostics.simulation_value_gap
            ),
            "normalized_value_change": self.normalized_value_change,
            "maximum_policy_regret": self.maximum_policy_regret,
            "maximum_optimizer_shortfall": (
                self.maximum_optimizer_shortfall
            ),
            "maximum_interpolated_policy_regret": (
                self.maximum_interpolated_policy_regret
            ),
            "normalized_monotonicity_violation": (
                self.normalized_monotonicity_violation
            ),
            "target_next_states_inside_domain": (
                self.target_next_states_inside_domain
            ),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BellmanConvergenceResult:
    """Result of automatic Bellman refinement."""

    params: ModelParams
    target_config: BellmanConfig
    convergence_config: BellmanConvergenceConfig
    levels: tuple[BellmanConvergenceLevel, ...]
    converged: bool
    stop_reason: str

    @property
    def solution(self) -> BellmanSolution:
        """Return the finest solution that was computed."""

        return self.levels[-1].solution


def _validate_convergence_config(config: BellmanConvergenceConfig) -> None:
    if config.max_levels < 2:
        raise ValueError("max_levels must be at least two")
    if not 1 <= config.required_consecutive_passes < config.max_levels:
        raise ValueError("required_consecutive_passes must lie in [1, max_levels)")
    if min(
        config.value_tolerance,
        config.policy_regret_tolerance,
        config.monotonicity_tolerance,
    ) <= 0.0:
        raise ValueError("convergence tolerances must be positive")
    if config.asset_padding_fraction <= 0.0:
        raise ValueError("asset_padding_fraction must be positive")
    if config.log_human_capital_padding_fraction <= 0.0:
        raise ValueError("log_human_capital_padding_fraction must be positive")
    if config.control_node_increment < 0 or config.consumption_node_increment < 0:
        raise ValueError("control-node increments cannot be negative")
    if config.maximum_refinement_starts < 1:
        raise ValueError("maximum_refinement_starts must be positive")
    if config.control_batch_size < 1:
        raise ValueError("control_batch_size must be positive")


def bellman_refinement_config(
    params: ModelParams,
    target: BellmanConfig,
    convergence: BellmanConvergenceConfig,
    level: int,
) -> BellmanConfig:
    """Construct one deterministic joint domain/time/state/control refinement."""

    if not 0 <= level < convergence.max_levels:
        raise ValueError("level is outside the configured refinement sequence")
    level_number = level + 1
    target_asset_width = target.asset_maximum - target.asset_minimum
    target_log_width = (
        target.log_human_capital_maximum
        - target.log_human_capital_minimum
    )
    asset_floor_distance = target.asset_minimum - params.asset_floor
    asset_minimum = params.asset_floor + asset_floor_distance / (10.0**level_number)
    asset_maximum = (
        target.asset_maximum
        + target_asset_width
        * convergence.asset_padding_fraction
        * level_number
    )
    asset_nodes = 1 + (target.asset_nodes - 1) * level_number
    asset_grid_curvature = target.asset_grid_curvature
    if asset_nodes > 2:
        target_floor_distance = target.asset_minimum - asset_minimum
        relative_first_interval = target_floor_distance / (
            asset_maximum - asset_minimum
        )
        required_curvature = np.log(relative_first_interval) / np.log(
            1.0 / (asset_nodes - 1)
        )
        asset_grid_curvature = max(
            asset_grid_curvature,
            float(required_curvature),
        )
    log_padding = (
        target_log_width
        * convergence.log_human_capital_padding_fraction
        * level_number
    )
    refinement_starts = min(
        2 + level,
        convergence.maximum_refinement_starts,
    )
    candidate_count = (
        (target.hours_nodes + convergence.control_node_increment * level)
        * (target.investment_nodes + convergence.control_node_increment * level)
        * (
            target.consumption_nodes
            + convergence.consumption_node_increment * level
        )
    )
    return replace(
        target,
        periods=target.periods * 2**level,
        asset_nodes=asset_nodes,
        human_capital_nodes=(
            1 + (target.human_capital_nodes - 1) * level_number
        ),
        asset_minimum=asset_minimum,
        asset_maximum=asset_maximum,
        asset_grid_curvature=asset_grid_curvature,
        log_human_capital_minimum=(
            target.log_human_capital_minimum - log_padding
        ),
        log_human_capital_maximum=(
            target.log_human_capital_maximum + log_padding
        ),
        hours_nodes=(
            target.hours_nodes + convergence.control_node_increment * level
        ),
        investment_nodes=(
            target.investment_nodes
            + convergence.control_node_increment * level
        ),
        consumption_nodes=(
            target.consumption_nodes
            + convergence.consumption_node_increment * level
        ),
        path_checkpoints=target.path_checkpoints * 2**level,
        refinement_steps=target.refinement_steps * level_number,
        refinement_starts=min(refinement_starts, candidate_count),
        control_batch_size=convergence.control_batch_size,
        neighbor_policy_sweeps=target.neighbor_policy_sweeps + level,
    )


def _insert_midpoints(grid: np.ndarray) -> np.ndarray:
    result = np.empty(2 * grid.size - 1, dtype=np.float64)
    result[::2] = grid
    result[1::2] = 0.5 * (grid[:-1] + grid[1:])
    return result


def _validation_states(
    target: BellmanConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_assets, target_log_human_capital = bellman_state_grids(target)
    assets = _insert_midpoints(target_assets)
    log_human_capital = _insert_midpoints(target_log_human_capital)
    asset_mesh, log_human_capital_mesh = np.meshgrid(
        assets,
        log_human_capital,
        indexing="ij",
    )
    return (
        assets,
        log_human_capital,
        asset_mesh,
        log_human_capital_mesh,
    )


def _common_period_indices(
    solution: BellmanSolution,
    target: BellmanConfig,
    *,
    include_terminal: bool,
) -> np.ndarray:
    if solution.config.periods % target.periods:
        raise ValueError("refined periods must be a multiple of target periods")
    ratio = solution.config.periods // target.periods
    count = target.periods + int(include_terminal)
    return np.arange(count, dtype=np.int64) * ratio


def _values_on_target(
    solution: BellmanSolution,
    target: BellmanConfig,
    asset_mesh: np.ndarray,
    log_human_capital_mesh: np.ndarray,
) -> np.ndarray:
    indices = _common_period_indices(
        solution,
        target,
        include_terminal=True,
    )
    human_capital = np.exp(log_human_capital_mesh)
    return np.stack(
        [
            value_at(
                solution,
                int(period),
                asset_mesh,
                human_capital,
            )
            for period in indices
        ]
    )


def _policy_value(
    solution: BellmanSolution,
    period: int,
    asset_mesh: np.ndarray,
    log_human_capital_mesh: np.ndarray,
    controls: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    consumption, hours, training_time = (jnp.asarray(value) for value in controls)
    states = jnp.stack(
        (
            jnp.asarray(asset_mesh),
            jnp.asarray(log_human_capital_mesh),
        ),
        axis=-1,
    )
    step = solution.params.horizon / solution.config.periods
    consumption_capacity = maximum_feasible_consumption(
        states[..., 0],
        states[..., 1],
        hours,
        training_time,
        solution.params,
        step,
        solution.config.asset_minimum,
        solution.config.path_checkpoints,
        method=solution.config.asset_feasibility,
    )
    next_states = constant_control_transition(
        states,
        jnp.stack((consumption, hours, training_time), axis=-1),
        solution.params,
        step,
    )
    if period == solution.config.periods - 1:
        continuation = bequest_utility(
            jnp.maximum(next_states[..., 0], solution.asset_grid[0]),
            solution.params,
        )
    else:
        continuation = _interpolate_value_jax(
            jnp.asarray(solution.values[period + 1]),
            jnp.asarray(solution.asset_grid),
            jnp.asarray(solution.log_human_capital_grid),
            next_states[..., 0],
            next_states[..., 1],
            solution.config,
        )
    objective = (
        step
        * _exprel(jnp.asarray(-solution.params.rho * step))
        * flow_utility(consumption, hours, solution.params)
        + np.exp(-solution.params.rho * step) * continuation
    )
    control_tolerance = 1e-10
    feasible = (
        (
            consumption
            >= solution.config.consumption_floor - control_tolerance
        )
        & (consumption <= consumption_capacity + 1e-10)
        & (hours >= -control_tolerance)
        & (
            hours
            <= 1.0 - solution.config.leisure_floor + control_tolerance
        )
        & (training_time >= -control_tolerance)
        & (training_time <= hours + control_tolerance)
        & (next_states[..., 0] >= solution.asset_grid[0] - 1e-10)
        & (next_states[..., 0] <= solution.asset_grid[-1] + 1e-10)
        & (
            next_states[..., 1]
            >= solution.log_human_capital_grid[0] - 1e-10
        )
        & (
            next_states[..., 1]
            <= solution.log_human_capital_grid[-1] + 1e-10
        )
    )
    return (
        np.asarray(jnp.where(feasible, objective, -jnp.inf)),
        np.asarray(next_states),
    )


def _project_policy_controls(
    solution: BellmanSolution,
    asset_mesh: np.ndarray,
    log_human_capital_mesh: np.ndarray,
    controls: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project inherited/interpolated controls into the finer feasible set."""

    consumption, hours, training_time = (
        jnp.asarray(value) for value in controls
    )
    hours = jnp.clip(hours, 0.0, 1.0 - solution.config.leisure_floor)
    training_time = jnp.clip(training_time, 0.0, hours)
    step = solution.params.horizon / solution.config.periods
    productivity = solution.params.human_capital_productivity
    if productivity > 0.0:
        training_lower = (
            (
                solution.log_human_capital_grid[0]
                - jnp.asarray(log_human_capital_mesh)
            )
            / step
            + solution.params.human_capital_depreciation
        ) / productivity
        training_upper = (
            (
                solution.log_human_capital_grid[-1]
                - jnp.asarray(log_human_capital_mesh)
            )
            / step
            + solution.params.human_capital_depreciation
        ) / productivity
        training_time = jnp.clip(
            training_time,
            jnp.maximum(0.0, training_lower),
            jnp.minimum(hours, training_upper),
        )
    consumption_capacity = maximum_feasible_consumption(
        jnp.asarray(asset_mesh),
        jnp.asarray(log_human_capital_mesh),
        hours,
        training_time,
        solution.params,
        step,
        solution.config.asset_minimum,
        solution.config.path_checkpoints,
        method=solution.config.asset_feasibility,
    )
    states = jnp.stack(
        (
            jnp.asarray(asset_mesh),
            jnp.asarray(log_human_capital_mesh),
        ),
        axis=-1,
    )
    zero_consumption_controls = jnp.stack(
        (
            jnp.zeros_like(hours),
            hours,
            training_time,
        ),
        axis=-1,
    )
    next_without_consumption = constant_control_transition(
        states,
        zero_consumption_controls,
        solution.params,
        step,
    )
    consumption_factor = step * _exprel(
        jnp.asarray(solution.params.interest_rate * step)
    )
    domain_consumption_floor = (
        next_without_consumption[..., 0] - solution.asset_grid[-1]
    ) / consumption_factor
    lower = jnp.maximum(
        solution.config.consumption_floor,
        domain_consumption_floor,
    )
    projected_consumption = jnp.clip(
        consumption,
        lower,
        consumption_capacity,
    )
    return (
        np.asarray(projected_consumption),
        np.asarray(hours),
        np.asarray(training_time),
    )


def _maximum_normalized_loss(
    best_values: np.ndarray,
    candidate_values: np.ndarray,
) -> float:
    if not np.all(np.isfinite(candidate_values)):
        return float("inf")
    scale = np.maximum(1.0, np.abs(best_values))
    return float(
        np.max(np.maximum(best_values - candidate_values, 0.0) / scale)
    )


def _policy_accuracy_on_target(
    current: BellmanSolution,
    previous_policy_history: (
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ),
    target: BellmanConfig,
    asset_mesh: np.ndarray,
    log_human_capital_mesh: np.ndarray,
) -> tuple[
    float | None,
    float,
    float,
    bool,
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    current_indices = _common_period_indices(
        current,
        target,
        include_terminal=False,
    )
    human_capital = np.exp(log_human_capital_mesh)
    maximum_previous_regret = 0.0
    maximum_optimizer_shortfall = 0.0
    maximum_interpolated_regret = 0.0
    target_inside_domain = True
    current_consumption_history: list[np.ndarray] = []
    current_hours_history: list[np.ndarray] = []
    current_training_history: list[np.ndarray] = []

    for common_period, current_period in enumerate(current_indices):
        greedy_value, greedy_consumption, greedy_hours, greedy_training = (
            greedy_policy_value_at(
                current,
                int(current_period),
                asset_mesh,
                human_capital,
            )
        )
        interpolated_controls = policy_at(
            current,
            int(current_period),
            asset_mesh,
            human_capital,
        )
        interpolated_controls = _project_policy_controls(
            current,
            asset_mesh,
            log_human_capital_mesh,
            interpolated_controls,
        )
        interpolated_value, _ = _policy_value(
            current,
            int(current_period),
            asset_mesh,
            log_human_capital_mesh,
            interpolated_controls,
        )
        greedy_controls = (
            greedy_consumption,
            greedy_hours,
            greedy_training,
        )
        current_consumption_history.append(greedy_consumption)
        current_hours_history.append(greedy_hours)
        current_training_history.append(greedy_training)
        recomputed_greedy_value, next_states = _policy_value(
            current,
            int(current_period),
            asset_mesh,
            log_human_capital_mesh,
            greedy_controls,
        )
        candidate_values = [np.asarray(greedy_value), recomputed_greedy_value]
        previous_value: np.ndarray | None = None
        if previous_policy_history is not None:
            previous_controls = (
                previous_policy_history[0][common_period],
                previous_policy_history[1][common_period],
                previous_policy_history[2][common_period],
            )
            previous_controls = _project_policy_controls(
                current,
                asset_mesh,
                log_human_capital_mesh,
                previous_controls,
            )
            previous_value, _ = _policy_value(
                current,
                int(current_period),
                asset_mesh,
                log_human_capital_mesh,
                previous_controls,
            )
            candidate_values.append(previous_value)
        best_value = np.maximum.reduce(candidate_values)
        maximum_optimizer_shortfall = max(
            maximum_optimizer_shortfall,
            _maximum_normalized_loss(best_value, np.asarray(greedy_value)),
        )
        maximum_interpolated_regret = max(
            maximum_interpolated_regret,
            _maximum_normalized_loss(best_value, interpolated_value),
        )
        if previous_value is not None:
            maximum_previous_regret = max(
                maximum_previous_regret,
                _maximum_normalized_loss(best_value, previous_value),
            )
        target_inside_domain = target_inside_domain and bool(
            np.all(next_states[..., 0] >= current.asset_grid[0] - 1e-9)
            and np.all(next_states[..., 0] <= current.asset_grid[-1] + 1e-9)
            and np.all(
                next_states[..., 1]
                >= current.log_human_capital_grid[0] - 1e-9
            )
            and np.all(
                next_states[..., 1]
                <= current.log_human_capital_grid[-1] + 1e-9
            )
        )

    return (
        (
            None
            if previous_policy_history is None
            else maximum_previous_regret
        ),
        maximum_optimizer_shortfall,
        maximum_interpolated_regret,
        target_inside_domain,
        (
            np.stack(current_consumption_history),
            np.stack(current_hours_history),
            np.stack(current_training_history),
        ),
    )


def _normalized_monotonicity_violation(values: np.ndarray) -> float:
    minimum_asset_difference = float(np.min(np.diff(values, axis=1)))
    minimum_human_capital_difference = float(np.min(np.diff(values, axis=2)))
    absolute_violation = max(
        0.0,
        -minimum_asset_difference,
        -minimum_human_capital_difference,
    )
    return absolute_violation / max(1.0, float(np.max(np.abs(values))))


def solve_bellman_converged(
    params: ModelParams | None = None,
    config: BellmanConfig | None = None,
    convergence_config: BellmanConvergenceConfig | None = None,
) -> BellmanConvergenceResult:
    """Jointly refine the Bellman domain, time, state, and control grids."""

    params = benchmark_params() if params is None else params
    target = BellmanConfig() if config is None else config
    convergence = (
        BellmanConvergenceConfig()
        if convergence_config is None
        else convergence_config
    )
    _validate_convergence_config(convergence)
    _, _, asset_mesh, log_human_capital_mesh = _validation_states(target)

    levels: list[BellmanConvergenceLevel] = []
    previous_values: np.ndarray | None = None
    previous_policy_history: (
        tuple[np.ndarray, np.ndarray, np.ndarray] | None
    ) = None
    consecutive_passes = 0
    converged = False

    for level in range(convergence.max_levels):
        level_config = bellman_refinement_config(
            params,
            target,
            convergence,
            level,
        )
        solution = solve_bellman(params, level_config)
        diagnostics = diagnose_bellman(solution)
        target_values = _values_on_target(
            solution,
            target,
            asset_mesh,
            log_human_capital_mesh,
        )
        normalized_value_change = (
            None
            if previous_values is None
            else float(
                np.max(
                    np.abs(target_values - previous_values)
                    / np.maximum(1.0, np.abs(target_values))
                )
            )
        )
        (
            maximum_policy_regret,
            maximum_optimizer_shortfall,
            maximum_interpolated_policy_regret,
            target_next_states_inside_domain,
            current_policy_history,
        ) = _policy_accuracy_on_target(
            solution,
            previous_policy_history,
            target,
            asset_mesh,
            log_human_capital_mesh,
        )
        monotonicity_violation = _normalized_monotonicity_violation(
            target_values
        )
        passed = bool(
            normalized_value_change is not None
            and maximum_policy_regret is not None
            and normalized_value_change <= convergence.value_tolerance
            and maximum_policy_regret <= convergence.policy_regret_tolerance
            and maximum_optimizer_shortfall
            <= convergence.policy_regret_tolerance
            and monotonicity_violation
            <= convergence.monotonicity_tolerance
            and target_next_states_inside_domain
            and diagnostics.accepted_node_solution
        )
        consecutive_passes = consecutive_passes + 1 if passed else 0
        levels.append(
            BellmanConvergenceLevel(
                level=level,
                solution=solution,
                diagnostics=diagnostics,
                normalized_value_change=normalized_value_change,
                maximum_policy_regret=maximum_policy_regret,
                maximum_optimizer_shortfall=maximum_optimizer_shortfall,
                maximum_interpolated_policy_regret=(
                    maximum_interpolated_policy_regret
                ),
                normalized_monotonicity_violation=monotonicity_violation,
                target_next_states_inside_domain=(
                    target_next_states_inside_domain
                ),
                passed=passed,
            )
        )
        if consecutive_passes >= convergence.required_consecutive_passes:
            converged = True
            break
        previous_values = target_values
        previous_policy_history = current_policy_history

    if converged:
        stop_reason = (
            "research tolerances satisfied for "
            f"{convergence.required_consecutive_passes} consecutive refinements"
        )
    else:
        stop_reason = (
            "maximum refinement levels reached before all research tolerances "
            "were satisfied"
        )
    return BellmanConvergenceResult(
        params=params,
        target_config=target,
        convergence_config=convergence,
        levels=tuple(levels),
        converged=converged,
        stop_reason=stop_reason,
    )
