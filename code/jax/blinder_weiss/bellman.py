"""Semi-Lagrangian Bellman solver for feedback policy functions.

The direct-collocation solver in :mod:`blinder_weiss.solver` computes one
open-loop lifecycle.  This module instead works backward over time and solves
the control problem at a collection of asset and log-human-capital states.  It
therefore returns approximations to ``V(t, A, K)`` and to the feedback policies
``c(t, A, K)``, ``h(t, A, K)``, and ``q(t, A, K)``.

No finite differences of the value function are taken.  The continuation
value is evaluated by monotone bilinear interpolation, and the state constraint
is imposed through the feasible consumption set over each semi-Lagrangian
step.  Controls are constant within a step, which makes the state transition
available in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .model import (
    ModelParams,
    benchmark_params,
    bequest_utility,
    effective_earnings_share,
    flow_utility,
)


@dataclass(frozen=True)
class BellmanConfig:
    """Discretization and optimization settings for backward induction.

    ``asset_minimum`` is a numerical state-domain floor and must lie strictly
    above the economic borrowing limit when terminal utility is singular at
    zero.  It can be made very small; concentrating the curved asset grid near
    this value resolves the state-constraint region without a large uniform
    grid. ``compute_platform="auto"`` uses a CUDA device when JAX can see one
    and otherwise uses CPU. Select ``"gpu"`` to require CUDA rather than
    allowing a silent CPU fallback; ``device_index`` selects among devices on
    the chosen platform.
    """

    periods: int = 70
    asset_nodes: int = 31
    human_capital_nodes: int = 25
    asset_minimum: float = 1e-4
    asset_maximum: float = 35.0
    asset_grid_curvature: float = 2.0
    log_human_capital_minimum: float = -2.0
    log_human_capital_maximum: float = 2.25
    hours_nodes: int = 11
    investment_nodes: int = 11
    consumption_nodes: int = 15
    consumption_fraction_minimum: float = 0.02
    consumption_floor: float = 1e-8
    leisure_floor: float = 1e-5
    path_checkpoints: int = 4
    refinement_steps: int = 12
    refinement_learning_rate: float = 0.025
    neighbor_policy_sweeps: int = 2
    compute_platform: Literal["auto", "cpu", "gpu"] = "auto"
    device_index: int = 0


@dataclass(frozen=True)
class BellmanSolution:
    """Value and feedback-policy arrays on the state-age collocation nodes."""

    params: ModelParams
    config: BellmanConfig
    time: np.ndarray
    asset_grid: np.ndarray
    log_human_capital_grid: np.ndarray
    values: np.ndarray
    consumption_policy: np.ndarray
    hours_policy: np.ndarray
    training_time_policy: np.ndarray
    solve_seconds: float
    backend: str
    device: str

    @property
    def human_capital_grid(self) -> np.ndarray:
        return np.exp(self.log_human_capital_grid)

    @property
    def investment_share_policy(self) -> np.ndarray:
        return np.divide(
            self.training_time_policy,
            self.hours_policy,
            out=np.zeros_like(self.training_time_policy),
            where=self.hours_policy > 1e-12,
        )

    @property
    def initial_value(self) -> float:
        return float(
            _interpolate_numpy(
                self.values[0],
                self.asset_grid,
                self.log_human_capital_grid,
                self.params.initial_assets,
                np.log(self.params.initial_human_capital),
            )
        )


@dataclass(frozen=True)
class BellmanSimulation:
    """A lifecycle generated from a :class:`BellmanSolution` policy."""

    solution: BellmanSolution
    policy_method: Literal["greedy", "interpolate"]
    time: np.ndarray
    assets: np.ndarray
    log_human_capital: np.ndarray
    human_capital: np.ndarray
    consumption: np.ndarray
    hours: np.ndarray
    training_time: np.ndarray
    lifetime_utility: float
    initial_value: float
    value_gap: float
    stayed_in_domain: bool
    minimum_assets: float

    @property
    def investment_share(self) -> np.ndarray:
        return np.divide(
            self.training_time,
            self.hours,
            out=np.zeros_like(self.training_time),
            where=self.hours > 1e-12,
        )

    @property
    def leisure(self) -> np.ndarray:
        return 1.0 - self.hours


@dataclass(frozen=True)
class BellmanDiagnostics:
    """Numerical checks for node policies and an interpolated simulation."""

    accepted_node_solution: bool
    all_values_finite: bool
    maximum_node_bellman_residual: float
    minimum_consumption_capacity_slack: float
    minimum_training_time: float
    minimum_training_slack: float
    minimum_hours: float
    maximum_hours: float
    maximum_value_monotonicity_violation: float
    next_states_inside_domain: bool
    simulation_stayed_in_domain: bool
    simulation_minimum_assets: float
    simulation_value_gap: float

    def as_dict(self) -> dict[str, bool | float]:
        return {
            "accepted_node_solution": self.accepted_node_solution,
            "all_values_finite": self.all_values_finite,
            "maximum_node_bellman_residual": self.maximum_node_bellman_residual,
            "minimum_consumption_capacity_slack": self.minimum_consumption_capacity_slack,
            "minimum_training_time": self.minimum_training_time,
            "minimum_training_slack": self.minimum_training_slack,
            "minimum_hours": self.minimum_hours,
            "maximum_hours": self.maximum_hours,
            "maximum_value_monotonicity_violation": (self.maximum_value_monotonicity_violation),
            "next_states_inside_domain": self.next_states_inside_domain,
            "simulation_stayed_in_domain": self.simulation_stayed_in_domain,
            "simulation_minimum_assets": self.simulation_minimum_assets,
            "simulation_value_gap": self.simulation_value_gap,
        }


def _validate_config(params: ModelParams, config: BellmanConfig) -> None:
    if params.horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if config.periods < 1:
        raise ValueError("periods must be positive")
    if config.asset_nodes < 2 or config.human_capital_nodes < 2:
        raise ValueError("each state dimension requires at least two nodes")
    if min(config.hours_nodes, config.investment_nodes, config.consumption_nodes) < 2:
        raise ValueError("each control dimension requires at least two nodes")
    if config.asset_minimum <= params.asset_floor:
        raise ValueError("asset_minimum must lie strictly above the economic asset_floor")
    if config.asset_maximum <= config.asset_minimum:
        raise ValueError("asset_maximum must exceed asset_minimum")
    if not config.asset_minimum <= params.initial_assets <= config.asset_maximum:
        raise ValueError("initial_assets must lie inside the Bellman asset domain")
    initial_log_human_capital = np.log(params.initial_human_capital)
    if not (
        config.log_human_capital_minimum
        <= initial_log_human_capital
        <= config.log_human_capital_maximum
    ):
        raise ValueError("initial human capital must lie inside the Bellman state domain")
    if config.asset_grid_curvature <= 0.0:
        raise ValueError("asset_grid_curvature must be positive")
    if not 0.0 < config.consumption_fraction_minimum < 1.0:
        raise ValueError("consumption_fraction_minimum must lie in (0, 1)")
    if config.consumption_floor <= 0.0:
        raise ValueError("consumption_floor must be positive")
    if not 0.0 < config.leisure_floor < 1.0:
        raise ValueError("leisure_floor must lie in (0, 1)")
    if config.path_checkpoints < 1:
        raise ValueError("path_checkpoints must be positive")
    if config.refinement_steps < 0:
        raise ValueError("refinement_steps cannot be negative")
    if config.refinement_learning_rate <= 0.0:
        raise ValueError("refinement_learning_rate must be positive")
    if config.neighbor_policy_sweeps < 0:
        raise ValueError("neighbor_policy_sweeps cannot be negative")
    if config.compute_platform not in {"auto", "cpu", "gpu"}:
        raise ValueError("compute_platform must be 'auto', 'cpu', or 'gpu'")
    if config.device_index < 0:
        raise ValueError("device_index cannot be negative")


def _select_compute_device(config: BellmanConfig) -> Any:
    """Resolve the requested JAX device, failing clearly for unavailable CUDA."""

    backend = None if config.compute_platform == "auto" else config.compute_platform
    gpu_error = (
        "compute_platform='gpu' was requested, but JAX could not initialize "
        "a CUDA device. Check the NVIDIA driver, launch the process with GPU "
        "access, and install the jax[cuda13] dependency."
    )
    try:
        devices = jax.devices(backend)
    except RuntimeError as error:
        if config.compute_platform == "gpu":
            raise RuntimeError(gpu_error) from error
        raise

    if config.compute_platform == "gpu":
        devices = [device for device in devices if device.platform == "gpu"]
        if not devices:
            raise RuntimeError(gpu_error)

    if config.device_index >= len(devices):
        platform = devices[0].platform if devices else config.compute_platform
        raise ValueError(
            f"device_index={config.device_index} is unavailable on the {platform!r} "
            f"platform; JAX found {len(devices)} device(s)"
        )
    return devices[config.device_index]


def bellman_state_grids(
    config: BellmanConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the curved asset grid and uniform log-human-capital grid."""

    unit_grid = np.linspace(0.0, 1.0, config.asset_nodes)
    assets = config.asset_minimum + (config.asset_maximum - config.asset_minimum) * np.power(
        unit_grid, config.asset_grid_curvature
    )
    log_human_capital = np.linspace(
        config.log_human_capital_minimum,
        config.log_human_capital_maximum,
        config.human_capital_nodes,
    )
    return assets.astype(np.float64), log_human_capital.astype(np.float64)


def _exprel(value: ArrayLike) -> Array:
    """Stable evaluation of ``expm1(value) / value`` near zero."""

    absolute_value = jnp.abs(value)
    safe_value = jnp.where(absolute_value > 1e-7, value, 1.0)
    quotient = jnp.expm1(value) / safe_value
    series = 1.0 + value / 2.0 + value**2 / 6.0 + value**3 / 24.0
    return jnp.where(absolute_value > 1e-7, quotient, series)


def _transition_factors(
    growth_rate: Array, interest_rate: float, duration: Array
) -> tuple[Array, Array, Array]:
    """Factors for assets, labor income, and constant consumption."""

    asset_factor = jnp.exp(interest_rate * duration)
    income_factor = (
        asset_factor * duration * _exprel((growth_rate[..., None] - interest_rate) * duration)
    )
    consumption_factor = duration * _exprel(interest_rate * duration)
    return asset_factor, income_factor, consumption_factor


def constant_control_transition(
    state: Array,
    control: Array,
    params: ModelParams,
    step: float,
) -> Array:
    """Exact one-step state transition for controls held constant.

    The returned state is ``[assets, log_human_capital]``.  The formula
    integrates both exponentially changing human capital and interest on
    assets, so time-discretization error comes from the piecewise-constant
    policy rather than an Euler state update.
    """

    assets, log_human_capital = state[..., 0], state[..., 1]
    consumption, hours, training_time = control[..., 0], control[..., 1], control[..., 2]
    human_capital_growth = (
        params.human_capital_productivity * training_time - params.human_capital_depreciation
    )
    duration = jnp.asarray(step)
    asset_factor = jnp.exp(params.interest_rate * duration)
    income_factor = (
        asset_factor * duration * _exprel((human_capital_growth - params.interest_rate) * duration)
    )
    consumption_factor = duration * _exprel(params.interest_rate * duration)
    earnings_capacity = effective_earnings_share(hours, training_time) * jnp.exp(log_human_capital)
    next_assets = (
        asset_factor * assets + income_factor * earnings_capacity - consumption_factor * consumption
    )
    next_log_human_capital = log_human_capital + human_capital_growth * duration
    return jnp.stack((next_assets, next_log_human_capital), axis=-1)


def maximum_feasible_consumption(
    assets: Array,
    log_human_capital: Array,
    hours: Array,
    training_time: Array,
    params: ModelParams,
    step: float,
    asset_minimum: ArrayLike,
    checkpoints: int,
) -> Array:
    """Largest constant consumption respecting the asset floor at checkpoints."""

    checkpoint_times = jnp.linspace(step / checkpoints, step, checkpoints)
    human_capital_growth = (
        params.human_capital_productivity * training_time - params.human_capital_depreciation
    )
    asset_factor, income_factor, consumption_factor = _transition_factors(
        human_capital_growth, params.interest_rate, checkpoint_times
    )
    earnings_capacity = effective_earnings_share(hours, training_time) * jnp.exp(log_human_capital)
    assets_without_consumption = (
        assets[..., None] * asset_factor + earnings_capacity[..., None] * income_factor
    )
    checkpoint_bounds = (assets_without_consumption - asset_minimum) / consumption_factor
    return jnp.min(checkpoint_bounds, axis=-1)


def _interpolate_jax(
    values: Array,
    asset_grid: Array,
    log_human_capital_grid: Array,
    assets: Array,
    log_human_capital: Array,
) -> Array:
    """Monotone bilinear interpolation on a possibly curved asset grid."""

    bounded_assets = jnp.clip(assets, asset_grid[0], asset_grid[-1])
    bounded_log_human_capital = jnp.clip(
        log_human_capital,
        log_human_capital_grid[0],
        log_human_capital_grid[-1],
    )
    asset_index = jnp.clip(
        jnp.searchsorted(asset_grid, bounded_assets, side="right") - 1,
        0,
        asset_grid.size - 2,
    )
    human_capital_index = jnp.clip(
        jnp.searchsorted(
            log_human_capital_grid,
            bounded_log_human_capital,
            side="right",
        )
        - 1,
        0,
        log_human_capital_grid.size - 2,
    )
    asset_lower = asset_grid[asset_index]
    asset_upper = asset_grid[asset_index + 1]
    human_capital_lower = log_human_capital_grid[human_capital_index]
    human_capital_upper = log_human_capital_grid[human_capital_index + 1]
    asset_weight = (bounded_assets - asset_lower) / (asset_upper - asset_lower)
    human_capital_weight = (bounded_log_human_capital - human_capital_lower) / (
        human_capital_upper - human_capital_lower
    )

    lower_lower = values[asset_index, human_capital_index]
    upper_lower = values[asset_index + 1, human_capital_index]
    lower_upper = values[asset_index, human_capital_index + 1]
    upper_upper = values[asset_index + 1, human_capital_index + 1]
    lower_value = lower_lower + asset_weight * (upper_lower - lower_lower)
    upper_value = lower_upper + asset_weight * (upper_upper - lower_upper)
    return lower_value + human_capital_weight * (upper_value - lower_value)


def _interpolate_numpy(
    values: np.ndarray,
    asset_grid: np.ndarray,
    log_human_capital_grid: np.ndarray,
    assets: float | np.ndarray,
    log_human_capital: float | np.ndarray,
) -> np.ndarray:
    return np.asarray(
        _interpolate_jax(
            jnp.asarray(values),
            jnp.asarray(asset_grid),
            jnp.asarray(log_human_capital_grid),
            jnp.asarray(assets),
            jnp.asarray(log_human_capital),
        )
    )


def _chebyshev_unit_nodes(nodes: int) -> Array:
    angles = jnp.linspace(0.0, jnp.pi, nodes)
    return 0.5 * (1.0 - jnp.cos(angles))


def _make_control_optimizer(
    params: ModelParams,
    config: BellmanConfig,
    asset_grid: Array,
    log_human_capital_grid: Array,
    *,
    neighbor_shape: tuple[int, int] | None = None,
):
    """Create a JIT-compatible optimizer for an arbitrary batch of states."""

    step = params.horizon / config.periods
    hours_grid = (1.0 - config.leisure_floor) * _chebyshev_unit_nodes(config.hours_nodes)
    investment_grid = _chebyshev_unit_nodes(config.investment_nodes)
    candidate_hours, candidate_investment = jnp.meshgrid(hours_grid, investment_grid, indexing="ij")
    candidate_hours = candidate_hours.ravel()
    candidate_investment = candidate_investment.ravel()
    candidate_training = candidate_hours * candidate_investment
    consumption_fractions = config.consumption_fraction_minimum + (
        1.0 - config.consumption_fraction_minimum
    ) * _chebyshev_unit_nodes(config.consumption_nodes)
    beta = jnp.exp(-params.rho * step)
    flow_discount = step * _exprel(-params.rho * step)
    asset_minimum = asset_grid[0]
    asset_maximum = asset_grid[-1]
    log_human_capital_minimum = log_human_capital_grid[0]
    log_human_capital_maximum = log_human_capital_grid[-1]

    def evaluate_controls(
        controls: Array,
        flat_states: Array,
        continuation_values: Array,
    ) -> tuple[Array, Array, Array, Array]:
        flat_assets = flat_states[:, 0]
        flat_log_human_capital = flat_states[:, 1]
        hours = controls[:, 0]
        investment_share = controls[:, 1]
        consumption_fraction = controls[:, 2]
        training_time = hours * investment_share
        consumption_capacity = maximum_feasible_consumption(
            flat_assets,
            flat_log_human_capital,
            hours,
            training_time,
            params,
            step,
            asset_minimum,
            config.path_checkpoints,
        )
        consumption_span = jnp.maximum(consumption_capacity - config.consumption_floor, 0.0)
        consumption = config.consumption_floor + consumption_fraction * consumption_span
        state_controls = jnp.column_stack((consumption, hours, training_time))
        next_states = constant_control_transition(flat_states, state_controls, params, step)
        continuation = _interpolate_jax(
            continuation_values,
            asset_grid,
            log_human_capital_grid,
            next_states[:, 0],
            next_states[:, 1],
        )
        objective = flow_discount * flow_utility(consumption, hours, params) + beta * continuation
        feasible = (
            (consumption_capacity >= config.consumption_floor)
            & (next_states[:, 0] >= asset_minimum - 1e-10)
            & (next_states[:, 0] <= asset_maximum + 1e-10)
            & (next_states[:, 1] >= log_human_capital_minimum - 1e-10)
            & (next_states[:, 1] <= log_human_capital_maximum + 1e-10)
        )
        return objective, feasible, consumption, training_time

    def smooth_objective(
        controls: Array,
        flat_states: Array,
        continuation_values: Array,
    ) -> Array:
        """Penalized separable objective used only for autodiff refinement."""

        flat_assets = flat_states[:, 0]
        flat_log_human_capital = flat_states[:, 1]
        hours = controls[:, 0]
        investment_share = controls[:, 1]
        consumption_fraction = controls[:, 2]
        training_time = hours * investment_share
        consumption_capacity = maximum_feasible_consumption(
            flat_assets,
            flat_log_human_capital,
            hours,
            training_time,
            params,
            step,
            asset_minimum,
            config.path_checkpoints,
        )
        consumption_span = jnp.maximum(consumption_capacity - config.consumption_floor, 0.0)
        consumption = config.consumption_floor + consumption_fraction * consumption_span
        state_controls = jnp.column_stack((consumption, hours, training_time))
        next_states = constant_control_transition(flat_states, state_controls, params, step)
        continuation = _interpolate_jax(
            continuation_values,
            asset_grid,
            log_human_capital_grid,
            next_states[:, 0],
            next_states[:, 1],
        )
        objective = flow_discount * flow_utility(consumption, hours, params) + beta * continuation
        scale_assets = jnp.maximum(asset_maximum - asset_minimum, 1.0)
        scale_human_capital = jnp.maximum(
            log_human_capital_maximum - log_human_capital_minimum, 1.0
        )
        violations = (
            jax.nn.relu(asset_minimum - next_states[:, 0]) / scale_assets
            + jax.nn.relu(next_states[:, 0] - asset_maximum) / scale_assets
            + jax.nn.relu(log_human_capital_minimum - next_states[:, 1]) / scale_human_capital
            + jax.nn.relu(next_states[:, 1] - log_human_capital_maximum) / scale_human_capital
            + jax.nn.relu(config.consumption_floor - consumption_capacity)
        )
        return jnp.sum(objective - 1e6 * violations**2)

    objective_gradient = jax.grad(smooth_objective, argnums=0)

    def refine_controls(
        initial_controls: Array,
        initial_values: Array,
        flat_states: Array,
        continuation_values: Array,
    ) -> tuple[Array, Array]:
        first_moment = jnp.zeros_like(initial_controls)
        second_moment = jnp.zeros_like(initial_controls)

        def body(_, carry):
            controls, moment_one, moment_two, best_controls, best_values = carry
            gradient = objective_gradient(controls, flat_states, continuation_values)
            moment_one = 0.9 * moment_one + 0.1 * gradient
            moment_two = 0.999 * moment_two + 0.001 * gradient**2
            proposal = controls + config.refinement_learning_rate * moment_one / (
                jnp.sqrt(moment_two) + 1e-8
            )
            proposal = proposal.at[:, 0].set(
                jnp.clip(proposal[:, 0], 0.0, 1.0 - config.leisure_floor)
            )
            proposal = proposal.at[:, 1].set(jnp.clip(proposal[:, 1], 0.0, 1.0))
            proposal = proposal.at[:, 2].set(
                jnp.clip(
                    proposal[:, 2],
                    config.consumption_fraction_minimum,
                    1.0,
                )
            )
            proposal_values, proposal_feasible, _, _ = evaluate_controls(
                proposal, flat_states, continuation_values
            )
            current_values, current_feasible, _, _ = evaluate_controls(
                controls, flat_states, continuation_values
            )
            accept_current = proposal_feasible & (proposal_values >= current_values - 1e-12)
            controls = jnp.where(accept_current[:, None], proposal, controls)
            improved = proposal_feasible & (proposal_values > best_values)
            best_controls = jnp.where(improved[:, None], proposal, best_controls)
            best_values = jnp.where(improved, proposal_values, best_values)
            controls = jnp.where(current_feasible[:, None], controls, best_controls)
            return controls, moment_one, moment_two, best_controls, best_values

        initial_carry = (
            initial_controls,
            first_moment,
            second_moment,
            initial_controls,
            initial_values,
        )
        final_carry = jax.lax.fori_loop(0, config.refinement_steps, body, initial_carry)
        return final_carry[3], final_carry[4]

    def translate_neighbor_controls(
        source_controls: Array,
        source_consumption: Array,
        flat_states: Array,
    ) -> tuple[Array, Array]:
        """Represent a neighbor's absolute controls at the current states."""

        hours = source_controls[:, 0]
        investment_share = source_controls[:, 1]
        training_time = hours * investment_share
        consumption_capacity = maximum_feasible_consumption(
            flat_states[:, 0],
            flat_states[:, 1],
            hours,
            training_time,
            params,
            step,
            asset_minimum,
            config.path_checkpoints,
        )
        consumption_span = jnp.maximum(consumption_capacity - config.consumption_floor, 0.0)
        safe_span = jnp.where(consumption_span > 0.0, consumption_span, 1.0)
        consumption_fraction = (source_consumption - config.consumption_floor) / safe_span
        translated_controls = source_controls.at[:, 2].set(jnp.clip(consumption_fraction, 0.0, 1.0))
        representable = (
            (consumption_capacity >= config.consumption_floor)
            & (source_consumption >= config.consumption_floor - 1e-10)
            & (source_consumption <= consumption_capacity + 1e-10)
        )
        return translated_controls, representable

    def consider_neighbor_axis(
        carry: tuple[Array, Array, Array, Array],
        flat_states: Array,
        continuation_values: Array,
        axis: int,
    ) -> tuple[Array, Array, Array, Array]:
        controls, values, consumption, training_time = carry
        assert neighbor_shape is not None
        controls_grid = controls.reshape((*neighbor_shape, 3))
        consumption_grid = consumption.reshape(neighbor_shape)
        source_controls = jnp.roll(controls_grid, shift=1, axis=axis).reshape((-1, 3))
        source_consumption = jnp.roll(consumption_grid, shift=1, axis=axis).ravel()
        translated_controls, representable = translate_neighbor_controls(
            source_controls, source_consumption, flat_states
        )
        candidate_values, candidate_feasible, candidate_consumption, candidate_training = (
            evaluate_controls(
                translated_controls,
                flat_states,
                continuation_values,
            )
        )
        if axis == 0:
            has_neighbor = jnp.broadcast_to(
                jnp.arange(neighbor_shape[0])[:, None] > 0, neighbor_shape
            ).ravel()
        else:
            has_neighbor = jnp.broadcast_to(
                jnp.arange(neighbor_shape[1])[None, :] > 0, neighbor_shape
            ).ravel()
        same_consumption = jnp.isclose(
            candidate_consumption,
            source_consumption,
            rtol=1e-10,
            atol=1e-10,
        )
        improved = (
            has_neighbor
            & representable
            & same_consumption
            & candidate_feasible
            & (candidate_values > values)
        )
        return (
            jnp.where(improved[:, None], translated_controls, controls),
            jnp.where(improved, candidate_values, values),
            jnp.where(improved, candidate_consumption, consumption),
            jnp.where(improved, candidate_training, training_time),
        )

    def consider_incumbent_policy(
        carry: tuple[Array, Array, Array, Array],
        incumbent_policy: Array,
        flat_states: Array,
        continuation_values: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Retain an interpolated policy whenever it beats the fresh search."""

        controls, values, consumption, training_time = carry
        flat_policy = incumbent_policy.reshape((-1, 3))
        incumbent_consumption = flat_policy[:, 0]
        incumbent_hours = jnp.clip(flat_policy[:, 1], 0.0, 1.0 - config.leisure_floor)
        incumbent_training = jnp.clip(flat_policy[:, 2], 0.0, incumbent_hours)
        incumbent_investment = jnp.where(
            incumbent_hours > 1e-12,
            incumbent_training / incumbent_hours,
            0.0,
        )
        incumbent_internal = jnp.column_stack(
            (
                incumbent_hours,
                incumbent_investment,
                jnp.zeros_like(incumbent_hours),
            )
        )
        translated_controls, representable = translate_neighbor_controls(
            incumbent_internal,
            incumbent_consumption,
            flat_states,
        )
        candidate_values, candidate_feasible, candidate_consumption, candidate_training = (
            evaluate_controls(
                translated_controls,
                flat_states,
                continuation_values,
            )
        )
        same_consumption = jnp.isclose(
            candidate_consumption,
            incumbent_consumption,
            rtol=1e-10,
            atol=1e-10,
        )
        improved = (
            representable
            & same_consumption
            & candidate_feasible
            & (candidate_values > values + 1e-12)
        )
        return (
            jnp.where(improved[:, None], translated_controls, controls),
            jnp.where(improved, candidate_values, values),
            jnp.where(improved, candidate_consumption, consumption),
            jnp.where(improved, candidate_training, training_time),
        )

    def optimize_states(
        states: Array,
        continuation_values: Array,
        incumbent_policy: Array | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        state_shape = states.shape[:-1]
        flat_states = states.reshape((-1, 2))
        flat_assets = flat_states[:, 0]
        flat_log_human_capital = flat_states[:, 1]
        state_consumption_capacity = maximum_feasible_consumption(
            flat_assets[:, None],
            flat_log_human_capital[:, None],
            candidate_hours[None, :],
            candidate_training[None, :],
            params,
            step,
            asset_minimum,
            config.path_checkpoints,
        )
        consumption_span = jnp.maximum(state_consumption_capacity - config.consumption_floor, 0.0)
        consumption = (
            config.consumption_floor
            + consumption_span[:, :, None] * (consumption_fractions[None, None, :])
        )
        hours = candidate_hours[None, :, None]
        training_time = candidate_training[None, :, None]
        state = jnp.stack(
            (
                jnp.broadcast_to(flat_assets[:, None, None], consumption.shape),
                jnp.broadcast_to(flat_log_human_capital[:, None, None], consumption.shape),
            ),
            axis=-1,
        )
        control = jnp.stack(
            (
                consumption,
                jnp.broadcast_to(hours, consumption.shape),
                jnp.broadcast_to(training_time, consumption.shape),
            ),
            axis=-1,
        )
        next_states = constant_control_transition(state, control, params, step)
        continuation = _interpolate_jax(
            continuation_values,
            asset_grid,
            log_human_capital_grid,
            next_states[..., 0],
            next_states[..., 1],
        )
        candidate_values = (
            flow_discount * flow_utility(consumption, hours, params) + beta * continuation
        )
        feasible = (
            (state_consumption_capacity[:, :, None] >= config.consumption_floor)
            & (next_states[..., 0] >= asset_minimum - 1e-10)
            & (next_states[..., 0] <= asset_maximum + 1e-10)
            & (next_states[..., 1] >= log_human_capital_minimum - 1e-10)
            & (next_states[..., 1] <= log_human_capital_maximum + 1e-10)
        )
        candidate_values = jnp.where(feasible, candidate_values, -jnp.inf)
        flat_candidate_values = candidate_values.reshape((flat_states.shape[0], -1))
        best_index = jnp.argmax(flat_candidate_values, axis=1)
        best_values = jnp.take_along_axis(flat_candidate_values, best_index[:, None], axis=1)[:, 0]
        consumption_index = best_index % config.consumption_nodes
        active_time_index = best_index // config.consumption_nodes
        best_controls = jnp.column_stack(
            (
                candidate_hours[active_time_index],
                candidate_investment[active_time_index],
                consumption_fractions[consumption_index],
            )
        )
        if config.refinement_steps:
            best_controls, best_values = refine_controls(
                best_controls,
                best_values,
                flat_states,
                continuation_values,
            )
        final_values, final_feasible, final_consumption, final_training = evaluate_controls(
            best_controls,
            flat_states,
            continuation_values,
        )
        final_values = jnp.where(final_feasible, final_values, best_values)
        if incumbent_policy is not None:
            best_controls, final_values, final_consumption, final_training = (
                consider_incumbent_policy(
                    (
                        best_controls,
                        final_values,
                        final_consumption,
                        final_training,
                    ),
                    incumbent_policy,
                    flat_states,
                    continuation_values,
                )
            )
        if neighbor_shape is not None and config.neighbor_policy_sweeps:
            initial_carry = (
                best_controls,
                final_values,
                final_consumption,
                final_training,
            )

            def neighbor_sweep(_, carry):
                carry = consider_neighbor_axis(carry, flat_states, continuation_values, axis=0)
                return consider_neighbor_axis(carry, flat_states, continuation_values, axis=1)

            best_controls, final_values, final_consumption, final_training = jax.lax.fori_loop(
                0,
                config.neighbor_policy_sweeps,
                neighbor_sweep,
                initial_carry,
            )
        return (
            final_values.reshape(state_shape),
            final_consumption.reshape(state_shape),
            best_controls[:, 0].reshape(state_shape),
            final_training.reshape(state_shape),
        )

    return optimize_states


def _make_bellman_step(
    params: ModelParams,
    config: BellmanConfig,
    asset_grid: Array,
    log_human_capital_grid: Array,
):
    """Create one JIT-compatible backward-induction step on the state grid."""

    state_assets, state_log_human_capital = jnp.meshgrid(
        asset_grid, log_human_capital_grid, indexing="ij"
    )
    states = jnp.stack((state_assets, state_log_human_capital), axis=-1)
    optimize_states = _make_control_optimizer(
        params,
        config,
        asset_grid,
        log_human_capital_grid,
        neighbor_shape=(config.asset_nodes, config.human_capital_nodes),
    )

    def bellman_step(continuation_values: Array) -> tuple[Array, Array, Array, Array]:
        return optimize_states(states, continuation_values)

    return bellman_step


def solve_bellman(
    params: ModelParams | None = None,
    config: BellmanConfig | None = None,
) -> BellmanSolution:
    """Solve the deterministic finite-horizon Bellman equation backward.

    The complete backward recursion is one compiled JAX program.  With the
    default ``compute_platform="auto"``, JAX selects CUDA when a GPU backend is
    available and otherwise falls back to CPU.  Set ``compute_platform="gpu"``
    to require CUDA and fail early if it is unavailable.
    """

    params = benchmark_params() if params is None else params
    config = BellmanConfig() if config is None else config
    _validate_config(params, config)
    compute_device = _select_compute_device(config)
    asset_grid_np, log_human_capital_grid_np = bellman_state_grids(config)
    asset_grid = jax.device_put(asset_grid_np, compute_device)
    log_human_capital_grid = jax.device_put(log_human_capital_grid_np, compute_device)
    bellman_step = _make_bellman_step(params, config, asset_grid, log_human_capital_grid)

    terminal_assets = jnp.broadcast_to(
        asset_grid[:, None], (config.asset_nodes, config.human_capital_nodes)
    )
    terminal_values = bequest_utility(terminal_assets, params)

    def backward_solve(
        terminal_continuation: Array,
    ) -> tuple[Array, Array, Array, Array]:
        def backward_step(
            next_values: Array, _: None
        ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
            period_result = bellman_step(next_values)
            return period_result[0], period_result

        _, reverse_histories = jax.lax.scan(
            backward_step,
            terminal_continuation,
            xs=None,
            length=config.periods,
        )
        reverse_values, reverse_consumption, reverse_hours, reverse_training = reverse_histories
        values = jnp.concatenate((reverse_values[::-1], terminal_continuation[None, ...]), axis=0)
        return (
            values,
            reverse_consumption[::-1],
            reverse_hours[::-1],
            reverse_training[::-1],
        )

    # ``terminal_values`` and all closed-over grids already reside on the
    # selected device, which determines where this compiled program executes.
    compiled_backward_solve = jax.jit(backward_solve)

    start = perf_counter()
    device_histories = compiled_backward_solve(terminal_values)
    values, consumption_policy, hours_policy, training_time_policy = (
        np.asarray(array) for array in jax.device_get(device_histories)
    )
    solve_seconds = perf_counter() - start
    if not all(
        np.all(np.isfinite(array))
        for array in (
            values,
            consumption_policy,
            hours_policy,
            training_time_policy,
        )
    ):
        raise RuntimeError(
            "Bellman recursion produced a non-finite value or policy at one or more "
            "state nodes; expand the state domain or refine the control grid"
        )

    return BellmanSolution(
        params=params,
        config=config,
        time=np.linspace(0.0, params.horizon, config.periods + 1),
        asset_grid=asset_grid_np,
        log_human_capital_grid=log_human_capital_grid_np,
        values=values,
        consumption_policy=consumption_policy,
        hours_policy=hours_policy,
        training_time_policy=training_time_policy,
        solve_seconds=solve_seconds,
        backend=compute_device.platform,
        device=f"{compute_device.platform}:{compute_device.id} ({compute_device.device_kind})",
    )


def policy_at(
    solution: BellmanSolution,
    period: int,
    assets: float | np.ndarray,
    human_capital: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate ``(c, h, q)`` at states in a specified discrete period."""

    if not 0 <= period < solution.config.periods:
        raise ValueError("period is outside the policy horizon")
    log_human_capital = np.log(human_capital)
    arguments = (
        solution.asset_grid,
        solution.log_human_capital_grid,
        assets,
        log_human_capital,
    )
    consumption = _interpolate_numpy(solution.consumption_policy[period], *arguments)
    hours = _interpolate_numpy(solution.hours_policy[period], *arguments)
    training_time = _interpolate_numpy(solution.training_time_policy[period], *arguments)
    training_time = np.clip(training_time, 0.0, hours)
    return consumption, hours, training_time


@lru_cache(maxsize=16)
def _cached_greedy_kernels(
    params: ModelParams,
    config: BellmanConfig,
    backend: str,
    device_index: int,
) -> tuple[Any, Any, Any]:
    """Compile and cache off-grid policy recovery for one model configuration."""

    devices = jax.devices(backend)
    if device_index >= len(devices):
        raise RuntimeError(
            f"The solution requires {backend}:{device_index}, but that JAX device "
            "is not available in this process"
        )
    device = devices[device_index]
    asset_grid_np, log_human_capital_grid_np = bellman_state_grids(config)
    asset_grid = jax.device_put(asset_grid_np, device)
    log_human_capital_grid = jax.device_put(log_human_capital_grid_np, device)
    optimize_states = _make_control_optimizer(
        params,
        config,
        asset_grid,
        log_human_capital_grid,
    )

    def recover_policy(
        states: Array,
        continuation_values: Array,
        node_policy: Array,
    ) -> tuple[Array, Array, Array, Array]:
        incumbent_policy = jnp.stack(
            tuple(
                _interpolate_jax(
                    node_policy[..., control_index],
                    asset_grid,
                    log_human_capital_grid,
                    states[..., 0],
                    states[..., 1],
                )
                for control_index in range(3)
            ),
            axis=-1,
        )
        return optimize_states(
            states,
            continuation_values,
            incumbent_policy,
        )

    def rollout(
        initial_state: Array,
        continuation_history: Array,
        node_policy_history: Array,
    ) -> tuple[Array, Array, Array]:
        def forward_step(
            state: Array,
            period_inputs: tuple[Array, Array],
        ) -> tuple[Array, tuple[Array, Array, Array]]:
            continuation_values, node_policy = period_inputs
            policy_value, consumption, hours, training_time = recover_policy(
                state,
                continuation_values,
                node_policy,
            )
            control = jnp.stack((consumption, hours, training_time), axis=-1)
            next_state = constant_control_transition(
                state,
                control,
                params,
                params.horizon / config.periods,
            )
            return next_state, (state, control, policy_value)

        final_state, (states, controls, policy_values) = jax.lax.scan(
            forward_step,
            initial_state,
            (continuation_history, node_policy_history),
        )
        full_states = jnp.concatenate((states, final_state[None, ...]), axis=0)
        return full_states, controls, policy_values

    return jax.jit(recover_policy), jax.jit(rollout), device


def greedy_policy_at(
    solution: BellmanSolution,
    period: int,
    assets: float | np.ndarray,
    human_capital: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the Bellman-greedy policy at arbitrary in-domain states.

    Unlike :func:`policy_at`, this function does not interpolate stored node
    controls. It batches the supplied states and maximizes the period Bellman
    objective against the interpolated next-period value function.
    """

    if not 0 <= period < solution.config.periods:
        raise ValueError("period is outside the policy horizon")
    assets_array, human_capital_array = np.broadcast_arrays(
        np.asarray(assets, dtype=np.float64),
        np.asarray(human_capital, dtype=np.float64),
    )
    if np.any(human_capital_array <= 0.0):
        raise ValueError("human_capital must be positive")
    log_human_capital = np.log(human_capital_array)
    in_domain = (
        (assets_array >= solution.asset_grid[0])
        & (assets_array <= solution.asset_grid[-1])
        & (log_human_capital >= solution.log_human_capital_grid[0])
        & (log_human_capital <= solution.log_human_capital_grid[-1])
    )
    if not np.all(in_domain):
        raise ValueError("greedy policy states must lie inside the Bellman domain")

    states = np.stack((assets_array, log_human_capital), axis=-1)
    recover_policy, _, device = _cached_greedy_kernels(
        solution.params,
        solution.config,
        solution.backend,
        solution.config.device_index,
    )
    node_policy = np.stack(
        (
            solution.consumption_policy[period],
            solution.hours_policy[period],
            solution.training_time_policy[period],
        ),
        axis=-1,
    )
    recovered = jax.device_get(
        recover_policy(
            jax.device_put(states, device),
            jax.device_put(solution.values[period + 1], device),
            jax.device_put(node_policy, device),
        )
    )
    policy_value, consumption, hours, training_time = (np.asarray(array) for array in recovered)
    if not np.all(np.isfinite(policy_value)):
        raise RuntimeError("no finite feasible Bellman control at an off-grid state")
    training_time = np.clip(training_time, 0.0, hours)
    return consumption, hours, training_time


def simulate_policy(
    solution: BellmanSolution,
    *,
    initial_assets: float | None = None,
    initial_human_capital: float | None = None,
    policy_method: Literal["greedy", "interpolate"] = "greedy",
) -> BellmanSimulation:
    """Simulate a piecewise-constant feedback policy from an initial state."""

    params = solution.params
    config = solution.config
    step = params.horizon / config.periods
    initial_assets = params.initial_assets if initial_assets is None else initial_assets
    initial_human_capital = (
        params.initial_human_capital if initial_human_capital is None else initial_human_capital
    )
    if initial_human_capital <= 0.0:
        raise ValueError("initial_human_capital must be positive")
    if policy_method not in {"greedy", "interpolate"}:
        raise ValueError("policy_method must be 'greedy' or 'interpolate'")

    initial_log_human_capital = np.log(initial_human_capital)
    initial_state_in_domain = (
        solution.asset_grid[0] <= initial_assets <= solution.asset_grid[-1]
        and solution.log_human_capital_grid[0]
        <= initial_log_human_capital
        <= solution.log_human_capital_grid[-1]
    )

    if policy_method == "greedy":
        if not initial_state_in_domain:
            raise ValueError("initial state must lie inside the Bellman domain")
        _, rollout, device = _cached_greedy_kernels(
            solution.params,
            solution.config,
            solution.backend,
            solution.config.device_index,
        )
        device_result = rollout(
            jax.device_put(np.asarray([initial_assets, initial_log_human_capital]), device),
            jax.device_put(solution.values[1:], device),
            jax.device_put(
                np.stack(
                    (
                        solution.consumption_policy,
                        solution.hours_policy,
                        solution.training_time_policy,
                    ),
                    axis=-1,
                ),
                device,
            ),
        )
        full_states, controls, recovered_values = (
            np.asarray(array) for array in jax.device_get(device_result)
        )
        if not np.all(np.isfinite(recovered_values)):
            raise RuntimeError("greedy rollout encountered a state with no feasible control")
        assets = full_states[:, 0]
        log_human_capital = full_states[:, 1]
        consumption = controls[:, 0]
        hours = controls[:, 1]
        training_time = controls[:, 2]
        states_in_domain = (
            (assets >= solution.asset_grid[0])
            & (assets <= solution.asset_grid[-1])
            & (log_human_capital >= solution.log_human_capital_grid[0])
            & (log_human_capital <= solution.log_human_capital_grid[-1])
        )
        stayed_in_domain = bool(np.all(states_in_domain))
    else:
        assets = np.empty(config.periods + 1, dtype=np.float64)
        log_human_capital = np.empty(config.periods + 1, dtype=np.float64)
        consumption = np.empty(config.periods, dtype=np.float64)
        hours = np.empty(config.periods, dtype=np.float64)
        training_time = np.empty(config.periods, dtype=np.float64)
        assets[0] = initial_assets
        log_human_capital[0] = initial_log_human_capital
        stayed_in_domain = initial_state_in_domain

        for period in range(config.periods):
            interpolated = policy_at(
                solution,
                period,
                assets[period],
                np.exp(log_human_capital[period]),
            )
            candidate_consumption = float(interpolated[0])
            candidate_hours = float(np.clip(interpolated[1], 0.0, 1.0 - config.leisure_floor))
            candidate_training = float(np.clip(interpolated[2], 0.0, candidate_hours))
            consumption_capacity = float(
                maximum_feasible_consumption(
                    jnp.asarray(assets[period]),
                    jnp.asarray(log_human_capital[period]),
                    jnp.asarray(candidate_hours),
                    jnp.asarray(candidate_training),
                    params,
                    step,
                    config.asset_minimum,
                    config.path_checkpoints,
                )
            )
            stayed_in_domain = stayed_in_domain and (
                consumption_capacity >= config.consumption_floor
            )
            consumption[period] = np.clip(
                candidate_consumption,
                config.consumption_floor,
                max(config.consumption_floor, consumption_capacity),
            )
            hours[period] = candidate_hours
            training_time[period] = candidate_training
            next_state = np.asarray(
                constant_control_transition(
                    jnp.asarray([assets[period], log_human_capital[period]]),
                    jnp.asarray(
                        [
                            consumption[period],
                            hours[period],
                            training_time[period],
                        ]
                    ),
                    params,
                    step,
                )
            )
            assets[period + 1] = next_state[0]
            log_human_capital[period + 1] = next_state[1]
            next_state_in_domain = (
                solution.asset_grid[0] <= next_state[0] <= solution.asset_grid[-1]
                and solution.log_human_capital_grid[0]
                <= next_state[1]
                <= solution.log_human_capital_grid[-1]
            )
            stayed_in_domain = stayed_in_domain and next_state_in_domain

    discount = np.exp(-params.rho * solution.time[:-1])
    flow_discount = float(step * _exprel(jnp.asarray(-params.rho * step)))
    utility = float(
        np.sum(
            discount
            * flow_discount
            * np.asarray(flow_utility(jnp.asarray(consumption), jnp.asarray(hours), params))
        )
        + np.exp(-params.rho * params.horizon)
        * float(bequest_utility(jnp.asarray(assets[-1]), params))
    )
    initial_value = float(
        _interpolate_numpy(
            solution.values[0],
            solution.asset_grid,
            solution.log_human_capital_grid,
            initial_assets,
            np.log(initial_human_capital),
        )
    )
    return BellmanSimulation(
        solution=solution,
        policy_method=policy_method,
        time=solution.time,
        assets=assets,
        log_human_capital=log_human_capital,
        human_capital=np.exp(log_human_capital),
        consumption=consumption,
        hours=hours,
        training_time=training_time,
        lifetime_utility=utility,
        initial_value=initial_value,
        value_gap=initial_value - utility,
        stayed_in_domain=bool(stayed_in_domain),
        minimum_assets=float(np.min(assets)),
    )


def diagnose_bellman(
    solution: BellmanSolution,
    simulation: BellmanSimulation | None = None,
    *,
    tolerance: float = 1e-8,
) -> BellmanDiagnostics:
    """Check Bellman consistency, node feasibility, and domain containment."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    simulation = simulate_policy(solution) if simulation is None else simulation
    params = solution.params
    config = solution.config
    step = params.horizon / config.periods
    asset_nodes, human_capital_nodes = np.meshgrid(
        solution.asset_grid,
        solution.log_human_capital_grid,
        indexing="ij",
    )
    state_nodes = jnp.stack((jnp.asarray(asset_nodes), jnp.asarray(human_capital_nodes)), axis=-1)
    beta = float(np.exp(-params.rho * step))
    flow_discount = float(step * _exprel(-params.rho * step))
    maximum_bellman_residual = 0.0
    minimum_capacity_slack = np.inf
    next_states_inside_domain = True

    for period in range(config.periods):
        consumption = jnp.asarray(solution.consumption_policy[period])
        hours = jnp.asarray(solution.hours_policy[period])
        training_time = jnp.asarray(solution.training_time_policy[period])
        controls = jnp.stack((consumption, hours, training_time), axis=-1)
        next_states = constant_control_transition(state_nodes, controls, params, step)
        continuation = _interpolate_jax(
            jnp.asarray(solution.values[period + 1]),
            jnp.asarray(solution.asset_grid),
            jnp.asarray(solution.log_human_capital_grid),
            next_states[..., 0],
            next_states[..., 1],
        )
        policy_value = (
            flow_discount * flow_utility(consumption, hours, params) + beta * continuation
        )
        residual = np.asarray(solution.values[period] - policy_value)
        maximum_bellman_residual = max(maximum_bellman_residual, float(np.max(np.abs(residual))))
        consumption_capacity = maximum_feasible_consumption(
            state_nodes[..., 0],
            state_nodes[..., 1],
            hours,
            training_time,
            params,
            step,
            config.asset_minimum,
            config.path_checkpoints,
        )
        minimum_capacity_slack = min(
            minimum_capacity_slack,
            float(np.min(np.asarray(consumption_capacity - consumption))),
        )
        next_states_np = np.asarray(next_states)
        period_inside = (
            np.min(next_states_np[..., 0]) >= solution.asset_grid[0] - tolerance
            and np.max(next_states_np[..., 0]) <= solution.asset_grid[-1] + tolerance
            and np.min(next_states_np[..., 1]) >= solution.log_human_capital_grid[0] - tolerance
            and np.max(next_states_np[..., 1]) <= solution.log_human_capital_grid[-1] + tolerance
        )
        next_states_inside_domain = next_states_inside_domain and period_inside

    minimum_asset_difference = float(np.min(np.diff(solution.values, axis=1)))
    minimum_human_capital_difference = float(np.min(np.diff(solution.values, axis=2)))
    maximum_monotonicity_violation = max(
        0.0,
        -minimum_asset_difference,
        -minimum_human_capital_difference,
    )
    all_values_finite = bool(np.all(np.isfinite(solution.values)))
    minimum_training_time = float(np.min(solution.training_time_policy))
    minimum_training_slack = float(np.min(solution.hours_policy - solution.training_time_policy))
    minimum_hours = float(np.min(solution.hours_policy))
    maximum_hours = float(np.max(solution.hours_policy))
    accepted = (
        all_values_finite
        and maximum_bellman_residual <= tolerance
        and minimum_capacity_slack >= -tolerance
        and minimum_training_time >= -tolerance
        and minimum_training_slack >= -tolerance
        and minimum_hours >= -tolerance
        and maximum_hours <= 1.0 - config.leisure_floor + tolerance
        and next_states_inside_domain
    )
    return BellmanDiagnostics(
        accepted_node_solution=bool(accepted),
        all_values_finite=all_values_finite,
        maximum_node_bellman_residual=maximum_bellman_residual,
        minimum_consumption_capacity_slack=minimum_capacity_slack,
        minimum_training_time=minimum_training_time,
        minimum_training_slack=minimum_training_slack,
        minimum_hours=minimum_hours,
        maximum_hours=maximum_hours,
        maximum_value_monotonicity_violation=maximum_monotonicity_violation,
        next_states_inside_domain=bool(next_states_inside_domain),
        simulation_stayed_in_domain=simulation.stayed_in_domain,
        simulation_minimum_assets=simulation.minimum_assets,
        simulation_value_gap=simulation.value_gap,
    )
