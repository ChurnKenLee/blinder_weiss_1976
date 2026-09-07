"""Batched heterogeneous lifecycles and weighted model age profiles.

These are model moments, not an ACS/ATUS measurement specification. Hours and
training remain fractions of the model time endowment; earnings are the flow
rate at the start of each decision period, ``h * g(q / h) * K``. Mapping these
quantities to survey units, ages, universes, and employment definitions is a
separate calibration decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import ArrayLike

from .bellman import (
    BellmanConfig,
    BellmanSolution,
    _cached_greedy_kernels,
    maximum_feasible_consumption,
)
from .model import ModelParams, effective_earnings_share

_FEASIBILITY_TOLERANCE = 1e-8


@dataclass(frozen=True)
class CohortSimulation:
    """Greedy lifecycles with axes ``(time, person, component)``.

    ``states`` contains assets and log human capital at all ``periods + 1``
    boundaries. ``controls`` contains consumption, hours, and training time
    during each of the ``periods`` decision intervals. ``weights`` sums to one.
    ``policy_values`` records the recovered Bellman objective at each state;
    it is not the realized remaining utility of the simulated trajectory.
    Every returned path passed the numerical domain and feasibility checks.
    """

    solution: BellmanSolution
    states: np.ndarray
    controls: np.ndarray
    weights: np.ndarray
    policy_values: np.ndarray
    earnings: np.ndarray
    minimum_consumption_capacity_slack: float

    @property
    def time(self) -> np.ndarray:
        return self.solution.time

    @property
    def assets(self) -> np.ndarray:
        return self.states[..., 0]

    @property
    def log_human_capital(self) -> np.ndarray:
        return self.states[..., 1]

    @property
    def human_capital(self) -> np.ndarray:
        return np.exp(self.log_human_capital)

    @property
    def consumption(self) -> np.ndarray:
        return self.controls[..., 0]

    @property
    def hours(self) -> np.ndarray:
        return self.controls[..., 1]

    @property
    def training_time(self) -> np.ndarray:
        return self.controls[..., 2]


@dataclass(frozen=True)
class CohortMoments:
    """Weighted, unconditional means at the start of each decision period.

    Participation is ``hours > participation_hours_threshold``. Model time
    starts at zero; choosing a calendar-age origin is left to the caller.
    Zero-weight people do not contribute, and nonparticipants remain in all
    denominators. Training is ``q``, not the conditional share ``q / h``.
    """

    time: np.ndarray
    hours: np.ndarray
    participation: np.ndarray
    training_time: np.ndarray
    consumption: np.ndarray
    earnings: np.ndarray
    participation_hours_threshold: float


def _initial_cohort(
    solution: BellmanSolution,
    initial_assets: ArrayLike,
    initial_human_capital: ArrayLike,
    weights: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    assets, human_capital = np.broadcast_arrays(
        np.asarray(initial_assets, dtype=np.float64),
        np.asarray(initial_human_capital, dtype=np.float64),
    )
    if assets.ndim > 1 or assets.size == 0:
        raise ValueError("initial states must be nonempty scalars or one-dimensional arrays")
    assets = assets.reshape(-1)
    human_capital = human_capital.reshape(-1)
    if not np.all(np.isfinite(assets)):
        raise ValueError("initial_assets must be finite")
    if not np.all(np.isfinite(human_capital)) or np.any(human_capital <= 0.0):
        raise ValueError("initial_human_capital must be finite and positive")
    log_human_capital = np.log(human_capital)
    if not np.all(
        (assets >= solution.asset_grid[0])
        & (assets <= solution.asset_grid[-1])
        & (log_human_capital >= solution.log_human_capital_grid[0])
        & (log_human_capital <= solution.log_human_capital_grid[-1])
    ):
        raise ValueError("initial cohort states must lie inside the Bellman domain")

    if weights is None:
        normalized_weights = np.full(assets.size, 1.0 / assets.size)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != assets.shape:
            raise ValueError("weights must have one entry per person")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        maximum_weight = np.max(weight_array)
        if maximum_weight <= 0.0:
            raise ValueError("weights must have positive total mass")
        # Scaling first avoids overflow even for very large finite survey weights.
        scaled_weights = weight_array / maximum_weight
        normalized_weights = scaled_weights / np.sum(scaled_weights)
    return np.stack((assets, log_human_capital), axis=-1), normalized_weights


@lru_cache(maxsize=16)
def _cached_cohort_checks(config: BellmanConfig) -> Any:
    """Compile cheap batch diagnostics once per discretization and shape."""

    @jax.jit
    def check(
        params: ModelParams,
        states: Array,
        controls: Array,
        policy_values: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        assets, log_human_capital = states[..., 0], states[..., 1]
        consumption, hours, training = controls[..., 0], controls[..., 1], controls[..., 2]
        capacity = maximum_feasible_consumption(
            assets[:-1],
            log_human_capital[:-1],
            hours,
            training,
            params,
            params.horizon / config.periods,
            config.asset_minimum,
            config.path_checkpoints,
        )
        minimum_slack = jnp.min(capacity - consumption)
        tolerance = _FEASIBILITY_TOLERANCE
        in_domain = jnp.all(
            (assets >= config.asset_minimum - tolerance)
            & (assets <= config.asset_maximum + tolerance)
            & (log_human_capital >= config.log_human_capital_minimum - tolerance)
            & (log_human_capital <= config.log_human_capital_maximum + tolerance)
        )
        feasible = jnp.all(
            (consumption >= config.consumption_floor - tolerance)
            & (consumption > 0.0)
            & (hours >= -tolerance)
            & (hours <= 1.0 - config.leisure_floor + tolerance)
            & (training >= -tolerance)
            & (training <= hours + tolerance)
        ) & (minimum_slack >= -tolerance)
        earnings = effective_earnings_share(hours, training) * jnp.exp(log_human_capital[:-1])
        finite = (
            jnp.all(jnp.isfinite(states))
            & jnp.all(jnp.isfinite(controls))
            & jnp.all(jnp.isfinite(policy_values))
            & jnp.all(jnp.isfinite(earnings))
        )
        return earnings, minimum_slack, in_domain, feasible, finite

    return check


def simulate_cohort(
    solution: BellmanSolution,
    initial_assets: ArrayLike,
    initial_human_capital: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> CohortSimulation:
    """Simulate heterogeneous initial states in one compiled greedy scan.

    Initial assets and human capital are scalars or broadcastable vectors.
    Optional nonnegative weights have one entry per person and are normalized
    internally. All people share the supplied solution's model parameters.
    Keep the cohort size and Bellman configuration fixed across calibration
    evaluations to reuse JAX compilation; changing states, weights, or model
    parameter values does not itself require a new executable.

    Raises ``ValueError`` for invalid initial states or weights and
    ``RuntimeError`` for nonfinite, infeasible, or out-of-domain trajectories.
    Feasibility uses the solver's within-period asset checkpoints and an
    absolute numerical tolerance of ``1e-8``; this is not a continuous-time
    mesh-convergence certificate.
    """

    initial_states, normalized_weights = _initial_cohort(
        solution, initial_assets, initial_human_capital, weights
    )
    _, rollout, device = _cached_greedy_kernels(
        solution.config, solution.backend, solution.config.device_index
    )
    params = jax.device_put(
        jax.tree.map(lambda value: np.asarray(value, dtype=np.float64), solution.params), device
    )
    node_policy = np.stack(
        (solution.consumption_policy, solution.hours_policy, solution.training_time_policy),
        axis=-1,
    )
    states_device, controls_device, values_device = rollout(
        params,
        jax.device_put(initial_states, device),
        jax.device_put(solution.values[1:], device),
        jax.device_put(node_policy, device),
    )
    checks = _cached_cohort_checks(solution.config)(
        params, states_device, controls_device, values_device
    )
    states, controls, policy_values, check_results = jax.device_get(
        (states_device, controls_device, values_device, checks)
    )
    earnings, minimum_slack, in_domain, feasible, finite = check_results
    if not finite:
        raise RuntimeError("cohort rollout encountered nonfinite states or no feasible control")
    if not in_domain:
        raise RuntimeError("cohort rollout left the Bellman domain; expand the state domain")
    if not feasible:
        raise RuntimeError("cohort rollout violated a control or within-period asset constraint")
    return CohortSimulation(
        solution=solution,
        states=np.asarray(states),
        controls=np.asarray(controls),
        weights=normalized_weights,
        policy_values=np.asarray(policy_values),
        earnings=np.asarray(earnings),
        minimum_consumption_capacity_slack=float(minimum_slack),
    )


def cohort_moments(
    simulation: CohortSimulation,
    *,
    participation_hours_threshold: float,
) -> CohortMoments:
    """Reduce a cohort to weighted model profiles without additional solves.

    The participation threshold must be explicitly chosen in model hours
    units in ``[0, 1)``. A person at exactly the threshold is not counted as
    participating. These are unconditional population means, including all
    people regardless of hours. Earnings are period-start rates, not total
    or average earnings over a Bellman interval.
    """

    threshold = float(participation_hours_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError("participation_hours_threshold must be finite and lie in [0, 1)")
    weights = simulation.weights
    return CohortMoments(
        time=simulation.time[:-1].copy(),
        hours=simulation.hours @ weights,
        participation=(simulation.hours > threshold) @ weights,
        training_time=simulation.training_time @ weights,
        consumption=simulation.consumption @ weights,
        earnings=simulation.earnings @ weights,
        participation_hours_threshold=threshold,
    )
