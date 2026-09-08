"""Conservative, deterministic population transport on an independent grid.

Arrays contain probability *masses*, never density samples. Row zero is the
exact numerical asset-floor face; all remaining rows are strictly interior.
The benchmark's economic borrowing limit is zero, whereas the Bellman solver
uses a positive epsilon floor. ``asset_floor_mass`` therefore refers to that
numerical approximation, not a verified economic atom at the borrowing limit.

The local transfer operator is independent of value interpolation. Interior
destinations below the first interior asset node remain interior and are
assigned to that node. This avoids manufactured floor atoms but introduces an
explicitly reported first-cell remapping error. Grid and floor refinement, and
agreement with quadrature, are required before using this backend by default.
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
    _DOMAIN_TOLERANCE,
    BellmanConfig,
    BellmanSolution,
    _cached_greedy_kernels,
    _exprel,
    constant_control_transition,
    endpoint_consumption_capacity,
    maximum_feasible_consumption,
    minimum_assets_during_step,
)
from .boundary import _FLOOR_CONTACT_ULPS, endpoint_floor_contact
from .model import ModelParams, effective_earnings_share
from .population import CohortMoments


@dataclass(frozen=True)
class DistributionGrid:
    """Independent asset/log-K nodes with a separate numerical floor face.

    ``asset_interior_nodes`` is strictly increasing and strictly above
    ``asset_floor``. Artificial upper-asset and both log-K limits are the
    extreme supplied nodes. Flattening is asset-major, including the floor
    row first. Log-K endpoint intersections belong to that face only once.
    """

    asset_floor: float
    asset_interior_nodes: np.ndarray
    log_human_capital_nodes: np.ndarray

    def __post_init__(self) -> None:
        assets = np.array(self.asset_interior_nodes, dtype=np.float64, copy=True)
        log_k = np.array(self.log_human_capital_nodes, dtype=np.float64, copy=True)
        if not np.isfinite(self.asset_floor) or self.asset_floor < 0:
            raise ValueError("asset_floor must be finite and nonnegative")
        for name, nodes in (("asset_interior_nodes", assets), ("log_human_capital_nodes", log_k)):
            if nodes.ndim != 1 or nodes.size < 2 or not np.all(np.isfinite(nodes)):
                raise ValueError(f"{name} must have at least two finite one-dimensional nodes")
            if np.any(np.diff(nodes) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
        if assets[0] <= self.asset_floor:
            raise ValueError("asset_interior_nodes must be strictly above asset_floor")
        assets.setflags(write=False)
        log_k.setflags(write=False)
        object.__setattr__(self, "asset_interior_nodes", assets)
        object.__setattr__(self, "log_human_capital_nodes", log_k)

    @property
    def asset_nodes(self) -> np.ndarray:
        return np.concatenate(([self.asset_floor], self.asset_interior_nodes))

    @property
    def shape(self) -> tuple[int, int]:
        return self.asset_interior_nodes.size + 1, self.log_human_capital_nodes.size

    @property
    def states(self) -> np.ndarray:
        assets, log_k = np.meshgrid(
            self.asset_nodes, self.log_human_capital_nodes, indexing="ij"
        )
        return np.stack((assets.ravel(), log_k.ravel()), axis=-1)


@dataclass(frozen=True)
class LocalTransport:
    """Four nonnegative destinations per source, implementing ``P``.

    ``push`` computes ``P.T @ mass``; ``pull`` computes ``P @ values``.
    These two methods, rather than cubic Bellman interpolation, are adjoints.
    """

    indices: np.ndarray
    weights: np.ndarray
    destination_size: int
    lower_interior_remap: np.ndarray
    asset_remap_error: np.ndarray

    def push(self, masses: ArrayLike) -> np.ndarray:
        source = np.asarray(masses, dtype=np.float64)
        if source.shape != (self.indices.shape[0],):
            raise ValueError("one mass is required per transport source")
        result = np.zeros(self.destination_size, dtype=np.float64)
        np.add.at(result, self.indices.ravel(), (source[:, None] * self.weights).ravel())
        return result

    def pull(self, values: ArrayLike) -> np.ndarray:
        destination = np.asarray(values, dtype=np.float64)
        if destination.shape != (self.destination_size,):
            raise ValueError("one value is required per transport destination")
        return np.sum(self.weights * destination[self.indices], axis=-1)


class DistributionDomainError(RuntimeError):
    """An attempted material domain exit; diagnostics precede any correction."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class DistributionStateMoments:
    """Unconditional state means and floor mass at all age boundaries."""

    time: np.ndarray
    assets: np.ndarray
    human_capital: np.ndarray
    log_human_capital: np.ndarray
    asset_floor_mass: np.ndarray
    near_asset_floor_mass: np.ndarray | None = None
    near_asset_floor_width: float | None = None


@dataclass(frozen=True)
class DistributionSimulation:
    solution: BellmanSolution
    grid: DistributionGrid
    moments: CohortMoments
    state_moments: DistributionStateMoments
    diagnostics: dict[str, Any]
    masses: np.ndarray | None
    terminal_mass: np.ndarray


def _transport_stencil(
    interior_assets: Array,
    log_k_nodes: Array,
    asset_floor: Array,
    destinations: Array,
    on_asset_floor: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Construct local rows, leaving material out-of-domain rows empty."""
    assets, log_k = destinations[:, 0], destinations[:, 1]
    violations = jnp.stack(
        (
            jnp.maximum(asset_floor - assets, 0.0),
            jnp.maximum(assets - interior_assets[-1], 0.0),
            jnp.maximum(log_k_nodes[0] - log_k, 0.0),
            jnp.maximum(log_k - log_k_nodes[-1], 0.0),
        ),
        axis=-1,
    )
    finite = jnp.all(jnp.isfinite(destinations), axis=-1)
    valid = finite & jnp.all(violations <= _DOMAIN_TOLERANCE, axis=-1)
    valid &= ~on_asset_floor | (jnp.abs(assets - asset_floor) <= _DOMAIN_TOLERANCE)
    valid &= (assets >= asset_floor) | on_asset_floor
    # Invalid rows carry zero weight. Clipping below handles roundoff only for
    # valid rows; it never silently reflects/absorbs a material exit.
    safe_assets = jnp.where(valid, assets, interior_assets[0])
    safe_log_k = jnp.where(valid, log_k, log_k_nodes[0])
    safe_assets = jnp.clip(safe_assets, interior_assets[0], interior_assets[-1])
    safe_log_k = jnp.clip(safe_log_k, log_k_nodes[0], log_k_nodes[-1])
    a_index = jnp.clip(
        jnp.searchsorted(interior_assets, safe_assets, side="right") - 1,
        0,
        interior_assets.size - 2,
    )
    k_index = jnp.clip(
        jnp.searchsorted(log_k_nodes, safe_log_k, side="right") - 1,
        0,
        log_k_nodes.size - 2,
    )
    a_weight = (safe_assets - interior_assets[a_index]) / (
        interior_assets[a_index + 1] - interior_assets[a_index]
    )
    k_weight = (safe_log_k - log_k_nodes[k_index]) / (
        log_k_nodes[k_index + 1] - log_k_nodes[k_index]
    )
    a_low = jnp.where(on_asset_floor, 0, a_index + 1)
    a_high = jnp.where(on_asset_floor, 0, a_index + 2)
    a_weight = jnp.where(on_asset_floor, 0.0, a_weight)
    indices = jnp.stack(
        (
            a_low * log_k_nodes.size + k_index,
            a_low * log_k_nodes.size + k_index + 1,
            a_high * log_k_nodes.size + k_index,
            a_high * log_k_nodes.size + k_index + 1,
        ),
        axis=-1,
    )
    weights = jnp.stack(
        (
            (1.0 - a_weight) * (1.0 - k_weight),
            (1.0 - a_weight) * k_weight,
            a_weight * (1.0 - k_weight),
            a_weight * k_weight,
        ),
        axis=-1,
    ) * valid[:, None]
    lower_remap = valid & ~on_asset_floor & (assets < interior_assets[0])
    remap_error = jnp.where(
        valid, jnp.where(on_asset_floor, asset_floor, safe_assets) - assets, 0.0
    )
    return indices, weights, valid, violations, lower_remap, remap_error


def build_transport(
    grid: DistributionGrid,
    destinations: ArrayLike,
    *,
    on_asset_floor: ArrayLike | None = None,
) -> LocalTransport:
    """Build a conservative stencil for arbitrary destination states ``(A,logK)``.

    By default only exact equality or accepted negative roundoff identifies
    the floor. A positive interior state is never declared a boundary atom by
    interpolation. ``on_asset_floor`` may explicitly identify exact constraint
    contacts whose computed destination differs from the floor by roundoff.
    Artificial exits larger than ``1e-10`` raise with uncorrected diagnostics.
    """
    states = np.asarray(destinations, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2 or not states.shape[0]:
        raise ValueError("destinations must be a nonempty (source, 2) array")
    floor = states[:, 0] <= grid.asset_floor if on_asset_floor is None else np.asarray(
        on_asset_floor, dtype=bool
    )
    if floor.shape != (states.shape[0],):
        raise ValueError("on_asset_floor must have one entry per source")
    arrays = _transport_stencil(
        jnp.asarray(grid.asset_interior_nodes),
        jnp.asarray(grid.log_human_capital_nodes),
        jnp.asarray(grid.asset_floor),
        jnp.asarray(states),
        jnp.asarray(floor),
    )
    indices, weights, valid, violations, lower_remap, remap_error = jax.device_get(arrays)
    if not np.all(valid):
        raise DistributionDomainError(
            "transport destination left the distribution domain or has an invalid floor tag; "
            "expand the domain and recheck policies",
            {"valid_rows": valid, "attempted_exit_distances": violations},
        )
    return LocalTransport(
        indices, weights, int(np.prod(grid.shape)), lower_remap, remap_error
    )


def initialize_distribution(
    grid: DistributionGrid,
    initial_assets: ArrayLike,
    initial_human_capital: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    on_asset_floor: ArrayLike | None = None,
) -> np.ndarray:
    """Deposit initial nodes into floor/interior probability compartments.

    Supplied weights must already be probabilities summing to one. No
    renormalization hides conservation errors. Call ``build_transport`` on
    the initial states to inspect initial first-cell remapping if needed.
    """
    assets, human_capital = np.broadcast_arrays(
        np.asarray(initial_assets, dtype=np.float64),
        np.asarray(initial_human_capital, dtype=np.float64),
    )
    if assets.ndim > 1 or assets.size == 0:
        raise ValueError("initial states must be nonempty scalar or one-dimensional arrays")
    assets, human_capital = assets.reshape(-1), human_capital.reshape(-1)
    if not np.all(np.isfinite(human_capital)) or np.any(human_capital <= 0):
        raise ValueError("initial human capital must be finite and positive")
    probability = (
        np.full(assets.size, 1.0 / assets.size)
        if weights is None else np.asarray(weights, dtype=np.float64)
    )
    _validate_mass(probability, (assets.size,))
    stencil = build_transport(
        grid, np.stack((assets, np.log(human_capital)), axis=-1),
        on_asset_floor=on_asset_floor,
    )
    return stencil.push(probability).reshape(grid.shape)


def _validate_mass(masses: np.ndarray, shape: tuple[int, ...]) -> None:
    if masses.shape != shape or not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("probability masses must be finite, nonnegative, and match the grid")
    if abs(float(np.sum(masses)) - 1.0) > 1e-12:
        raise ValueError("probability masses must sum to one; they are not silently normalized")


@lru_cache(maxsize=16)
def _cached_distribution_scan(
    config: BellmanConfig, backend: str, device_index: int, store_snapshots: bool
) -> tuple[Any, Any]:
    recover_policy, _, device = _cached_greedy_kernels(config, backend, device_index)

    @jax.jit
    def rollout(
        params: ModelParams,
        interior_assets: Array,
        log_k_nodes: Array,
        initial_mass: Array,
        continuation_history: Array,
        node_policy_history: Array,
        threshold: Array,
        near_floor_width: Array,
    ) -> Any:
        asset_nodes = jnp.concatenate((jnp.array([config.asset_minimum]), interior_assets))
        assets, log_k = jnp.meshgrid(asset_nodes, log_k_nodes, indexing="ij")
        states = jnp.stack((assets.ravel(), log_k.ravel()), axis=-1)
        state_observables = jnp.stack(
            (states[:, 0], jnp.exp(states[:, 1]), states[:, 1],
             (states[:, 0] == config.asset_minimum).astype(jnp.float64),
             (states[:, 0] >= interior_assets[-2]).astype(jnp.float64),
             (states[:, 1] <= log_k_nodes[1]).astype(jnp.float64),
             (states[:, 1] >= log_k_nodes[-2]).astype(jnp.float64),
             (states[:, 0] <= config.asset_minimum + near_floor_width).astype(jnp.float64)),
            axis=-1
        )
        floor_source = states[:, 0] == config.asset_minimum
        step = params.horizon / config.periods

        def forward_step(carry, inputs):
            mass, running = carry
            continuation, node_policy, terminal = inputs
            value, consumption, hours, training = recover_policy(
                params, states, continuation, node_policy, terminal
            )
            controls = jnp.stack((consumption, hours, training), axis=-1)
            destinations = constant_control_transition(states, controls, params, step)
            capacity = maximum_feasible_consumption(
                states[:, 0], states[:, 1], hours, training, params, step,
                config.asset_minimum, config.path_checkpoints, method=config.asset_feasibility,
            )
            endpoint_capacity = endpoint_consumption_capacity(
                states[:, 0], states[:, 1], hours, training, params, step,
                config.asset_minimum,
            )
            floor_destination = endpoint_floor_contact(
                destinations[:, 0], config.asset_minimum, consumption, endpoint_capacity,
                step * _exprel(params.interest_rate * step),
            )
            indices, weights, valid, violations, lower_remap, remap_error = _transport_stencil(
                interior_assets, log_k_nodes, jnp.asarray(config.asset_minimum),
                destinations, floor_destination,
            )
            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(controls), axis=-1)
            feasible = (
                (consumption > 0) & (consumption >= config.consumption_floor - 1e-8)
                & (hours >= -1e-8) & (hours <= 1 - config.leisure_floor + 1e-8)
                & (training >= -1e-8) & (training <= hours + 1e-8)
                & (capacity - consumption >= -1e-8)
            )
            path_minimum = minimum_assets_during_step(states, controls, params, step)
            occupied = mass > 0
            active_valid = jnp.all(~occupied | (valid & finite & feasible))
            next_running = running & active_valid
            transferred = jnp.zeros_like(mass).at[indices.ravel()].add(
                (mass[:, None] * weights).ravel()
            )
            # Failed steps freeze the distribution. The wrapper raises rather
            # than return moments from a clipped or depleted population.
            next_mass = jnp.where(next_running, transferred, mass)
            earnings = effective_earnings_share(hours, training) * jnp.exp(states[:, 1])
            observables = jnp.stack(
                (hours, (hours > threshold).astype(jnp.float64), training, consumption, earnings),
                axis=-1,
            )
            moments = mass @ jnp.where(occupied[:, None], observables, 0.0)
            exit_mass = jnp.sum(mass[:, None] * (violations > 0), axis=0)
            material_exit_mass = jnp.sum(
                mass[:, None] * (violations > _DOMAIN_TOLERANCE), axis=0
            )
            max_exit = jnp.max(jnp.where(occupied[:, None], violations, 0.0), axis=0)
            row_error = jnp.max(
                jnp.where(occupied & valid, jnp.abs(weights.sum(axis=-1) - 1), 0.0)
            )
            diagnostic = jnp.concatenate((
                exit_mass, material_exit_mass, max_exit,
                jnp.array([
                    row_error, jnp.min(next_mass), jnp.sum(next_mass) - jnp.sum(initial_mass),
                    jnp.sum(mass * lower_remap), mass @ remap_error,
                    jnp.sum(mass * ~floor_source * floor_destination),
                    jnp.sum(mass * floor_source * ~floor_destination),
                    jnp.sum(mass * floor_source * floor_destination),
                    jnp.sum(mass * ~finite), jnp.sum(mass * ~feasible),
                    jnp.min(jnp.where(occupied, capacity - consumption, jnp.inf)),
                    jnp.sum(mass * floor_destination
                            * (destinations[:, 0] != config.asset_minimum)),
                    next_running.astype(jnp.float64),
                    jnp.min(jnp.where(occupied, path_minimum, jnp.inf)),
                    jnp.sum(mass * (path_minimum < config.asset_minimum - _DOMAIN_TOLERANCE)),
                ]),
            ))
            snapshot = mass if store_snapshots else jnp.zeros((0,), dtype=mass.dtype)
            return (next_mass, next_running), (
                moments, mass @ state_observables, diagnostic, snapshot
            )

        (terminal_mass, _), (moments, state_moments, diagnostics, snapshots) = jax.lax.scan(
            forward_step, (initial_mass, jnp.asarray(True)),
            (continuation_history, node_policy_history,
             jnp.arange(config.periods) == config.periods - 1),
        )
        state_moments = jnp.concatenate(
            (state_moments, (terminal_mass @ state_observables)[None, :]), axis=0
        )
        return terminal_mass, moments, state_moments, diagnostics, snapshots

    return rollout, device


def simulate_distribution(
    solution: BellmanSolution,
    grid: DistributionGrid,
    initial_mass: ArrayLike,
    *,
    participation_hours_threshold: float,
    store_snapshots: bool = False,
    near_asset_floor_width: float | None = None,
) -> DistributionSimulation:
    """Evolve probability mass with feasible Bellman-greedy policies in a scan.

    No discounting, mortality, age pooling, or repeated normalization enters
    transport. Moments match ``cohort_moments`` timing. Grid limits must lie
    inside the Bellman domain and the floor must be identical. Material exits
    of positive mass raise ``DistributionDomainError`` carrying diagnostics;
    expand the distribution/Bellman domains and rerun in that case. The
    positive numerical floor and first-cell spreading require convergence.
    A positive ``near_asset_floor_width`` additionally measures mass in the
    closed band from the numerical floor to floor + width in model asset units.
    It includes exact floor mass and does not alter constraint classification.
    ``outer_cell_mass`` reports all-age probability at the nodes of each
    artificial outermost grid cell, including its inner endpoint. Face counts
    may overlap; they diagnose boundary exposure even when the Bellman policy
    rejects outward controls and recorded attempted exits are zero.
    """
    if near_asset_floor_width is not None and (
        not np.isfinite(near_asset_floor_width) or near_asset_floor_width <= 0
    ):
        raise ValueError("near_asset_floor_width must be finite and strictly positive")
    threshold = float(participation_hours_threshold)
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError("participation_hours_threshold must be finite and lie in [0, 1)")
    mass = np.asarray(initial_mass, dtype=np.float64)
    _validate_mass(mass, grid.shape)
    if grid.asset_floor != solution.config.asset_minimum:
        raise ValueError("distribution asset_floor must equal the Bellman numerical asset floor")
    if (
        grid.asset_interior_nodes[-1] > solution.asset_grid[-1]
        or grid.log_human_capital_nodes[0] < solution.log_human_capital_grid[0]
        or grid.log_human_capital_nodes[-1] > solution.log_human_capital_grid[-1]
    ):
        raise ValueError("distribution grid must lie inside the Bellman domain")
    rollout, device = _cached_distribution_scan(
        solution.config, solution.backend, solution.config.device_index, store_snapshots
    )
    params = jax.device_put(
        jax.tree.map(lambda value: np.asarray(value, dtype=np.float64), solution.params), device
    )
    node_policy = np.stack(
        (solution.consumption_policy, solution.hours_policy, solution.training_time_policy), axis=-1
    )
    terminal, moment_array, state_array, checks, snapshots = jax.device_get(rollout(
        params, jax.device_put(grid.asset_interior_nodes, device),
        jax.device_put(grid.log_human_capital_nodes, device),
        jax.device_put(mass.ravel(), device), jax.device_put(solution.values[1:], device),
        jax.device_put(node_policy, device), jax.device_put(np.asarray(threshold), device),
        jax.device_put(np.asarray(near_asset_floor_width or 0.0), device),
    ))
    diagnostics = {
        "exit_face_order": ("asset_floor", "asset_upper", "log_k_lower", "log_k_upper"),
        "outer_cell_face_order": ("asset_upper", "log_k_lower", "log_k_upper"),
        "outer_cell_mass": state_array[:, 4:7],
        "outer_cell_thresholds": {
            "asset_upper_at_least": float(grid.asset_interior_nodes[-2]),
            "log_k_lower_at_most": float(grid.log_human_capital_nodes[1]),
            "log_k_upper_at_least": float(grid.log_human_capital_nodes[-2]),
        },
        "attempted_exit_mass": checks[:, :4],
        "material_exit_mass": checks[:, 4:8],
        "maximum_exit_distance": checks[:, 8:12],
        "maximum_row_sum_error": float(np.max(checks[:, 12])),
        "row_sum_scope": "occupied feasible source rows",
        "minimum_probability_mass": float(np.min(checks[:, 13])),
        "maximum_mass_drift": float(np.max(np.abs(checks[:, 14]))),
        "lower_interior_remap_mass": checks[:, 15],
        "asset_first_moment_remap_error": checks[:, 16],
        "entered_asset_floor_mass": checks[:, 17],
        "left_asset_floor_mass": checks[:, 18],
        "stayed_asset_floor_mass": checks[:, 19],
        "nonfinite_policy_mass": checks[:, 20],
        "infeasible_policy_mass": checks[:, 21],
        "minimum_consumption_capacity_slack": float(np.min(checks[:, 22])),
        "floor_roundoff_corrected_mass": checks[:, 23],
        "completed_periods": int(np.sum(checks[:, 24])),
        "minimum_assets_during_period": float(np.min(checks[:, 25])),
        "maximum_full_period_floor_violation": float(
            max(0.0, grid.asset_floor - np.min(checks[:, 25]))
        ),
        "full_period_floor_violation_mass": checks[:, 26],
        "near_asset_floor_width": near_asset_floor_width,
        "terminal_total_mass": float(np.sum(terminal)),
        "asset_floor": grid.asset_floor,
        "economic_asset_floor": float(solution.params.asset_floor),
        "asset_floor_is_numerical_approximation": grid.asset_floor > solution.params.asset_floor,
        "asset_floor_contact": "binding endpoint within scaled float64 arithmetic budget",
        "asset_floor_contact_error_multiplier": _FLOOR_CONTACT_ULPS,
        "accepted_for_calibration_default": False,
    }
    if not np.all(checks[:, 24]):
        if np.any(checks[:, 4:8] > 0):
            raise DistributionDomainError(
                "positive population mass left the distribution domain; expand the domain "
                "and recheck policies", diagnostics,
            )
        raise DistributionDomainError(
            "forward distribution encountered nonfinite or infeasible occupied policies",
            diagnostics,
        )
    if diagnostics["maximum_mass_drift"] > 1e-10 or diagnostics["minimum_probability_mass"] < 0:
        raise RuntimeError("forward probability conservation or positivity check failed")
    moments = CohortMoments(
        time=solution.time[:-1].copy(), hours=moment_array[:, 0],
        participation=moment_array[:, 1], training_time=moment_array[:, 2],
        consumption=moment_array[:, 3], earnings=moment_array[:, 4],
        participation_hours_threshold=threshold,
    )
    state_moments = DistributionStateMoments(
        time=solution.time.copy(), assets=state_array[:, 0], human_capital=state_array[:, 1],
        log_human_capital=state_array[:, 2], asset_floor_mass=state_array[:, 3],
        near_asset_floor_mass=state_array[:, 7] if near_asset_floor_width is not None else None,
        near_asset_floor_width=near_asset_floor_width,
    )
    full_mass = None
    if store_snapshots:
        full_mass = np.concatenate((snapshots, terminal[None, :]), axis=0).reshape(
            (solution.config.periods + 1, *grid.shape)
        )
    return DistributionSimulation(
        solution, grid, moments, state_moments, diagnostics,
        full_mass, terminal.reshape(grid.shape),
    )
