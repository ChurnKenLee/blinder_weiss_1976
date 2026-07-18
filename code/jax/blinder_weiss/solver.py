"""SciPy constrained optimization driven by JAX derivatives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from cyipopt import minimize_ipopt
from scipy.optimize import Bounds, NonlinearConstraint, OptimizeResult, minimize
from scipy.sparse import coo_array

from .model import ModelParams, benchmark_params, earnings_tradeoff
from .transcription import (
    CollocationScheme,
    DecisionLayout,
    asset_path_slack,
    decision_bounds,
    equality_constraints,
    interpolate_decision,
    lifetime_utility,
    make_initial_guess,
    scaled_negative_objective,
    training_time_slack,
)


@dataclass(frozen=True)
class SolverConfig:
    """Numerical settings for one direct-collocation solve."""

    intervals: int = 24
    scheme: CollocationScheme = "hermite-simpson"
    optimizer: str = "SLSQP"
    max_iterations: int = 2_000
    slsqp_restarts: int = 1
    objective_tolerance: float = 1e-9
    constraint_tolerance: float = 1e-7
    consumption_floor: float = 1e-8
    leisure_floor: float = 1e-8
    terminal_asset_floor: float = 1e-8
    display: bool = False


@dataclass(frozen=True)
class CollocationResult:
    """Economic paths and solver metadata from one mesh."""

    params: ModelParams
    config: SolverConfig
    time: np.ndarray
    decision: np.ndarray
    assets: np.ndarray
    log_human_capital: np.ndarray
    human_capital: np.ndarray
    consumption: np.ndarray
    hours: np.ndarray
    training_time: np.ndarray
    investment_share: np.ndarray
    lifetime_utility: float
    max_constraint_violation: float
    success: bool
    optimizer_success: bool
    status: int
    message: str
    iterations: int
    optimizer_result: OptimizeResult

    @property
    def leisure(self) -> np.ndarray:
        return 1.0 - self.hours

    @property
    def wage(self) -> np.ndarray:
        tradeoff = np.asarray(earnings_tradeoff(jnp.asarray(self.investment_share)))
        return tradeoff * self.human_capital

    @property
    def labor_earnings(self) -> np.ndarray:
        return self.hours * np.asarray(self.wage)


def _validate_inputs(params: ModelParams, config: SolverConfig) -> None:
    if config.intervals < 2:
        raise ValueError("At least two collocation intervals are required")
    if config.scheme not in ("trapezoid", "hermite-simpson"):
        raise ValueError(f"Unknown collocation scheme: {config.scheme}")
    if config.optimizer not in ("ipopt", "SLSQP", "trust-constr"):
        raise ValueError("optimizer must be 'ipopt', 'SLSQP', or 'trust-constr'")
    if config.slsqp_restarts < 0:
        raise ValueError("slsqp_restarts cannot be negative")
    if params.horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if params.initial_human_capital <= 0.0:
        raise ValueError("initial_human_capital must be positive")
    if params.initial_assets < params.asset_floor:
        raise ValueError("initial_assets cannot be below asset_floor")
    for name in ("consumption_power", "leisure_power", "bequest_power"):
        if getattr(params, name) == 0.0:
            raise ValueError(f"{name} cannot be zero; use a small nonzero value")


def _jax_scipy_functions(
    params: ModelParams, config: SolverConfig
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    objective_jax = jax.jit(
        lambda decision: scaled_negative_objective(
            decision,
            params,
            config.intervals,
            config.scheme,
        )
    )
    gradient_jax = jax.jit(jax.grad(objective_jax))
    constraints_jax = jax.jit(
        lambda decision: equality_constraints(decision, params, config.intervals, config.scheme)
    )
    constraint_jacobian_jax = jax.jit(jax.jacrev(constraints_jax))
    training_slack_jax = jax.jit(lambda decision: training_time_slack(decision, config.intervals))
    training_slack_jacobian_jax = jax.jit(jax.jacrev(training_slack_jax))
    path_slack_jax = jax.jit(
        lambda decision: asset_path_slack(
            decision,
            params,
            config.intervals,
            config.scheme,
        )
    )
    path_slack_jacobian_jax = jax.jit(jax.jacrev(path_slack_jax))

    def objective_np(decision: np.ndarray) -> float:
        return float(objective_jax(jnp.asarray(decision)))

    def gradient_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(gradient_jax(jnp.asarray(decision)), dtype=np.float64)

    def constraints_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(constraints_jax(jnp.asarray(decision)), dtype=np.float64)

    def constraint_jacobian_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(constraint_jacobian_jax(jnp.asarray(decision)), dtype=np.float64)

    def training_slack_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(training_slack_jax(jnp.asarray(decision)), dtype=np.float64)

    def training_slack_jacobian_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(training_slack_jacobian_jax(jnp.asarray(decision)), dtype=np.float64)

    def path_slack_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(path_slack_jax(jnp.asarray(decision)), dtype=np.float64)

    def path_slack_jacobian_np(decision: np.ndarray) -> np.ndarray:
        return np.asarray(path_slack_jacobian_jax(jnp.asarray(decision)), dtype=np.float64)

    return (
        objective_np,
        gradient_np,
        constraints_np,
        constraint_jacobian_np,
        training_slack_np,
        training_slack_jacobian_np,
        path_slack_np,
        path_slack_jacobian_np,
    )


def _constraint_sparsity(intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Conservative local sparsity pattern for the collocation Jacobian."""

    layout = DecisionLayout(intervals)
    rows = [0, 1]
    columns = [0, 1]
    for interval in range(intervals):
        state_columns = [
            2 * interval,
            2 * interval + 1,
            2 * (interval + 1),
            2 * (interval + 1) + 1,
        ]
        control_offset = layout.state_size
        control_columns = [
            control_offset + 3 * node + control
            for node in (interval, interval + 1)
            for control in range(3)
        ]
        local_columns = state_columns + control_columns
        for equation in range(2):
            row = 2 + 2 * interval + equation
            rows.extend([row] * len(local_columns))
            columns.extend(local_columns)
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)


def _path_constraint_sparsity(intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Conservative local sparsity pattern for midpoint asset constraints."""

    layout = DecisionLayout(intervals)
    rows: list[int] = []
    columns: list[int] = []
    for interval in range(intervals):
        state_columns = [
            2 * interval,
            2 * interval + 1,
            2 * (interval + 1),
            2 * (interval + 1) + 1,
        ]
        control_columns = [
            layout.state_size + 3 * node + control
            for node in (interval, interval + 1)
            for control in range(3)
        ]
        local_columns = state_columns + control_columns
        rows.extend([interval] * len(local_columns))
        columns.extend(local_columns)
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)


def solve_lifecycle(
    params: ModelParams | None = None,
    config: SolverConfig | None = None,
    *,
    initial_decision: np.ndarray | None = None,
) -> CollocationResult:
    """Solve one deterministic lifecycle path by direct collocation."""

    params = benchmark_params() if params is None else params
    config = SolverConfig() if config is None else config
    _validate_inputs(params, config)
    layout = DecisionLayout(config.intervals)

    if initial_decision is None:
        initial_decision = make_initial_guess(params, config.intervals)
    initial_decision = np.asarray(initial_decision, dtype=np.float64)
    if initial_decision.shape != (layout.size,):
        raise ValueError(
            f"initial_decision must have shape {(layout.size,)}, got {initial_decision.shape}"
        )

    lower, upper = decision_bounds(
        params,
        config.intervals,
        consumption_floor=config.consumption_floor,
        leisure_floor=config.leisure_floor,
        terminal_asset_floor=config.terminal_asset_floor,
    )
    initial_decision = np.clip(initial_decision, lower, upper)
    (
        objective_np,
        gradient_np,
        constraints_np,
        constraint_jacobian_np,
        training_slack_np,
        training_slack_jacobian_np,
        path_slack_np,
        path_slack_jacobian_np,
    ) = _jax_scipy_functions(params, config)

    sparsity_rows, sparsity_columns = _constraint_sparsity(config.intervals)
    path_rows, path_columns = _path_constraint_sparsity(config.intervals)

    def constraint_jacobian_sparse_np(decision: np.ndarray) -> coo_array:
        dense_jacobian = constraint_jacobian_np(decision)
        values = dense_jacobian[sparsity_rows, sparsity_columns]
        return coo_array(
            (values, (sparsity_rows, sparsity_columns)),
            shape=dense_jacobian.shape,
        )

    training_rows = np.arange(layout.nodes, dtype=np.int32)
    training_hours_columns = layout.state_size + 3 * training_rows + 1
    training_time_columns = layout.state_size + 3 * training_rows + 2
    training_columns = np.concatenate((training_hours_columns, training_time_columns))
    repeated_training_rows = np.concatenate((training_rows, training_rows))

    def training_slack_jacobian_sparse_np(decision: np.ndarray) -> coo_array:
        dense_jacobian = training_slack_jacobian_np(decision)
        values = dense_jacobian[repeated_training_rows, training_columns]
        return coo_array(
            (values, (repeated_training_rows, training_columns)),
            shape=dense_jacobian.shape,
        )

    def path_slack_jacobian_sparse_np(decision: np.ndarray) -> coo_array:
        dense_jacobian = path_slack_jacobian_np(decision)
        values = dense_jacobian[path_rows, path_columns]
        return coo_array(
            (values, (path_rows, path_columns)),
            shape=dense_jacobian.shape,
        )

    # Trigger compilation outside SciPy so compilation time is not confused
    # with a slow optimizer evaluation in notebook diagnostics.
    objective_np(initial_decision)
    gradient_np(initial_decision)
    constraints_np(initial_decision)
    constraint_jacobian_np(initial_decision)
    training_slack_np(initial_decision)
    training_slack_jacobian_np(initial_decision)
    path_slack_np(initial_decision)
    path_slack_jacobian_np(initial_decision)
    # SciPy accepts array-like bounds; its untyped source is inferred too narrowly
    # by basedpyright because the constructor defaults are scalar infinities.
    variable_bounds = Bounds(lower, upper)  # pyright: ignore[reportArgumentType]

    if config.optimizer == "ipopt":
        constraints = [
            {
                "type": "eq",
                "fun": constraints_np,
                "jac": constraint_jacobian_sparse_np,
            },
            {
                "type": "ineq",
                "fun": training_slack_np,
                "jac": training_slack_jacobian_sparse_np,
            },
            {
                "type": "ineq",
                "fun": path_slack_np,
                "jac": path_slack_jacobian_sparse_np,
            },
        ]
        optimizer_result = minimize_ipopt(
            objective_np,
            initial_decision,
            jac=gradient_np,
            bounds=variable_bounds,
            constraints=constraints,
            tol=config.objective_tolerance,
            options={
                "acceptable_constr_viol_tol": config.constraint_tolerance,
                "acceptable_tol": max(config.objective_tolerance, 1e-8),
                "bound_relax_factor": 0.0,
                "disp": config.display,
                "hessian_approximation": "limited-memory",
                "honor_original_bounds": "yes",
                "maxiter": config.max_iterations,
                "mu_strategy": "adaptive",
            },
        )
    elif config.optimizer == "SLSQP":
        constraints: Any = [
            {
                "type": "eq",
                "fun": constraints_np,
                "jac": constraint_jacobian_np,
            },
            {
                "type": "ineq",
                "fun": training_slack_np,
                "jac": training_slack_jacobian_np,
            },
            {
                "type": "ineq",
                "fun": path_slack_np,
                "jac": path_slack_jacobian_np,
            },
        ]
        options = {
            "disp": config.display,
            "ftol": config.objective_tolerance,
            "maxiter": config.max_iterations,
        }
        optimizer_result = minimize(
            objective_np,
            initial_decision,
            method=config.optimizer,
            jac=gradient_np,
            bounds=variable_bounds,
            constraints=constraints,
            options=options,
        )
        total_iterations = int(getattr(optimizer_result, "nit", 0))
        # SLSQP can terminate on a tiny objective change while its BFGS model
        # still gives poor stationarity at a newly active path constraint. A
        # restart resets that model and is an inexpensive, effective polish.
        for _ in range(config.slsqp_restarts):
            if not optimizer_result.success:
                break
            restart_result = minimize(
                objective_np,
                np.asarray(optimizer_result.x),
                method=config.optimizer,
                jac=gradient_np,
                bounds=variable_bounds,
                constraints=constraints,
                options=options,
            )
            total_iterations += int(getattr(restart_result, "nit", 0))
            if (
                restart_result.success
                and float(restart_result.fun) <= float(optimizer_result.fun) + 1e-10
            ):
                optimizer_result = restart_result
            else:
                break
        optimizer_result.nit = total_iterations
    else:
        constraints = [
            NonlinearConstraint(
                constraints_np,
                lb=0.0,
                ub=0.0,
                jac=constraint_jacobian_np,
            ),
            NonlinearConstraint(
                training_slack_np,
                lb=0.0,
                ub=np.inf,
                jac=training_slack_jacobian_np,
            ),
            NonlinearConstraint(
                path_slack_np,
                lb=0.0,
                ub=np.inf,
                jac=path_slack_jacobian_np,
            ),
        ]
        options = {
            "barrier_tol": config.objective_tolerance,
            "disp": config.display,
            "gtol": config.objective_tolerance,
            "maxiter": config.max_iterations,
            "xtol": config.objective_tolerance,
        }

        optimizer_result = minimize(
            objective_np,
            initial_decision,
            method=config.optimizer,
            jac=gradient_np,
            bounds=variable_bounds,
            constraints=constraints,
            options=options,
        )

    decision = np.asarray(optimizer_result.x, dtype=np.float64)
    states_jax, controls_jax = layout.unpack(jnp.asarray(decision))
    states = np.asarray(states_jax)
    controls = np.asarray(controls_jax)
    constraint_residual = constraints_np(decision)
    minimum_training_slack = float(np.min(training_slack_np(decision)))
    minimum_path_slack = float(np.min(path_slack_np(decision)))
    max_constraint_violation = max(
        float(np.max(np.abs(constraint_residual))),
        max(0.0, -minimum_training_slack),
        max(0.0, -minimum_path_slack),
    )
    optimizer_success = bool(optimizer_result.success)
    success = optimizer_success and max_constraint_violation <= config.constraint_tolerance
    utility = float(
        lifetime_utility(jnp.asarray(decision), params, config.intervals, config.scheme)
    )

    training_time = controls[:, 2]
    investment_share = np.divide(
        training_time,
        controls[:, 1],
        out=np.zeros_like(training_time),
        where=controls[:, 1] > 1e-10,
    )
    investment_share = np.clip(investment_share, 0.0, 1.0)

    raw_message = optimizer_result.message
    message = (
        raw_message.decode("utf-8", errors="replace")
        if isinstance(raw_message, bytes)
        else str(raw_message)
    )

    return CollocationResult(
        params=params,
        config=config,
        time=np.linspace(0.0, params.horizon, layout.nodes),
        decision=decision,
        assets=states[:, 0],
        log_human_capital=states[:, 1],
        human_capital=np.exp(states[:, 1]),
        consumption=controls[:, 0],
        hours=controls[:, 1],
        training_time=training_time,
        investment_share=investment_share,
        lifetime_utility=utility,
        max_constraint_violation=max_constraint_violation,
        success=success,
        optimizer_success=optimizer_success,
        status=int(optimizer_result.status),
        message=message,
        iterations=int(getattr(optimizer_result, "nit", -1)),
        optimizer_result=optimizer_result,
    )


def solve_mesh_sequence(
    intervals: tuple[int, ...] | list[int],
    params: ModelParams | None = None,
    config: SolverConfig | None = None,
) -> list[CollocationResult]:
    """Solve coarse-to-fine meshes, interpolating each path as the next start."""

    if not intervals:
        raise ValueError("intervals cannot be empty")
    if any(next_n <= current_n for current_n, next_n in pairwise(intervals)):
        raise ValueError("intervals must be strictly increasing")

    params = benchmark_params() if params is None else params
    base_config = SolverConfig() if config is None else config
    results: list[CollocationResult] = []
    warm_start: np.ndarray | None = None
    old_intervals: int | None = None

    for mesh_intervals in intervals:
        mesh_config = replace(base_config, intervals=mesh_intervals)
        if warm_start is not None and old_intervals is not None:
            warm_start = interpolate_decision(
                warm_start, old_intervals, mesh_intervals, params.horizon
            )
        result = solve_lifecycle(params, mesh_config, initial_decision=warm_start)
        results.append(result)
        warm_start = result.decision
        old_intervals = mesh_intervals

    return results
