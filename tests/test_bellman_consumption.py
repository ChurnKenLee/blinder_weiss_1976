from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import benchmark_params, constant_control_transition
from blinder_weiss.bellman import _interpolate_numpy, maximum_feasible_consumption
from blinder_weiss.bellman_consumption import optimize_conditional_consumption
from scipy.optimize import minimize_scalar


def test_conditional_consumption_matches_global_piecewise_reference() -> None:
    """A nonconcave continuation table must not reduce to one local search."""

    params = benchmark_params()
    asset_grid = np.array([0.001, 0.15, 0.7, 1.8, 4.0, 10.0, 30.0])
    log_grid = np.array([-1.0, 0.0, 1.0])
    # Include increasing slopes and a negative slope, so a concavity-based
    # binary search or a single scalar minimization would not be sufficient.
    asset_values = np.array([-20.0, -6.0, -4.0, -4.5, -1.0, -0.5, -0.05])
    continuation = asset_values[:, None] + np.array([-0.3, 0.0, 0.2])[None, :]
    states = np.array([[0.03, -0.5], [0.6, 0.2], [2.0, 0.0], [7.0, 0.4]])
    hours = np.array([0.6, 0.5, 0.8, 0.2])
    training = np.array([0.2, 0.0, 0.5, 0.0])
    duration = 0.65
    floor = 1e-8
    optimizer = jax.jit(partial(
        optimize_conditional_consumption,
        step=duration,
        asset_minimum=asset_grid[0],
        consumption_floor=floor,
        path_checkpoints=8,
    ))
    consumption, values = optimizer(
        jnp.asarray(states), jnp.asarray(hours), jnp.asarray(training),
        jnp.asarray(continuation), jnp.asarray(asset_grid), jnp.asarray(log_grid), params,
    )
    beta = np.exp(-params.rho * duration)
    flow_discount = -np.expm1(-params.rho * duration) / params.rho
    consumption_factor = np.expm1(params.interest_rate * duration) / params.interest_rate

    for index, state in enumerate(states):
        cap = float(maximum_feasible_consumption(
            jnp.asarray(state[0]), jnp.asarray(state[1]), jnp.asarray(hours[index]),
            jnp.asarray(training[index]), params, duration, asset_grid[0], 8,
        ))
        zero_consumption_state = np.asarray(constant_control_transition(
            jnp.asarray(state), jnp.asarray([0.0, hours[index], training[index]]),
            params, duration,
        ))
        lower = max(floor, (zero_consumption_state[0] - asset_grid[-1]) / consumption_factor)
        upper = min(cap, (zero_consumption_state[0] - asset_grid[0]) / consumption_factor)

        def objective(candidate, zero_consumption_state=zero_consumption_state, index=index):
            next_assets = zero_consumption_state[0] - consumption_factor * candidate
            next_value = float(_interpolate_numpy(
                continuation, asset_grid, log_grid, next_assets, zero_consumption_state[1]
            ))
            return flow_discount * (
                params.consumption_weight * candidate**params.consumption_power
                / params.consumption_power
                + params.leisure_weight * (1.0 - hours[index])**params.leisure_power
                / params.leisure_power
            ) + beta * next_value

        boundaries = np.clip(
            (zero_consumption_state[0] - asset_grid) / consumption_factor, lower, upper
        )
        boundaries = np.unique(np.concatenate(([lower, upper], boundaries)))
        candidate_values = [objective(candidate) for candidate in boundaries]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            result = minimize_scalar(
                lambda candidate: -objective(candidate), bounds=(left, right),
                method="bounded", options={"xatol": 1e-13},
            )
            candidate_values.append(-result.fun)
        assert float(values[index]) == pytest.approx(max(candidate_values), abs=2e-10)
        assert float(values[index]) == pytest.approx(
            objective(float(consumption[index])), abs=2e-10
        )
        assert lower - 1e-10 <= float(consumption[index]) <= upper + 1e-10


def test_terminal_consumption_is_analytic_and_parameters_are_dynamic() -> None:
    params = benchmark_params()
    duration = 0.005
    states = jnp.asarray([[35.0, 0.0], [5.0, 0.2]])
    hours = jnp.zeros(2)
    training = jnp.zeros(2)
    asset_grid = jnp.asarray([1e-4, 1.0, 40.0])
    log_grid = jnp.asarray([-1.0, 0.0, 1.0])
    # Terminal consumption must use the exact bequest rather than this table.
    continuation = jnp.full((3, 3), 1e6)
    traces: list[None] = []

    @jax.jit
    def solve(dynamic_params):
        traces.append(None)
        return optimize_conditional_consumption(
            states, hours, training, continuation, asset_grid, log_grid, dynamic_params,
            step=duration, asset_minimum=asset_grid[0], consumption_floor=1e-8,
            path_checkpoints=4, continuation_is_terminal=True,
            incumbent_consumption=jnp.ones(2),
        )

    for consumption_weight in (1.0, 2.0):
        dynamic_params = params._replace(consumption_weight=consumption_weight)
        consumption, value = solve(dynamic_params)
        beta = np.exp(-params.rho * duration)
        flow_discount = -np.expm1(-params.rho * duration) / params.rho
        factor = np.expm1(params.interest_rate * duration) / params.interest_rate
        wealth = np.asarray(states[:, 0]) * np.exp(params.interest_rate * duration)
        expected = wealth / (
            factor + np.sqrt(beta * factor * params.bequest_weight
                             / (flow_discount * consumption_weight))
        )
        np.testing.assert_allclose(consumption, expected, rtol=1e-11)
        expected_value = flow_discount * (-consumption_weight / expected - 1.0) - beta / (
            wealth - factor * expected
        )
        np.testing.assert_allclose(value, expected_value, rtol=1e-12)
        # At this time step the old 2% capacity floor excluded the true optimum.
        assert np.all(expected < 0.02 * (wealth - float(asset_grid[0])) / factor)
    assert len(traces) == 1


def test_consumption_respects_path_constraint_and_retains_incumbent() -> None:
    params = benchmark_params()
    asset_grid = jnp.asarray([1e-4, 0.01, 0.1, 1.0, 10.0])
    log_grid = jnp.asarray([-1.0, 0.0, 1.0])
    continuation = -1.0 / asset_grid[:, None] + jnp.zeros((5, 3))
    states = jnp.asarray([[[0.005, 0.0], [1.0, 0.2]]])
    hours = jnp.asarray([[0.8, 0.5]])
    training = jnp.asarray([[0.6, 0.1]])
    arguments = (states, hours, training, continuation, asset_grid, log_grid, params)
    optimizer = partial(
        optimize_conditional_consumption,
        step=2.0, asset_minimum=1e-4, consumption_floor=1e-8, path_checkpoints=16,
    )
    consumption, value = optimizer(*arguments)
    repeated_consumption, repeated_value = optimizer(
        *arguments, incumbent_consumption=consumption
    )
    assert consumption.shape == (1, 2)
    assert np.all(np.asarray(repeated_value) >= np.asarray(value))
    np.testing.assert_allclose(repeated_consumption, consumption, atol=1e-12)
    controls = jnp.stack((consumption, hours, training), axis=-1)
    for duration in np.linspace(2.0 / 16, 2.0, 16):
        next_state = constant_control_transition(states, controls, params, float(duration))
        assert np.min(np.asarray(next_state[..., 0])) >= float(asset_grid[0]) - 1e-10


def test_conditional_consumption_rejects_unavoidable_domain_exit() -> None:
    params = benchmark_params()
    consumption, value = optimize_conditional_consumption(
        jnp.asarray([[1.0, -1.0]]), jnp.asarray([0.0]), jnp.asarray([0.0]),
        jnp.zeros((3, 3)), jnp.asarray([1e-4, 1.0, 10.0]), jnp.asarray([-1.0, 0.0, 1.0]),
        params, step=1.0, asset_minimum=1e-4, consumption_floor=1e-8, path_checkpoints=4,
    )
    assert np.isfinite(float(consumption[0]))
    assert np.isneginf(float(value[0]))
