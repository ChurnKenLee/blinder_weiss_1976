"""Regression checks for reusable compilation during parameter calibration."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blinder_weiss import BellmanConfig, bellman, benchmark_params, bequest_utility, solve_bellman


def _small_config() -> BellmanConfig:
    return BellmanConfig(
        periods=1,
        asset_nodes=5,
        human_capital_nodes=4,
        asset_minimum=1e-3,
        asset_maximum=12.0,
        log_human_capital_minimum=-1.0,
        log_human_capital_maximum=1.0,
        hours_nodes=3,
        investment_nodes=3,
        consumption_nodes=4,
        refinement_steps=1,
        refinement_backtracking_steps=2,
        neighbor_policy_sweeps=1,
        compute_platform="cpu",
    )


def test_parameter_changes_reuse_trace_and_update_solution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _small_config()
    params = benchmark_params(horizon=1.0)
    changed_params = params._replace(
        horizon=np.float64(1.1),
        rho=np.float64(0.031),
        human_capital_productivity=np.float64(0.225),
        consumption_weight=np.float64(1.01),
        leisure_weight=np.float64(1.02),
        bequest_weight=np.float64(1.03),
        initial_assets=5,
    )
    original_make_step = bellman._make_bellman_step
    traces = []

    def counted_make_step(*args, **kwargs):
        traces.append(None)
        return original_make_step(*args, **kwargs)

    bellman._cached_backward_solver.cache_clear()
    monkeypatch.setattr(bellman, "_make_bellman_step", counted_make_step)
    baseline = solve_bellman(params, config)
    changed = solve_bellman(changed_params, config)
    repeated = solve_bellman(params, config)

    assert len(traces) == 1
    assert bellman._cached_backward_solver.cache_info().hits == 2
    assert changed.params == changed_params
    assert changed.backend == "cpu"
    assert changed.time[-1] == changed_params.horizon
    assert not np.allclose(changed.values[0], baseline.values[0])
    np.testing.assert_array_equal(repeated.values, baseline.values)

    # A separate one-period solve closes over the changed parameters. This
    # reference detects any parameter that was accidentally frozen by caching.
    asset_grid = jnp.asarray(changed.asset_grid)
    log_human_capital_grid = jnp.asarray(changed.log_human_capital_grid)
    terminal_assets = jnp.broadcast_to(
        asset_grid[:, None], (config.asset_nodes, config.human_capital_nodes)
    )
    terminal_values = bequest_utility(terminal_assets, changed_params)
    step = original_make_step(changed_params, config, asset_grid, log_human_capital_grid)
    reference = jax.jit(step)(
        terminal_values,
        jnp.zeros((config.asset_nodes, config.human_capital_nodes, 3)),
        jnp.asarray(False),
    )
    for actual, expected in zip(
        (
            changed.values[0],
            changed.consumption_policy[0],
            changed.hours_policy[0],
            changed.training_time_policy[0],
        ),
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(changed.values[-1], terminal_values, rtol=1e-12)


def test_static_configuration_changes_get_distinct_kernels() -> None:
    config = _small_config()
    device = jax.devices("cpu")[0]
    kernel = bellman._cached_backward_solver(config, device)
    assert bellman._cached_backward_solver(replace(config), device) is kernel
    assert bellman._cached_backward_solver(replace(config, periods=2), device) is not kernel
    assert bellman._cached_backward_solver(replace(config, asset_nodes=6), device) is not kernel


def test_greedy_recovery_and_rollout_reuse_parameter_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_small_config(), periods=2)
    baseline_params = benchmark_params(horizon=1.0)
    changed_params = baseline_params._replace(
        horizon=np.float64(1.1),
        interest_rate=np.float64(0.055),
        human_capital_productivity=np.float64(0.225),
        leisure_weight=np.float64(1.02),
        bequest_weight=np.float64(1.03),
        initial_assets=5,
    )
    baseline = solve_bellman(baseline_params, config)
    changed = solve_bellman(changed_params, config)
    original_optimizer = bellman._make_control_optimizer
    traces = []

    def counted_optimizer(*args, **kwargs):
        traces.append(None)
        return original_optimizer(*args, **kwargs)

    bellman._cached_greedy_kernels.cache_clear()
    monkeypatch.setattr(bellman, "_make_control_optimizer", counted_optimizer)
    query_assets = np.asarray([2.0, 5.0])
    query_human_capital = np.asarray([0.9, 1.1])
    baseline_policy = bellman.greedy_policy_value_at(
        baseline, 0, query_assets, query_human_capital
    )
    changed_policy = bellman.greedy_policy_value_at(
        changed, 0, query_assets, query_human_capital
    )
    repeated_policy = bellman.greedy_policy_value_at(
        baseline, 0, query_assets, query_human_capital
    )
    assert len(traces) == 1
    assert not np.allclose(changed_policy[0], baseline_policy[0])
    for actual, expected in zip(repeated_policy, baseline_policy, strict=True):
        np.testing.assert_array_equal(actual, expected)

    baseline_rollout = bellman.simulate_policy(baseline)
    changed_rollout = bellman.simulate_policy(changed)
    repeated_rollout = bellman.simulate_policy(baseline)
    assert len(traces) == 2
    np.testing.assert_array_equal(repeated_rollout.assets, baseline_rollout.assets)
    assert not np.allclose(changed_rollout.assets, baseline_rollout.assets)
    assert changed_rollout.stayed_in_domain

    # Independently create the optimizer with changed parameters as constants.
    # Both cached entry points must agree with this uncached reference.
    optimizer = original_optimizer(
        changed_params,
        config,
        jnp.asarray(changed.asset_grid),
        jnp.asarray(changed.log_human_capital_grid),
    )

    def reference_policy(states, continuation, incumbent, terminal):
        return optimizer(states, continuation, incumbent, True, terminal)

    reference_policy = jax.jit(reference_policy)
    query_states = jnp.stack((query_assets, np.log(query_human_capital)), axis=-1)
    query_incumbent = jnp.stack(
        bellman.policy_at(changed, 0, query_assets, query_human_capital), axis=-1
    )
    reference = reference_policy(
        query_states, changed.values[1], query_incumbent, jnp.asarray(False)
    )
    for actual, expected in zip(changed_policy, reference, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    state = jnp.asarray(
        [changed_params.initial_assets, np.log(changed_params.initial_human_capital)]
    )
    for period in range(config.periods):
        incumbent = jnp.stack(
            bellman.policy_at(changed, period, float(state[0]), float(jnp.exp(state[1]))),
            axis=-1,
        )
        _, consumption, hours, training = reference_policy(
            state,
            changed.values[period + 1],
            incumbent,
            jnp.asarray(period == config.periods - 1),
        )
        control = jnp.stack((consumption, hours, training))
        state = bellman.constant_control_transition(
            state, control, changed_params, changed_params.horizon / config.periods
        )
        np.testing.assert_allclose(
            [
                changed_rollout.consumption[period],
                changed_rollout.hours[period],
                changed_rollout.training_time[period],
            ],
            control,
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            [changed_rollout.assets[period + 1], changed_rollout.log_human_capital[period + 1]],
            state,
            rtol=1e-11,
            atol=1e-11,
        )
