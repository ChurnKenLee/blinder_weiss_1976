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
